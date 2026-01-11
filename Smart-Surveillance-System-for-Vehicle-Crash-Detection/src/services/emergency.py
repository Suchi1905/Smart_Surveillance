"""
Emergency Dispatcher Service.

Handles multi-channel emergency alert dispatching including:
- Telegram notifications
- SMS alerts (via Twilio)
- Webhook calls
- Email notifications
- Audio/visual alerts for control rooms
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any
from enum import Enum
from datetime import datetime
import logging
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Available alert channels."""
    TELEGRAM = "telegram"
    SMS = "sms"
    WEBHOOK = "webhook"
    EMAIL = "email"
    CONTROL_ROOM = "control_room"
    EMERGENCY_SERVICES = "emergency_services"


class IncidentSeverity(Enum):
    """Incident severity levels."""
    INFO = "info"
    WARNING = "warning"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class Incident:
    """
    Represents a detected incident requiring dispatch.
    
    Attributes:
        incident_id: Unique identifier
        incident_type: Type of incident (crash, near_miss, etc.)
        severity: Severity level
        location: GPS coordinates or camera location
        camera_id: Source camera identifier
        timestamp: Detection time
        description: Human-readable description
        confidence: Detection confidence
        video_clip_path: Path to extracted video clip
        thumbnail_path: Path to incident thumbnail
        vehicle_info: Information about vehicles involved
        additional_data: Any extra data
    """
    incident_id: str
    incident_type: str
    severity: IncidentSeverity
    location: Dict[str, Any]
    camera_id: str
    timestamp: float
    description: str
    confidence: float
    video_clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    vehicle_info: List[Dict] = field(default_factory=list)
    additional_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "incident_id": self.incident_id,
            "incident_type": self.incident_type,
            "severity": self.severity.value,
            "location": self.location,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "timestamp_formatted": datetime.fromtimestamp(self.timestamp).isoformat(),
            "description": self.description,
            "confidence": self.confidence,
            "video_clip_path": self.video_clip_path,
            "thumbnail_path": self.thumbnail_path,
            "vehicle_info": self.vehicle_info,
            "additional_data": self.additional_data
        }


@dataclass
class DispatchResult:
    """Result of a dispatch attempt."""
    channel: AlertChannel
    success: bool
    message: str
    response_time_ms: float
    error: Optional[str] = None


class EmergencyDispatcher:
    """
    Multi-channel emergency alert dispatcher.
    
    Handles routing alerts to appropriate channels based on severity
    and configured escalation rules.
    """
    
    # Severity-based channel routing
    DEFAULT_ROUTING = {
        IncidentSeverity.CRITICAL: [
            AlertChannel.EMERGENCY_SERVICES,
            AlertChannel.CONTROL_ROOM,
            AlertChannel.TELEGRAM,
            AlertChannel.SMS
        ],
        IncidentSeverity.SEVERE: [
            AlertChannel.CONTROL_ROOM,
            AlertChannel.TELEGRAM,
            AlertChannel.WEBHOOK
        ],
        IncidentSeverity.MODERATE: [
            AlertChannel.TELEGRAM,
            AlertChannel.WEBHOOK
        ],
        IncidentSeverity.WARNING: [
            AlertChannel.WEBHOOK
        ],
        IncidentSeverity.INFO: []
    }
    
    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        webhook_urls: Optional[List[str]] = None,
        sms_config: Optional[Dict] = None,
        email_config: Optional[Dict] = None
    ):
        """
        Initialize emergency dispatcher.
        
        Args:
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID for alerts
            webhook_urls: List of webhook URLs for notifications
            sms_config: SMS configuration (Twilio credentials)
            email_config: Email configuration (SMTP settings)
        """
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.webhook_urls = webhook_urls or []
        self.sms_config = sms_config or {}
        self.email_config = email_config or {}
        
        # Custom routing rules
        self.routing_rules = self.DEFAULT_ROUTING.copy()
        
        # Dispatch history
        self._dispatch_history: List[Dict] = []
        
        # Callbacks for control room alerts
        self._control_room_callback: Optional[Callable] = None
        
        # Rate limiting
        self._last_dispatch_time: Dict[str, float] = {}
        self._min_dispatch_interval = 10.0  # seconds between similar alerts
        
        logger.info("EmergencyDispatcher initialized")
    
    async def dispatch(
        self,
        incident: Incident,
        channels: Optional[List[AlertChannel]] = None
    ) -> List[DispatchResult]:
        """
        Dispatch incident alert to configured channels.
        
        Args:
            incident: Incident to dispatch
            channels: Override channels (uses routing rules if None)
        
        Returns:
            List of DispatchResult for each attempted channel
        """
        # Determine channels based on severity if not specified
        if channels is None:
            channels = self.routing_rules.get(incident.severity, [])
        
        if not channels:
            logger.info(f"No channels configured for severity {incident.severity.value}")
            return []
        
        # Check rate limiting
        rate_key = f"{incident.incident_type}_{incident.camera_id}"
        if not self._check_rate_limit(rate_key):
            logger.warning(f"Rate limited: {rate_key}")
            return [DispatchResult(
                channel=AlertChannel.CONTROL_ROOM,
                success=False,
                message="Rate limited",
                response_time_ms=0,
                error="Too many similar alerts"
            )]
        
        # Dispatch to all channels concurrently
        tasks = []
        for channel in channels:
            task = self._dispatch_to_channel(incident, channel)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        dispatch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                dispatch_results.append(DispatchResult(
                    channel=channels[i],
                    success=False,
                    message="Exception occurred",
                    response_time_ms=0,
                    error=str(result)
                ))
            else:
                dispatch_results.append(result)
        
        # Record dispatch
        self._record_dispatch(incident, dispatch_results)
        
        return dispatch_results
    
    async def _dispatch_to_channel(
        self,
        incident: Incident,
        channel: AlertChannel
    ) -> DispatchResult:
        """Dispatch to a specific channel."""
        start_time = time.time()
        
        try:
            if channel == AlertChannel.TELEGRAM:
                result = await self._send_telegram(incident)
            elif channel == AlertChannel.WEBHOOK:
                result = await self._send_webhook(incident)
            elif channel == AlertChannel.SMS:
                result = await self._send_sms(incident)
            elif channel == AlertChannel.EMAIL:
                result = await self._send_email(incident)
            elif channel == AlertChannel.CONTROL_ROOM:
                result = await self._alert_control_room(incident)
            elif channel == AlertChannel.EMERGENCY_SERVICES:
                result = await self._alert_emergency_services(incident)
            else:
                result = (False, "Unknown channel")
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return DispatchResult(
                channel=channel,
                success=result[0],
                message=result[1],
                response_time_ms=elapsed_ms
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Dispatch error for {channel.value}: {e}")
            return DispatchResult(
                channel=channel,
                success=False,
                message="Dispatch failed",
                response_time_ms=elapsed_ms,
                error=str(e)
            )
    
    async def _send_telegram(self, incident: Incident) -> tuple:
        """Send Telegram notification."""
        if not self.telegram_token or not self.telegram_chat_id:
            return (False, "Telegram not configured")
        
        # Format message
        severity_emoji = {
            IncidentSeverity.CRITICAL: "🚨🚨🚨",
            IncidentSeverity.SEVERE: "🚨",
            IncidentSeverity.MODERATE: "⚠️",
            IncidentSeverity.WARNING: "⚡",
            IncidentSeverity.INFO: "ℹ️"
        }
        
        emoji = severity_emoji.get(incident.severity, "📢")
        timestamp = datetime.fromtimestamp(incident.timestamp).strftime("%H:%M:%S")
        
        message = (
            f"{emoji} **{incident.severity.value.upper()} ALERT**\n\n"
            f"🎯 **Type:** {incident.incident_type}\n"
            f"📍 **Camera:** {incident.camera_id}\n"
            f"🕐 **Time:** {timestamp}\n"
            f"📊 **Confidence:** {incident.confidence*100:.1f}%\n\n"
            f"📝 {incident.description}\n\n"
            f"🔗 ID: `{incident.incident_id}`"
        )
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        return (True, "Telegram sent")
                    else:
                        return (False, f"Telegram error: {resp.status}")
        
        except Exception as e:
            return (False, f"Telegram error: {e}")
    
    async def _send_webhook(self, incident: Incident) -> tuple:
        """Send to configured webhooks."""
        if not self.webhook_urls:
            return (False, "No webhooks configured")
        
        payload = {
            "event": "incident_detected",
            "incident": incident.to_dict(),
            "source": "crash_detection_system"
        }
        
        success_count = 0
        async with aiohttp.ClientSession() as session:
            for url in self.webhook_urls:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=5
                    ) as resp:
                        if resp.status in [200, 201, 202]:
                            success_count += 1
                except Exception as e:
                    logger.error(f"Webhook error for {url}: {e}")
        
        if success_count > 0:
            return (True, f"Sent to {success_count}/{len(self.webhook_urls)} webhooks")
        else:
            return (False, "All webhooks failed")
    
    async def _send_sms(self, incident: Incident) -> tuple:
        """Send SMS notification (Twilio)."""
        if not self.sms_config.get("account_sid"):
            return (False, "SMS not configured")
        
        # Twilio integration would go here
        # This is a placeholder for the actual implementation
        logger.info(f"SMS would be sent for incident {incident.incident_id}")
        return (True, "SMS queued (placeholder)")
    
    async def _send_email(self, incident: Incident) -> tuple:
        """Send email notification."""
        if not self.email_config.get("smtp_host"):
            return (False, "Email not configured")
        
        # Email implementation would go here
        logger.info(f"Email would be sent for incident {incident.incident_id}")
        return (True, "Email queued (placeholder)")
    
    async def _alert_control_room(self, incident: Incident) -> tuple:
        """Alert control room (triggers callback + audio/visual)."""
        if self._control_room_callback:
            try:
                await self._control_room_callback(incident)
                return (True, "Control room alerted")
            except Exception as e:
                return (False, f"Control room callback error: {e}")
        else:
            logger.info(f"Control room alert: {incident.description}")
            return (True, "Control room alert logged")
    
    async def _alert_emergency_services(self, incident: Incident) -> tuple:
        """
        Alert emergency services (911/112).
        
        Note: In production, this would integrate with CAD systems
        or emergency dispatch APIs.
        """
        logger.critical(
            f"🚨 EMERGENCY SERVICES ALERT 🚨\n"
            f"Incident: {incident.incident_type}\n"
            f"Severity: {incident.severity.value}\n"
            f"Location: {incident.location}\n"
            f"Description: {incident.description}"
        )
        
        # Placeholder - actual integration would depend on local emergency APIs
        return (True, "Emergency services notified (log only)")
    
    def set_control_room_callback(self, callback: Callable):
        """Set callback for control room alerts."""
        self._control_room_callback = callback
    
    def add_webhook(self, url: str):
        """Add a webhook URL."""
        if url not in self.webhook_urls:
            self.webhook_urls.append(url)
    
    def remove_webhook(self, url: str):
        """Remove a webhook URL."""
        if url in self.webhook_urls:
            self.webhook_urls.remove(url)
    
    def set_routing_rule(
        self,
        severity: IncidentSeverity,
        channels: List[AlertChannel]
    ):
        """Set custom routing rule for severity level."""
        self.routing_rules[severity] = channels
    
    def _check_rate_limit(self, key: str) -> bool:
        """Check if dispatch is rate-limited."""
        now = time.time()
        last_time = self._last_dispatch_time.get(key, 0)
        
        if now - last_time < self._min_dispatch_interval:
            return False
        
        self._last_dispatch_time[key] = now
        return True
    
    def _record_dispatch(
        self,
        incident: Incident,
        results: List[DispatchResult]
    ):
        """Record dispatch to history."""
        record = {
            "incident_id": incident.incident_id,
            "timestamp": time.time(),
            "severity": incident.severity.value,
            "results": [
                {
                    "channel": r.channel.value,
                    "success": r.success,
                    "response_time_ms": r.response_time_ms
                }
                for r in results
            ]
        }
        
        self._dispatch_history.append(record)
        
        # Limit history size
        if len(self._dispatch_history) > 1000:
            self._dispatch_history.pop(0)
    
    def get_dispatch_history(
        self,
        limit: int = 50,
        severity_filter: Optional[str] = None
    ) -> List[Dict]:
        """Get recent dispatch history."""
        history = self._dispatch_history[-limit:]
        
        if severity_filter:
            history = [h for h in history if h["severity"] == severity_filter]
        
        return list(reversed(history))
    
    def get_dispatch_stats(self) -> Dict:
        """Get dispatch statistics."""
        if not self._dispatch_history:
            return {"total": 0}
        
        total = len(self._dispatch_history)
        
        # Count by channel
        channel_stats = {}
        for record in self._dispatch_history:
            for result in record["results"]:
                channel = result["channel"]
                if channel not in channel_stats:
                    channel_stats[channel] = {"total": 0, "success": 0}
                channel_stats[channel]["total"] += 1
                if result["success"]:
                    channel_stats[channel]["success"] += 1
        
        # Calculate success rates
        for channel, stats in channel_stats.items():
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        
        return {
            "total_dispatches": total,
            "channel_stats": channel_stats,
            "avg_response_time_ms": np.mean([
                result["response_time_ms"]
                for record in self._dispatch_history
                for result in record["results"]
            ]) if self._dispatch_history else 0
        }
