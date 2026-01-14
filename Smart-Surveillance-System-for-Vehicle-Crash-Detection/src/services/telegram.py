"""
Telegram alert service for crash notifications.

This module handles sending anonymized crash alerts to Telegram.
"""

import requests
import cv2
import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from ..config import get_settings

logger = logging.getLogger(__name__)


class TelegramAlertService:
    """
    Telegram notification service for crash alerts.
    
    Sends anonymized crash images with severity information to a
    configured Telegram chat.
    
    Attributes:
        settings: Application settings
        enabled: Whether Telegram is properly configured
    """
    
    def __init__(self):
        """Initialize the Telegram alert service."""
        self.settings = get_settings()
        self.enabled = self.settings.telegram_configured
        
        if not self.enabled:
            logger.warning("Telegram not configured - alerts disabled")
    
    def send_alert(
        self, 
        confidence: float, 
        frame, 
        severity_info: Optional[dict] = None
    ) -> bool:
        """
        Send crash alert to Telegram with anonymized frame.
        
        Args:
            confidence: Detection confidence score
            frame: Anonymized frame (numpy array)
            severity_info: Optional severity analysis results
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Telegram not configured, skipping alert")
            return False
        
        try:
            # Save frame temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = Path(f"temp_{timestamp}.jpg")
            cv2.imwrite(str(temp_path), frame)
            
            # Build message
            message = self._build_message(confidence, severity_info)
            
            # Send to Telegram
            success = self._send_photo(temp_path, message)
            
            # Cleanup
            if temp_path.exists():
                os.remove(temp_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            return False
    
    def _build_message(
        self, 
        confidence: float, 
        severity_info: Optional[dict]
    ) -> str:
        """Build alert message text with severity advice."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not severity_info:
            return (
                f"🚨 *CRASH DETECTED*\n"
                f"🕒 Time: {timestamp}\n"
                f"📊 Confidence: {confidence:.2f}\n\n"
                f"⚠️ *Analysis pending...*\n"
                f"🔒 Frame anonymized for privacy"
            )

        category = severity_info.get('severity_category', 'Unknown')
        index = severity_info.get('severity_index', 0)
        track_id = severity_info.get('track_id', 'N/A')

        # Severity-specific messaging
        if category == "Severe":
            icon = "🔴"
            status = "CRITICAL: IMMEDIATE ACTION REQUIRED"
            advice = "🚑 Call Emergency Services (911/112) immediately.\n👮 Report serious collision."
        elif category == "Moderate":
            icon = "🟠"
            status = "WARNING: Moderate Impact"
            advice = "🩺 Check for injuries.\n🚙 Assess vehicle damage."
        elif category == "Mild":
            icon = "🟡"
            status = "NOTICE: Minor Incident"
            advice = "👀 Monitor situation.\n📝 Log incident for review."
        else:
            icon = "⚪"
            status = "MONITORING: Traffic Event"
            advice = "👁️ Ongoing surveillance."

        message = (
            f"{icon} *{status}*\n\n"
            f"🕒 Time: {timestamp}\n"
            f"📊 Severity Index: {index:.2f}\n"
            f"🎯 Track ID: #{track_id}\n\n"
            f"💡 *Action Required:*\n{advice}\n\n"
            f"🔒 _Frame anonymized for privacy_"
        )
        
        return message
    
    def _send_photo(self, photo_path: Path, caption: str) -> bool:
        """
        Send photo to Telegram chat.
        
        Args:
            photo_path: Path to image file
            caption: Message caption
            
        Returns:
            True if sent successfully
        """
        url = f"https://api.telegram.org/bot{self.settings.bot_token}/sendPhoto"
        
        payload = {
            "chat_id": self.settings.chat_id,
            "caption": caption
        }
        
        with open(photo_path, 'rb') as photo:
            files = {"photo": photo}
            response = requests.post(url, data=payload, files=files, timeout=30)
        
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully")
            return True
        else:
            logger.error(f"Telegram error: {response.status_code}, {response.text}")
            return False
    
    def send_test_message(self) -> bool:
        """
        Send a test message to verify Telegram configuration.
        
        Returns:
            True if test message sent successfully
        """
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.settings.bot_token}/sendMessage"
            payload = {
                "chat_id": self.settings.chat_id,
                "text": "🧪 Smart Surveillance System - Test Alert\n\nTelegram integration is working correctly!"
            }
            
            response = requests.post(url, data=payload, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Test message error: {e}")
            return False


# Singleton instance
_alert_service: Optional[TelegramAlertService] = None


def get_alert_service() -> TelegramAlertService:
    """Get or create the Telegram alert service singleton."""
    global _alert_service
    if _alert_service is None:
        _alert_service = TelegramAlertService()
    return _alert_service


def send_telegram_alert(
    confidence: float, 
    frame, 
    severity_info: Optional[dict] = None
) -> bool:
    """
    Convenience function to send Telegram alert.
    
    Args:
        confidence: Detection confidence
        frame: Anonymized frame
        severity_info: Optional severity info dict
        
    Returns:
        True if sent successfully
    """
    service = get_alert_service()
    
    # Convert SeverityResult to dict if needed
    if severity_info and hasattr(severity_info, '__dict__'):
        severity_info = {
            'severity_category': severity_info.severity_category,
            'severity_index': severity_info.severity_index,
            'track_id': severity_info.track_id
        }
    
    return service.send_alert(confidence, frame, severity_info)
