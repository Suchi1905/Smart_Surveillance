"""
WebSocket Router for Real-time Events.

Provides WebSocket endpoints for live updates including:
- Real-time alerts
- Track updates
- System status
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, List, Any
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasting.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "alerts": set(),
            "tracks": set(),
            "status": set(),
            "all": set()
        }
        self._message_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self, websocket: WebSocket, channel: str = "all"):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        
        if channel not in self.active_connections:
            channel = "all"
        
        self.active_connections[channel].add(websocket)
        self.active_connections["all"].add(websocket)
        
        logger.info(f"WebSocket connected to channel: {channel}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "channel": channel,
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        for channel in self.active_connections.values():
            channel.discard(websocket)
        logger.info("WebSocket disconnected")
    
    async def broadcast(self, message: dict, channel: str = "all"):
        """Broadcast message to all connections in a channel."""
        connections = self.active_connections.get(channel, set())
        
        if not connections:
            return
        
        message["timestamp"] = datetime.now().isoformat()
        
        disconnected = set()
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_alert(self, alert: dict):
        """Broadcast an alert to alert subscribers."""
        message = {
            "type": "alert",
            "data": alert
        }
        await self.broadcast(message, "alerts")
        await self.broadcast(message, "all")
    
    async def broadcast_tracks(self, tracks: List[dict]):
        """Broadcast track updates."""
        message = {
            "type": "tracks",
            "data": tracks,
            "count": len(tracks)
        }
        await self.broadcast(message, "tracks")
    
    async def broadcast_status(self, status: dict):
        """Broadcast system status update."""
        message = {
            "type": "status",
            "data": status
        }
        await self.broadcast(message, "status")
    
    def get_connection_count(self) -> Dict[str, int]:
        """Get count of connections per channel."""
        return {
            channel: len(connections)
            for channel, connections in self.active_connections.items()
        }


# Global connection manager
manager = ConnectionManager()


def get_ws_manager() -> ConnectionManager:
    """Get the global WebSocket connection manager."""
    return manager


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for all events.
    
    Connect to receive all types of real-time updates:
    - alerts: Crash and behavior alerts
    - tracks: Vehicle tracking updates  
    - status: System status changes
    """
    await manager.connect(websocket, "all")
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await _handle_client_message(websocket, message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for alert events only."""
    await manager.connect(websocket, "alerts")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/tracks")
async def websocket_tracks(websocket: WebSocket):
    """WebSocket endpoint for track updates only."""
    await manager.connect(websocket, "tracks")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for system status updates."""
    await manager.connect(websocket, "status")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def _handle_client_message(websocket: WebSocket, message: dict):
    """Handle incoming client messages."""
    msg_type = message.get("type", "")
    
    if msg_type == "ping":
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif msg_type == "subscribe":
        channel = message.get("channel", "all")
        if channel in manager.active_connections:
            manager.active_connections[channel].add(websocket)
            await websocket.send_json({
                "type": "subscribed",
                "channel": channel
            })
    
    elif msg_type == "unsubscribe":
        channel = message.get("channel")
        if channel and channel in manager.active_connections:
            manager.active_connections[channel].discard(websocket)
            await websocket.send_json({
                "type": "unsubscribed",
                "channel": channel
            })


# Export the router
websocket_router = router
