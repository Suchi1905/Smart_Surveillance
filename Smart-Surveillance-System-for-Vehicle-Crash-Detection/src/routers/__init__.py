"""
Routers package for FastAPI endpoints.
"""

from .health import router as health_router
from .system import router as system_router
from .crashes import router as crashes_router
from .video import router as video_router

__all__ = ["health_router", "system_router", "crashes_router", "video_router"]
