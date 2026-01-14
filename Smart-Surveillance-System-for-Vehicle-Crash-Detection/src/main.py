"""
FastAPI Main Application Entry Point.

This is the main entry point for the Smart Surveillance System API.
Run with: uvicorn src.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from .config import get_settings
from .database import init_db
from .routers import health_router, system_router, crashes_router, video_router, websocket_router, analytics_router
from .routers.health import set_model_status
from .routers.system import set_system_status
from .routers.video import set_frame_generator, set_detection_service
from .services.detection import DetectionService
from .services.telegram import send_telegram_alert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global detection service
detection_service: DetectionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes database and loads models on startup.
    """
    global detection_service
    
    logger.info("=" * 60)
    logger.info("🚗 Smart Crash Detection System - FastAPI Backend")
    logger.info("=" * 60)
    
    # Initialize database
    logger.info("📊 Initializing database...")
    init_db()
    logger.info("✅ Database initialized")
    
    # Initialize detection service
    logger.info("🤖 Loading ML models...")
    detection_service = DetectionService()
    detection_loaded, face_loaded = detection_service.load_models()
    
    # Update router status
    set_model_status(detection_loaded)
    set_system_status(
        model_loaded=detection_loaded,
        model_path=get_settings().find_model_path(),
        face_model_loaded=face_loaded,
        db_connected=True
    )
    
    # Set frame generator for video router
    set_frame_generator(detection_service.generate_frames)
    set_detection_service(detection_service)
    
    # Set alert callback with WebSocket notification
    from .routers.websocket import get_ws_manager
    import asyncio
    
    def alert_callback_wrapper(confidence, frame, severity_info=None):
        """
        Wraps Telegram alert to also notify frontend via WebSocket.
        """
        # 1. Send Telegram Alert
        sent = send_telegram_alert(confidence, frame, severity_info)
        
        # 2. Notify Frontend via WebSocket if successful
        if sent and severity_info:
            try:
                manager = get_ws_manager()
                
                # Construct clean notification data
                notification_data = {
                    "type": "notification_sent",
                    "platform": "Telegram",
                    "timestamp": get_settings().get_timestamp(),
                    "track_id": severity_info.get('track_id', 'N/A'),
                    "severity": severity_info.get('severity_category', 'Unknown')
                }
                
                # Broadcast properly using the manager's queue
                loop = asyncio.get_event_loop()
                if loop.is_running():
                     asyncio.run_coroutine_threadsafe(
                        manager.broadcast({
                            "type": "notification_sent",
                            "data": notification_data
                        }, "alerts"),
                        loop
                    )
            except Exception as e:
                logger.error(f"Failed to broadcast alert notification: {e}")
                
        return sent

    detection_service.set_alert_callback(alert_callback_wrapper)
    
    settings = get_settings()
    logger.info(f"📡 API running on: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"📚 API Docs: http://localhost:{settings.api_port}/docs")
    logger.info(f"🎥 Video stream: http://localhost:{settings.api_port}/video?conf=0.6")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Smart Surveillance System API",
    description="""
    Privacy-Preserving Triage Framework for Vehicle Crash Detection.
    
    ## Features
    - Real-time crash detection with YOLOv8
    - Severity triage analysis (Severe/Moderate/Mild)
    - Edge-based anonymization (GDPR compliant)
    - Weather-robust detection via augmented training
    - Telegram alerts for severe crashes
    
    ## API Endpoints
    - `/health` - Health check
    - `/api/v1/system/status` - System status
    - `/api/v1/crashes` - Crash event CRUD
    - `/video` - Live MJPEG stream
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # FastAPI
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(system_router)
app.include_router(crashes_router)
app.include_router(video_router)
app.include_router(websocket_router)
app.include_router(analytics_router)


# Serve React frontend (production)
frontend_build = Path(__file__).parent.parent / "frontend" / "build"
if frontend_build.exists():
    app.mount("/static", StaticFiles(directory=frontend_build / "static"), name="static")
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA for all non-API routes."""
        file_path = frontend_build / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_build / "index.html")


# Legacy API compatibility routes
@app.get("/api/status")
async def legacy_status():
    """Legacy status endpoint for backward compatibility."""
    from .routers.system import get_system_status
    status = await get_system_status()
    return {
        "detection": status.ml_service.available,
        "triage": status.triage,
        "anonymization": status.anonymization,
        "model_loaded": status.ml_service.available
    }


@app.get("/api/system/status")
async def legacy_system_status():
    """Legacy system status endpoint."""
    from .routers.system import get_system_status
    status = await get_system_status()
    return {
        "ml_service": {
            "available": status.ml_service.available,
            "model_path": status.ml_service.model_path
        },
        "database": {"connected": status.database.connected},
        "triage": status.triage,
        "anonymization": status.anonymization
    }


@app.get("/api/config")
async def legacy_config():
    """Legacy config endpoint."""
    from .routers.system import get_config
    return await get_config()


@app.get("/api/crashes/recent/{hours}")
async def legacy_crashes_recent(hours: int):
    """Legacy crashes endpoint."""
    from .routers.crashes import list_crashes
    from .database import SessionLocal
    db = SessionLocal()
    try:
        result = await list_crashes(skip=0, limit=50, hours=hours, db=db)
        return [e.dict() for e in result.events]
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
