"""
System status router for FastAPI.
"""

from fastapi import APIRouter
from schemas import SystemStatusResponse, MLServiceStatus, DatabaseStatus, ConfigResponse
from config import get_settings

router = APIRouter(prefix="/api/v1/system", tags=["System"])


# Global references (set by main.py)
_model_loaded = False
_model_path = None
_face_model_loaded = False
_db_connected = True


def set_system_status(model_loaded: bool, model_path: str = None, 
                      face_model_loaded: bool = False, db_connected: bool = True):
    """Set the system status values."""
    global _model_loaded, _model_path, _face_model_loaded, _db_connected
    _model_loaded = model_loaded
    _model_path = model_path
    _face_model_loaded = face_model_loaded
    _db_connected = db_connected


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get comprehensive system status.
    
    Returns status of ML service, database, triage, and anonymization.
    Anonymization is always enabled (using YOLO face model or Haar Cascade fallback).
    """
    return SystemStatusResponse(
        ml_service=MLServiceStatus(
            available=_model_loaded,
            model_path=_model_path
        ),
        database=DatabaseStatus(connected=_db_connected),
        triage=True,
        anonymization=True  # Always enabled - uses Haar Cascade fallback if no YOLO face model
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get current system configuration.
    
    Returns model paths and feature flags.
    """
    settings = get_settings()
    
    return ConfigResponse(
        model_path=_model_path,
        face_model_available=_face_model_loaded,
        telegram_configured=settings.telegram_configured,
        confidence_threshold=settings.confidence_threshold,
        anonymization_enabled=settings.anonymization_enabled
    )
