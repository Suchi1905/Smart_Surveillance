"""
Health check router for FastAPI.
"""

from fastapi import APIRouter
from datetime import datetime
from ..schemas import HealthResponse

router = APIRouter(tags=["Health"])


# Global reference to model status (set by main.py)
_model_loaded = False


def set_model_status(loaded: bool):
    """Set the model loaded status."""
    global _model_loaded
    _model_loaded = loaded


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and model loading state.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded,
        timestamp=datetime.utcnow()
    )
