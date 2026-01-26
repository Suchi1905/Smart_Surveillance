"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ==================== Health & Status Schemas ====================

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MLServiceStatus(BaseModel):
    """ML service status details."""
    model_config = {"protected_namespaces": ()}
    
    available: bool
    model_path: Optional[str] = None


class DatabaseStatus(BaseModel):
    """Database status details."""
    connected: bool


class SystemStatusResponse(BaseModel):
    """System status response."""
    ml_service: MLServiceStatus
    database: DatabaseStatus
    triage: bool



class ConfigResponse(BaseModel):
    """Configuration response."""
    model_config = {"protected_namespaces": ()}
    
    model_path: Optional[str]
    telegram_configured: bool
    confidence_threshold: float


# ==================== Crash Event Schemas ====================

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int


class CrashEventBase(BaseModel):
    """Base schema for crash events."""
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.85)
    class_name: str = Field(..., example="Accident")
    severity_category: str = Field(..., example="Severe")
    severity_index: float = Field(default=0.0, ge=0.0, le=1.0)
    track_id: Optional[int] = None
    frame_number: Optional[int] = None
    camera_id: str = Field(default="CAM-01")
    location: str = Field(default="North Intersection")


class CrashEventCreate(CrashEventBase):
    """Schema for creating a crash event."""
    bbox_x1: Optional[int] = None
    bbox_y1: Optional[int] = None
    bbox_x2: Optional[int] = None
    bbox_y2: Optional[int] = None
    alert_sent: bool = False
    anonymized: bool = False


class CrashEventResponse(CrashEventBase):
    """Schema for crash event response."""
    id: int
    timestamp: datetime
    bbox: Optional[BoundingBox] = None
    alert_sent: bool
    anonymized: bool
    
    class Config:
        from_attributes = True


class CrashEventListResponse(BaseModel):
    """Schema for list of crash events."""
    total: int
    events: List[CrashEventResponse]


# ==================== Statistics Schemas ====================

class SeverityStats(BaseModel):
    """Severity category statistics."""
    severe: int = 0
    moderate: int = 0
    mild: int = 0
    monitoring: int = 0


class CrashStatsResponse(BaseModel):
    """Crash statistics response."""
    total_events: int
    events_today: int
    events_last_hour: int
    severity_breakdown: SeverityStats
    average_confidence: float
    alerts_sent: int


# ==================== Detection Schemas ====================

class DetectionResult(BaseModel):
    """Single detection result."""
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None


class SeverityResult(BaseModel):
    """Severity analysis result."""
    track_id: int
    severity_index: float
    severity_category: str
    class_name: str
    confidence: float
    bbox: BoundingBox


class FrameAnalysisResponse(BaseModel):
    """Frame analysis response."""
    frame_number: int
    detections: List[DetectionResult]
    severity_results: List[SeverityResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
