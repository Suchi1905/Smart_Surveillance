"""
Crash events CRUD router for FastAPI.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import List, Optional

from ..database import get_db, CrashEvent
from ..schemas import (
    CrashEventCreate, 
    CrashEventResponse, 
    CrashEventListResponse,
    CrashStatsResponse,
    SeverityStats,
    BoundingBox
)

router = APIRouter(prefix="/api/v1/crashes", tags=["Crashes"])


@router.get("", response_model=CrashEventListResponse)
async def list_crashes(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum records to return"),
    severity: Optional[str] = Query(None, description="Filter by severity category"),
    hours: Optional[int] = Query(None, ge=1, description="Filter to last N hours"),
    db: Session = Depends(get_db)
):
    """
    List crash events with pagination and filters.
    
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **severity**: Filter by severity category (Severe, Moderate, Mild, Monitoring)
    - **hours**: Filter to events within last N hours
    """
    query = db.query(CrashEvent)
    
    # Apply filters
    if severity:
        query = query.filter(CrashEvent.severity_category == severity)
    
    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(CrashEvent.timestamp >= cutoff)
    
    # Get total count
    total = query.count()
    
    # Get paginated results
    events = query.order_by(CrashEvent.timestamp.desc()).offset(skip).limit(limit).all()
    
    # Convert to response model
    event_responses = []
    for event in events:
        bbox = None
        if event.bbox_x1 is not None:
            bbox = BoundingBox(
                x1=event.bbox_x1, y1=event.bbox_y1,
                x2=event.bbox_x2, y2=event.bbox_y2
            )
        
        event_responses.append(CrashEventResponse(
            id=event.id,
            timestamp=event.timestamp,
            confidence=event.confidence,
            class_name=event.class_name,
            severity_category=event.severity_category,
            severity_index=event.severity_index,
            track_id=event.track_id,
            frame_number=event.frame_number,
            bbox=bbox,
            alert_sent=event.alert_sent,
            anonymized=event.anonymized,
            camera_id=event.camera_id,
            location=event.location
        ))
    
    return CrashEventListResponse(total=total, events=event_responses)


@router.post("", response_model=CrashEventResponse, status_code=201)
async def create_crash(
    crash: CrashEventCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new crash event.
    
    Records a detected crash with severity information.
    """
    db_crash = CrashEvent(
        confidence=crash.confidence,
        class_name=crash.class_name,
        severity_category=crash.severity_category,
        severity_index=crash.severity_index,
        track_id=crash.track_id,
        frame_number=crash.frame_number,
        bbox_x1=crash.bbox_x1,
        bbox_y1=crash.bbox_y1,
        bbox_x2=crash.bbox_x2,
        bbox_y2=crash.bbox_y2,
        alert_sent=crash.alert_sent,
        anonymized=crash.anonymized,
        camera_id=crash.camera_id,
        location=crash.location
    )
    
    db.add(db_crash)
    db.commit()
    db.refresh(db_crash)
    
    bbox = None
    if db_crash.bbox_x1 is not None:
        bbox = BoundingBox(
            x1=db_crash.bbox_x1, y1=db_crash.bbox_y1,
            x2=db_crash.bbox_x2, y2=db_crash.bbox_y2
        )
    
    return CrashEventResponse(
        id=db_crash.id,
        timestamp=db_crash.timestamp,
        confidence=db_crash.confidence,
        class_name=db_crash.class_name,
        severity_category=db_crash.severity_category,
        severity_index=db_crash.severity_index,
        track_id=db_crash.track_id,
        frame_number=db_crash.frame_number,
        bbox=bbox,
        alert_sent=db_crash.alert_sent,
        anonymized=db_crash.anonymized,
        camera_id=db_crash.camera_id,
        location=db_crash.location
    )


@router.get("/{crash_id}", response_model=CrashEventResponse)
async def get_crash(crash_id: int, db: Session = Depends(get_db)):
    """
    Get a specific crash event by ID.
    """
    crash = db.query(CrashEvent).filter(CrashEvent.id == crash_id).first()
    
    if not crash:
        raise HTTPException(status_code=404, detail="Crash event not found")
    
    bbox = None
    if crash.bbox_x1 is not None:
        bbox = BoundingBox(
            x1=crash.bbox_x1, y1=crash.bbox_y1,
            x2=crash.bbox_x2, y2=crash.bbox_y2
        )
    
    return CrashEventResponse(
        id=crash.id,
        timestamp=crash.timestamp,
        confidence=crash.confidence,
        class_name=crash.class_name,
        severity_category=crash.severity_category,
        severity_index=crash.severity_index,
        track_id=crash.track_id,
        frame_number=crash.frame_number,
        bbox=bbox,
        alert_sent=crash.alert_sent,
        anonymized=crash.anonymized,
        camera_id=crash.camera_id,
        location=crash.location
    )


@router.get("/stats/summary", response_model=CrashStatsResponse)
async def get_crash_stats(db: Session = Depends(get_db)):
    """
    Get crash event statistics summary.
    
    Returns total counts, breakdown by severity, and time-based metrics.
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hour_ago = now - timedelta(hours=1)
    
    # Total events
    total_events = db.query(CrashEvent).count()
    
    # Events today
    events_today = db.query(CrashEvent).filter(
        CrashEvent.timestamp >= today_start
    ).count()
    
    # Events last hour
    events_last_hour = db.query(CrashEvent).filter(
        CrashEvent.timestamp >= hour_ago
    ).count()
    
    # Severity breakdown
    severity_counts = db.query(
        CrashEvent.severity_category,
        func.count(CrashEvent.id)
    ).group_by(CrashEvent.severity_category).all()
    
    severity_stats = SeverityStats()
    for category, count in severity_counts:
        if category.lower() == "severe":
            severity_stats.severe = count
        elif category.lower() == "moderate":
            severity_stats.moderate = count
        elif category.lower() == "mild":
            severity_stats.mild = count
        else:
            severity_stats.monitoring = count
    
    # Average confidence
    avg_conf = db.query(func.avg(CrashEvent.confidence)).scalar() or 0.0
    
    # Alerts sent
    alerts_sent = db.query(CrashEvent).filter(CrashEvent.alert_sent == True).count()
    
    return CrashStatsResponse(
        total_events=total_events,
        events_today=events_today,
        events_last_hour=events_last_hour,
        severity_breakdown=severity_stats,
        average_confidence=round(avg_conf, 4),
        alerts_sent=alerts_sent
    )


@router.delete("/{crash_id}", status_code=204)
async def delete_crash(crash_id: int, db: Session = Depends(get_db)):
    """
    Delete a crash event by ID.
    """
    crash = db.query(CrashEvent).filter(CrashEvent.id == crash_id).first()
    
    if not crash:
        raise HTTPException(status_code=404, detail="Crash event not found")
    
    db.delete(crash)
    db.commit()
    
    return None
