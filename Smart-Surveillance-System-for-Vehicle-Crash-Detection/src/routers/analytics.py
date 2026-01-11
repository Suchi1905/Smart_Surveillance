"""
Analytics Router for traffic statistics and reports.

Provides endpoints for:
- Speed statistics
- Behavior analytics
- Incident summaries
- Traffic flow data
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


# Response models
class SpeedStats(BaseModel):
    """Speed statistics response."""
    average_speed: float
    max_speed: float
    min_speed: float
    speeding_count: int
    total_vehicles: int
    speed_distribution: Dict[str, int]


class BehaviorStats(BaseModel):
    """Behavior analytics response."""
    total_alerts: int
    alerts_by_type: Dict[str, int]
    alerts_by_severity: Dict[str, int]
    recent_alerts: List[Dict]


class TrafficFlow(BaseModel):
    """Traffic flow statistics."""
    vehicle_count: int
    average_speed: float
    congestion_level: str
    timestamp: str


class IncidentSummary(BaseModel):
    """Incident summary response."""
    total_incidents: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    recent: List[Dict]


# In-memory analytics store (would be DB in production)
_analytics_store = {
    "speeds": [],
    "behaviors": [],
    "incidents": [],
    "traffic": []
}


def record_speed(track_id: int, speed_kmh: float, timestamp: float = None):
    """Record a speed measurement for analytics."""
    _analytics_store["speeds"].append({
        "track_id": track_id,
        "speed": speed_kmh,
        "timestamp": timestamp or datetime.now().timestamp()
    })
    # Keep last 10000 records
    if len(_analytics_store["speeds"]) > 10000:
        _analytics_store["speeds"] = _analytics_store["speeds"][-10000:]


def record_behavior(alert: Dict):
    """Record a behavior alert for analytics."""
    _analytics_store["behaviors"].append(alert)
    if len(_analytics_store["behaviors"]) > 1000:
        _analytics_store["behaviors"] = _analytics_store["behaviors"][-1000:]


def record_incident(incident: Dict):
    """Record an incident for analytics."""
    _analytics_store["incidents"].append(incident)
    if len(_analytics_store["incidents"]) > 500:
        _analytics_store["incidents"] = _analytics_store["incidents"][-500:]


@router.get("/speed", response_model=SpeedStats)
async def get_speed_stats(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to analyze")
):
    """
    Get speed statistics for the specified time period.
    
    Returns average, max, min speeds and distribution.
    """
    cutoff = datetime.now().timestamp() - (hours * 3600)
    recent_speeds = [
        s["speed"] for s in _analytics_store["speeds"]
        if s["timestamp"] >= cutoff
    ]
    
    if not recent_speeds:
        return SpeedStats(
            average_speed=0,
            max_speed=0,
            min_speed=0,
            speeding_count=0,
            total_vehicles=0,
            speed_distribution={}
        )
    
    # Calculate distribution
    distribution = {
        "0-30": 0,
        "30-50": 0,
        "50-70": 0,
        "70-100": 0,
        "100+": 0
    }
    
    speeding_count = 0
    for speed in recent_speeds:
        if speed < 30:
            distribution["0-30"] += 1
        elif speed < 50:
            distribution["30-50"] += 1
        elif speed < 70:
            distribution["50-70"] += 1
        elif speed < 100:
            distribution["70-100"] += 1
        else:
            distribution["100+"] += 1
            
        if speed > 50:  # Simple speeding threshold
            speeding_count += 1
    
    return SpeedStats(
        average_speed=round(sum(recent_speeds) / len(recent_speeds), 1),
        max_speed=round(max(recent_speeds), 1),
        min_speed=round(min(recent_speeds), 1),
        speeding_count=speeding_count,
        total_vehicles=len(recent_speeds),
        speed_distribution=distribution
    )


@router.get("/behavior", response_model=BehaviorStats)
async def get_behavior_stats(
    hours: int = Query(default=24, ge=1, le=168)
):
    """
    Get behavior analytics for the specified time period.
    
    Returns alert counts by type and severity.
    """
    cutoff = datetime.now().timestamp() - (hours * 3600)
    recent = [
        b for b in _analytics_store["behaviors"]
        if b.get("timestamp", 0) >= cutoff
    ]
    
    by_type = {}
    by_severity = {}
    
    for b in recent:
        btype = b.get("behavior_type", "unknown")
        severity = b.get("severity", "unknown")
        by_type[btype] = by_type.get(btype, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
    
    return BehaviorStats(
        total_alerts=len(recent),
        alerts_by_type=by_type,
        alerts_by_severity=by_severity,
        recent_alerts=recent[-10:]
    )


@router.get("/incidents", response_model=IncidentSummary)
async def get_incident_summary(
    hours: int = Query(default=24, ge=1, le=168)
):
    """
    Get incident summary for the specified time period.
    """
    cutoff = datetime.now().timestamp() - (hours * 3600)
    recent = [
        i for i in _analytics_store["incidents"]
        if i.get("timestamp", 0) >= cutoff
    ]
    
    by_severity = {}
    by_type = {}
    
    for i in recent:
        severity = i.get("severity", "unknown")
        itype = i.get("incident_type", "unknown")
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_type[itype] = by_type.get(itype, 0) + 1
    
    return IncidentSummary(
        total_incidents=len(recent),
        by_severity=by_severity,
        by_type=by_type,
        recent=recent[-5:]
    )


@router.get("/traffic")
async def get_traffic_flow():
    """
    Get current traffic flow statistics.
    """
    # This would be calculated from real-time tracking data
    speeds = _analytics_store["speeds"][-100:] if _analytics_store["speeds"] else []
    
    avg_speed = sum(s["speed"] for s in speeds) / len(speeds) if speeds else 0
    
    # Determine congestion level
    if avg_speed < 20:
        congestion = "heavy"
    elif avg_speed < 40:
        congestion = "moderate"
    else:
        congestion = "light"
    
    return TrafficFlow(
        vehicle_count=len(speeds),
        average_speed=round(avg_speed, 1),
        congestion_level=congestion,
        timestamp=datetime.now().isoformat()
    )


@router.get("/dashboard")
async def get_dashboard_data():
    """
    Get combined dashboard analytics data.
    """
    # Aggregate all analytics for dashboard
    speeds = _analytics_store["speeds"][-1000:]
    behaviors = _analytics_store["behaviors"][-100:]
    incidents = _analytics_store["incidents"][-50:]
    
    return {
        "summary": {
            "total_vehicles_tracked": len(set(s["track_id"] for s in speeds)),
            "total_behavior_alerts": len(behaviors),
            "total_incidents": len(incidents),
            "active_cameras": 4  # Placeholder
        },
        "recent_alerts": behaviors[-5:],
        "recent_incidents": incidents[-5:],
        "speed_stats": {
            "average": round(sum(s["speed"] for s in speeds) / len(speeds), 1) if speeds else 0,
            "max": round(max(s["speed"] for s in speeds), 1) if speeds else 0
        },
        "timestamp": datetime.now().isoformat()
    }


# Export router
analytics_router = router
