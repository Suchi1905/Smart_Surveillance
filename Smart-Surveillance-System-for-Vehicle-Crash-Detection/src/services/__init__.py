"""
Services package for business logic.

Includes detection, tracking, speed estimation, behavior analysis,
collision prediction, and emergency dispatch services.
"""

from .severity_triage import SeverityTriageSystem
from .anonymization import anonymize_frame
from .detection import DetectionService
from .telegram import TelegramAlertService
from .tracker import ByteTracker, Track, TrackState
from .speed_estimator import SpeedEstimator, SpeedMeasurement, SpeedZone
from .collision import CollisionPredictor, CollisionRisk, NearMissEvent
from .behavior import BehaviorAnalyzer, BehaviorAlert, BehaviorType
from .emergency import EmergencyDispatcher, Incident, IncidentSeverity, AlertChannel

__all__ = [
    # Core
    "SeverityTriageSystem",
    "anonymize_frame", 
    "DetectionService",
    "TelegramAlertService",
    # Tracking
    "ByteTracker",
    "Track",
    "TrackState",
    # Speed
    "SpeedEstimator",
    "SpeedMeasurement",
    "SpeedZone",
    # Collision
    "CollisionPredictor",
    "CollisionRisk",
    "NearMissEvent",
    # Behavior
    "BehaviorAnalyzer",
    "BehaviorAlert",
    "BehaviorType",
    # Emergency
    "EmergencyDispatcher",
    "Incident",
    "IncidentSeverity",
    "AlertChannel"
]
