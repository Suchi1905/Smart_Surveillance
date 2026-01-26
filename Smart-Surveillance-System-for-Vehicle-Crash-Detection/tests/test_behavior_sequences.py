"""
Tests for Pre-Accident Behavior Sequences.
"""

import pytest
import time
from src.services.behavior import BehaviorAnalyzer, BehaviorAlert, BehaviorType
from src.services.traffic_profiles import US_PROFILE

def test_speed_analysis():
    """Test speeding detection."""
    analyzer = BehaviorAnalyzer(profile=US_PROFILE)
    track_id = 1
    
    # Normal speed
    alert = analyzer._analyze_speed(track_id, 50.0, 60.0, (0,0))
    assert alert is None
    
    # Speeding (>1.2x limit)
    alert = analyzer._analyze_speed(track_id, 80.0, 60.0, (0,0))
    assert alert is not None
    assert alert.behavior_type == BehaviorType.SPEEDING
    assert alert.confidence > 0.0

def test_road_rage_sequence():
    """Test Road Rage sequence (Aggressive Accel + Tailgating)."""
    analyzer = BehaviorAnalyzer(profile=US_PROFILE)
    track_id = 1
    
    # 1. Aggressive Acceleration
    accel_alert = BehaviorAlert(
        track_id=track_id,
        behavior_type=BehaviorType.AGGRESSIVE_ACCEL,
        severity="warning",
        confidence=0.8,
        description="Accel",
        location=(0,0),
        timestamp=time.time()
    )
    analyzer._record_alert(accel_alert)
    
    # 2. Tailgating (External Alert)
    tailgating_alert = BehaviorAlert(
        track_id=track_id,
        behavior_type=BehaviorType.TAILGATING,
        severity="critical",
        confidence=0.9,
        description="Tailgating",
        location=(0,0),
        timestamp=time.time()
    )
    
    # Adding external alert should trigger sequence check
    seq_alert = analyzer.add_external_alert(tailgating_alert)
    
    assert seq_alert is not None
    assert seq_alert.behavior_type == BehaviorType.ROAD_RAGE
    assert "ROAD RAGE" in seq_alert.description

def test_reckless_driving_sequence():
    """Test Reckless Driving sequence (Speeding + Swerving)."""
    analyzer = BehaviorAnalyzer(profile=US_PROFILE)
    track_id = 1
    
    # 1. Speeding
    speed_alert = BehaviorAlert(
        track_id=track_id,
        behavior_type=BehaviorType.SPEEDING,
        severity="violation",
        confidence=0.9,
        description="Speeding",
        location=(0,0),
        timestamp=time.time()
    )
    analyzer._record_alert(speed_alert)
    
    # 2. Swerving (Simulated by analyzing trajectory that causes swerve)
    # OR simpler: just manually record swerve alert then check sequence manually 
    # (since _detect_sequences is internal, but add_external_alert calls it, or analyze_trajectory calls it)
    
    # Let's inject a Swerve alert internally
    swerve_alert = BehaviorAlert(
        track_id=track_id,
        behavior_type=BehaviorType.SWERVING,
        severity="warning",
        confidence=0.8,
        description="Swerving",
        location=(0,0),
        timestamp=time.time()
    )
    analyzer._record_alert(swerve_alert)
    
    # Trigger sequence check (usually called at end of analysis)
    seq_alert = analyzer._detect_sequences(track_id)
    
    assert seq_alert is not None
    assert seq_alert.behavior_type == BehaviorType.RECKLESS_DRIVING

def test_panic_maneuver_sequence():
    """Test Panic Maneuver sequence (Brake + Swerve)."""
    analyzer = BehaviorAnalyzer(profile=US_PROFILE)
    track_id = 1
    
    # 1. Sudden Brake
    brake_alert = BehaviorAlert(
        track_id=track_id,
        behavior_type=BehaviorType.SUDDEN_BRAKE,
        severity="warning",
        confidence=0.8,
        description="Brake",
        location=(0,0),
        timestamp=time.time()
    )
    analyzer._record_alert(brake_alert)
    
    # 2. Swerve
    swerve_alert = BehaviorAlert(
        track_id=track_id,
        behavior_type=BehaviorType.SWERVING,
        severity="critical",
        confidence=0.9,
        description="Swerve",
        location=(0,0),
        timestamp=time.time()
    )
    analyzer._record_alert(swerve_alert)
    
    seq_alert = analyzer._detect_sequences(track_id)
    
    assert seq_alert is not None
    assert seq_alert.behavior_type == BehaviorType.PANIC_MANEUVER
