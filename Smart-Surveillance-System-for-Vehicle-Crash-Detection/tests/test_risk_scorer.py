"""
Tests for Risk Scoring System.
"""

import pytest
from dataclasses import dataclass
from typing import List

from src.services.risk_scorer import RiskScorer, RiskScore
from src.services.traffic_profiles import US_PROFILE, INDIAN_PROFILE
from src.services.collision import CollisionRisk
from src.services.behavior import BehaviorAlert, BehaviorType

# Mock classes for testing
@dataclass
class MockCollisionRisk:
    track_id_1: int
    track_id_2: int
    risk_level: str

@dataclass
class MockBehaviorAlert:
    track_id: int
    behavior_type: BehaviorType
    severity: str

def test_risk_scorer_init():
    """Test initialization."""
    scorer = RiskScorer()
    assert scorer.profile == US_PROFILE

def test_calculate_risk_empty():
    """Test risk calculation with no factors."""
    scorer = RiskScorer()
    score = scorer.calculate_risk(1, [], [], 0.0)
    assert score.total_score == 0.0
    assert score.collision_score == 0.0
    assert score.behavior_score == 0.0
    assert score.speed_score == 0.0

def test_calculate_risk_collision():
    """Test risk calculation with collision risk."""
    scorer = RiskScorer(profile=US_PROFILE)
    # US collision weight is 0.5
    
    risks = [
        MockCollisionRisk(1, 2, "critical")  # Score 1.0
    ]
    
    score = scorer.calculate_risk(1, risks, [], 0.0)
    
    # 1.0 * 0.5 = 0.5
    assert score.collision_score == 1.0
    assert score.total_score == 0.5

def test_calculate_risk_behavior():
    """Test risk calculation with behavior alerts."""
    scorer = RiskScorer(profile=US_PROFILE)
    # US behavior weight is 0.3
    
    alerts = [
        MockBehaviorAlert(1, BehaviorType.SWERVING, "critical") # Score 0.8
    ]
    
    score = scorer.calculate_risk(1, [], alerts, 0.0)
    
    # 0.8 * 0.3 = 0.24
    assert score.behavior_score == 0.8
    assert abs(score.total_score - 0.24) < 0.001

def test_calculate_risk_speed():
    """Test risk calculation with speeding."""
    scorer = RiskScorer(profile=US_PROFILE)
    # US speed weight is 0.2
    
    # Speed 120 vs Limit 60 -> Ratio 1.0 (capped)
    score = scorer.calculate_risk(1, [], [], 120.0, speed_limit=60.0)
    
    assert score.speed_score == 1.0
    assert score.total_score == 0.2

def test_composite_risk():
    """Test combined risk factors."""
    scorer = RiskScorer(profile=US_PROFILE)
    
    risks = [MockCollisionRisk(1, 2, "medium")] # 0.5 * 0.5 = 0.25 contribution
    alerts = [MockBehaviorAlert(1, BehaviorType.SWERVING, "warning")] # 0.2 * 0.3 = 0.06 contribution
    speed = 90.0 # (90-60)/60 = 0.5 * 0.2 = 0.10 contribution
    
    score = scorer.calculate_risk(1, risks, alerts, speed, speed_limit=60.0)
    
    expected = 0.25 + 0.06 + 0.10
    assert abs(score.total_score - expected) < 0.001
    assert len(score.factors) == 3
