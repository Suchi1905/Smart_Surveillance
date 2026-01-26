"""
Risk Scoring Service.

Calculates composite risk scores for vehicles based on:
- Collision Prediction (Time to Collision)
- Dangerous Behavior (Swerving, etc.)
- Speeding / Speed Variance
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
import time

from .traffic_profiles import TrafficProfile, US_PROFILE
from .collision import CollisionRisk
from .behavior import BehaviorAlert

logger = logging.getLogger(__name__)

@dataclass
class RiskScore:
    """Composite risk score for a vehicle."""
    track_id: int
    total_score: float  # 0.0 to 1.0
    collision_score: float
    behavior_score: float
    speed_score: float
    factors: List[str]  # Descriptions of contributing factors
    timestamp: float

class RiskScorer:
    """
    Aggregates various risk factors into a single score.
    """
    
    def __init__(self, profile: Optional[TrafficProfile] = None):
        if profile is None:
            self.profile = US_PROFILE
        else:
            self.profile = profile
            
        logger.info(f"RiskScorer initialized with profile={self.profile.name}")
        
    def calculate_risk(
        self,
        track_id: int,
        collision_risks: List[CollisionRisk],
        behavior_alerts: List[BehaviorAlert],
        speed_kmh: Optional[float],
        speed_limit: float = 60.0
    ) -> RiskScore:
        """
        Calculate composite risk score.
        
        Args:
            track_id: ID of the vehicle
            collision_risks: List of active collision risks for this vehicle
            behavior_alerts: Recent behavior alerts for this vehicle
            speed_kmh: Current speed estimate
            speed_limit: Speed limit for the zone
            
        Returns:
            RiskScore object
        """
        factors = []
        
        # 1. Collision Score (0.0 - 1.0)
        c_score = 0.0
        relevant_risks = [r for r in collision_risks if r.track_id_1 == track_id or r.track_id_2 == track_id]
        
        if relevant_risks:
            # Take the highest risk
            worst_risk = max(relevant_risks, key=lambda r: self._risk_level_to_score(r.risk_level))
            c_score = self._risk_level_to_score(worst_risk.risk_level)
            if c_score > 0:
                factors.append(f"Collision Risk: {worst_risk.risk_level}")
        
        # 2. Behavior Score (0.0 - 1.0)
        b_score = 0.0
        if behavior_alerts:
            # Sum up behavior severities, capped at 1.0
            for alert in behavior_alerts:
                severity_val = self._behavior_severity_to_score(alert.severity)
                b_score += severity_val
                factors.append(f"Behavior: {alert.behavior_type.value} ({alert.severity})")
            b_score = min(1.0, b_score)
            
        # 3. Speed Score (0.0 - 1.0)
        s_score = 0.0
        if speed_kmh is not None and speed_kmh > 0:
            # Penalize speeding above limit
            if speed_kmh > speed_limit:
                excessratio = (speed_kmh - speed_limit) / speed_limit
                s_score = min(1.0, excessratio)
                if s_score > 0.2:
                    factors.append(f"Speeding: {speed_kmh:.0f}km/h")
        
        # Composite Score
        total_score = (
            (c_score * self.profile.weight_collision) +
            (b_score * self.profile.weight_behavior) +
            (s_score * self.profile.weight_speed)
        )
        
        # Normalize/Cap at 1.0 (though weights sum to ~1.0, individual scores can be high)
        total_score = min(1.0, total_score)
        
        return RiskScore(
            track_id=track_id,
            total_score=total_score,
            collision_score=c_score,
            behavior_score=b_score,
            speed_score=s_score,
            factors=factors,
            timestamp=time.time()
        )
    
    def _risk_level_to_score(self, level: str) -> float:
        """Map risk level string to float score."""
        mapping = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
            "none": 0.0
        }
        return mapping.get(level, 0.0)
    
    def _behavior_severity_to_score(self, severity: str) -> float:
        """Map behavior severity to float score."""
        mapping = {
            "critical": 0.8,   # Single critical behavior is very risky
            "violation": 0.5,
            "warning": 0.2
        }
        return mapping.get(severity, 0.1)
