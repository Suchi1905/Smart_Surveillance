"""
Collision Detection and Prediction Service.

This module provides time-to-collision (TTC) calculation and
collision risk assessment between tracked vehicles.

Uses trajectory extrapolation and spatial intersection analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CollisionRisk:
    """
    Collision risk assessment result.
    
    Attributes:
        track_id_1: First vehicle track ID
        track_id_2: Second vehicle track ID
        ttc_seconds: Time to collision in seconds
        collision_point: Predicted collision location (x, y)
        risk_level: "low", "medium", "high", "critical"
        confidence: Prediction confidence (0-1)
        relative_speed: Relative speed between vehicles
        minimum_distance: Closest approach distance
    """
    track_id_1: int
    track_id_2: int
    ttc_seconds: float
    collision_point: Optional[Tuple[float, float]]
    risk_level: str
    confidence: float
    relative_speed: float
    minimum_distance: float


@dataclass 
class NearMissEvent:
    """
    Near-miss detection event.
    
    Attributes:
        track_id_1: First vehicle track ID
        track_id_2: Second vehicle track ID  
        min_distance: Minimum distance reached (pixels)
        time_at_closest: Timestamp at closest approach
        relative_speed: Relative speed at closest approach
        was_collision_predicted: Whether collision was predicted
    """
    track_id_1: int
    track_id_2: int
    min_distance: float
    time_at_closest: float
    relative_speed: float
    was_collision_predicted: bool


class CollisionPredictor:
    """
    Predicts potential collisions between tracked vehicles.
    
    Uses trajectory extrapolation and bounding box intersection
    to estimate time-to-collision and risk levels.
    """
    
    # Risk thresholds (seconds)
    TTC_CRITICAL = 1.0
    TTC_HIGH = 2.0
    TTC_MEDIUM = 4.0
    TTC_LOW = 8.0
    
    # Distance thresholds (pixels) - adjust based on camera view
    COLLISION_DISTANCE = 50.0  # Boxes overlapping/touching
    NEAR_MISS_DISTANCE = 100.0  # Very close call
    DANGER_DISTANCE = 200.0  # Getting dangerous
    
    def __init__(
        self,
        prediction_horizon: float = 5.0,
        fps: float = 30.0,
        safe_following_time: float = 2.0
    ):
        """
        Initialize collision predictor.
        
        Args:
            prediction_horizon: How far ahead to predict (seconds)
            fps: Camera frame rate
            safe_following_time: Minimum safe following time (seconds)
        """
        self.prediction_horizon = prediction_horizon
        self.fps = fps
        self.safe_following_time = safe_following_time
        
        # Track near-miss history
        self._near_misses: List[NearMissEvent] = []
        
        logger.info(f"CollisionPredictor initialized with {prediction_horizon}s horizon")
    
    def calculate_ttc(
        self,
        pos1: Tuple[float, float],
        vel1: Tuple[float, float],
        pos2: Tuple[float, float],
        vel2: Tuple[float, float]
    ) -> Optional[float]:
        """
        Calculate time to collision between two objects.
        
        Uses relative velocity and position to estimate collision time.
        
        Args:
            pos1: Position of object 1 (x, y)
            vel1: Velocity of object 1 (vx, vy) pixels/frame
            pos2: Position of object 2 (x, y)
            vel2: Velocity of object 2 (vx, vy) pixels/frame
        
        Returns:
            TTC in seconds, or None if no collision predicted
        """
        # Relative position and velocity
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dvx = vel2[0] - vel1[0]
        dvy = vel2[1] - vel1[1]
        
        # Distance
        distance = np.sqrt(dx**2 + dy**2)
        
        # Relative velocity magnitude
        rel_speed = np.sqrt(dvx**2 + dvy**2)
        
        if rel_speed < 0.1:  # Essentially moving together
            return None
        
        # Check if approaching each other
        # Dot product of position difference and velocity difference
        closing = dx * dvx + dy * dvy
        
        if closing >= 0:  # Moving apart
            return None
        
        # Time to collision (simplified linear model)
        # TTC = distance / closing_speed
        closing_speed = -closing / distance  # positive when approaching
        
        if closing_speed <= 0:
            return None
        
        ttc_frames = distance / closing_speed
        ttc_seconds = ttc_frames / self.fps
        
        return ttc_seconds if ttc_seconds <= self.prediction_horizon else None
    
    def predict_collision_point(
        self,
        pos1: Tuple[float, float],
        vel1: Tuple[float, float],
        pos2: Tuple[float, float],
        vel2: Tuple[float, float],
        ttc_frames: float
    ) -> Tuple[float, float]:
        """
        Predict the location where collision would occur.
        
        Returns:
            (x, y) collision point
        """
        # Extrapolate positions
        future_x1 = pos1[0] + vel1[0] * ttc_frames
        future_y1 = pos1[1] + vel1[1] * ttc_frames
        future_x2 = pos2[0] + vel2[0] * ttc_frames
        future_y2 = pos2[1] + vel2[1] * ttc_frames
        
        # Collision point is midpoint
        return ((future_x1 + future_x2) / 2, (future_y1 + future_y2) / 2)
    
    def assess_risk(self, ttc_seconds: Optional[float]) -> str:
        """
        Assess risk level based on TTC.
        
        Returns:
            "critical", "high", "medium", "low", or "none"
        """
        if ttc_seconds is None:
            return "none"
        
        if ttc_seconds <= self.TTC_CRITICAL:
            return "critical"
        elif ttc_seconds <= self.TTC_HIGH:
            return "high"
        elif ttc_seconds <= self.TTC_MEDIUM:
            return "medium"
        elif ttc_seconds <= self.TTC_LOW:
            return "low"
        else:
            return "none"
    
    def analyze_pair(
        self,
        track1_id: int,
        track1_pos: Tuple[float, float],
        track1_vel: Tuple[float, float],
        track1_bbox: np.ndarray,
        track2_id: int,
        track2_pos: Tuple[float, float],
        track2_vel: Tuple[float, float],
        track2_bbox: np.ndarray
    ) -> Optional[CollisionRisk]:
        """
        Analyze collision risk between two tracked vehicles.
        
        Returns:
            CollisionRisk or None if no significant risk
        """
        # Current distance
        dx = track2_pos[0] - track1_pos[0]
        dy = track2_pos[1] - track1_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Relative velocity
        dvx = track2_vel[0] - track1_vel[0]
        dvy = track2_vel[1] - track1_vel[1]
        rel_speed = np.sqrt(dvx**2 + dvy**2) * self.fps  # pixels/second
        
        # Calculate TTC
        ttc = self.calculate_ttc(track1_pos, track1_vel, track2_pos, track2_vel)
        risk_level = self.assess_risk(ttc)
        
        if risk_level == "none" and distance > self.DANGER_DISTANCE:
            return None
        
        # Calculate collision point if TTC exists
        collision_point = None
        if ttc is not None:
            ttc_frames = ttc * self.fps
            collision_point = self.predict_collision_point(
                track1_pos, track1_vel, track2_pos, track2_vel, ttc_frames
            )
        
        # Confidence based on trajectory consistency
        confidence = 0.8 if rel_speed > 10 else 0.5
        
        # Check for near-miss (close but no collision)
        if distance < self.NEAR_MISS_DISTANCE and risk_level in ["none", "low"]:
            self._record_near_miss(
                track1_id, track2_id, distance, rel_speed, ttc is not None
            )
        
        return CollisionRisk(
            track_id_1=track1_id,
            track_id_2=track2_id,
            ttc_seconds=ttc if ttc else float('inf'),
            collision_point=collision_point,
            risk_level=risk_level if risk_level != "none" else "low",
            confidence=confidence,
            relative_speed=rel_speed,
            minimum_distance=distance
        )
    
    def analyze_all_tracks(
        self,
        tracks: List[Dict]
    ) -> List[CollisionRisk]:
        """
        Analyze collision risks between all track pairs.
        
        Args:
            tracks: List of track dicts with keys:
                   id, center, velocity, bbox
        
        Returns:
            List of CollisionRisk for significant risks
        """
        risks = []
        
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                t1 = tracks[i]
                t2 = tracks[j]
                
                risk = self.analyze_pair(
                    t1['id'], t1['center'], t1['velocity'], t1.get('bbox', np.zeros(4)),
                    t2['id'], t2['center'], t2['velocity'], t2.get('bbox', np.zeros(4))
                )
                
                if risk and risk.risk_level in ["critical", "high", "medium"]:
                    risks.append(risk)
        
        # Sort by risk (most critical first)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        risks.sort(key=lambda r: (risk_order.get(r.risk_level, 4), r.ttc_seconds))
        
        return risks
    
    def detect_tailgating(
        self,
        lead_pos: Tuple[float, float],
        lead_vel: Tuple[float, float],
        follow_pos: Tuple[float, float],
        follow_vel: Tuple[float, float]
    ) -> Optional[Dict]:
        """
        Detect unsafe following distance (tailgating).
        
        Uses 2-second rule: following distance should allow 2 seconds of reaction.
        
        Returns:
            Tailgating info dict or None
        """
        # Direction of travel (assuming horizontal movement dominant)
        follow_speed = np.sqrt(follow_vel[0]**2 + follow_vel[1]**2)
        
        if follow_speed < 0.5:  # Stationary
            return None
        
        # Distance between vehicles
        dx = lead_pos[0] - follow_pos[0]
        dy = lead_pos[1] - follow_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Safe following distance (pixels)
        # 2-second rule: distance = speed * time
        safe_distance = follow_speed * self.fps * self.safe_following_time
        
        if distance < safe_distance:
            gap_time = distance / (follow_speed * self.fps) if follow_speed > 0 else 0
            
            severity = "critical" if gap_time < 0.5 else "high" if gap_time < 1.0 else "warning"
            
            return {
                "type": "tailgating",
                "following_distance": distance,
                "safe_distance": safe_distance,
                "gap_time_seconds": round(gap_time, 2),
                "severity": severity,
                "follow_speed_px": follow_speed
            }
        
        return None
    
    def _record_near_miss(
        self,
        track1_id: int,
        track2_id: int,
        distance: float,
        rel_speed: float,
        was_predicted: bool
    ):
        """Record a near-miss event."""
        event = NearMissEvent(
            track_id_1=track1_id,
            track_id_2=track2_id,
            min_distance=distance,
            time_at_closest=time.time(),
            relative_speed=rel_speed,
            was_collision_predicted=was_predicted
        )
        self._near_misses.append(event)
        
        # Keep only recent near-misses
        if len(self._near_misses) > 100:
            self._near_misses.pop(0)
        
        logger.info(f"Near-miss recorded: tracks {track1_id}-{track2_id}, distance={distance:.1f}")
    
    def get_near_misses(self, since_seconds: float = 300) -> List[NearMissEvent]:
        """Get recent near-miss events."""
        cutoff = time.time() - since_seconds
        return [nm for nm in self._near_misses if nm.time_at_closest >= cutoff]
    
    def get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        """
        Get BGR color for risk visualization.
        
        Returns:
            (B, G, R) color tuple
        """
        colors = {
            "critical": (0, 0, 255),      # Red
            "high": (0, 128, 255),        # Orange
            "medium": (0, 255, 255),      # Yellow
            "low": (0, 255, 0),           # Green
            "none": (128, 128, 128)       # Gray
        }
        return colors.get(risk_level, (128, 128, 128))
    
    def reset(self):
        """Reset collision predictor state."""
        self._near_misses.clear()
        logger.info("CollisionPredictor reset")
