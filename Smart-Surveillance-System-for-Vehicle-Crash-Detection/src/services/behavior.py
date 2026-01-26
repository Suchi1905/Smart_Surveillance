"""
Dangerous Driving Behavior Analysis Service.

Detects various dangerous driving patterns:
- Swerving/erratic lane changes
- Wrong-way driving
- Sudden braking
- Aggressive acceleration
- Lane discipline violations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from .traffic_profiles import TrafficProfile
from enum import Enum
import logging
import time

MIN_HISTORY_LENGTH = 10  # minimum trajectory points

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Types of dangerous driving behaviors."""
    SWERVING = "swerving"
    WRONG_WAY = "wrong_way"
    SUDDEN_BRAKE = "sudden_brake"
    AGGRESSIVE_ACCEL = "aggressive_acceleration"
    ERRATIC_LANE_CHANGE = "erratic_lane_change"
    TAILGATING = "tailgating"
    RED_LIGHT_VIOLATION = "red_light_violation"
    STOP_SIGN_VIOLATION = "stop_sign_violation"
    SPEEDING = "speeding"
    DANGEROUS_OVERTAKE = "dangerous_overtake"
    # Composite/Sequence Behaviors
    ROAD_RAGE = "road_rage"            # Aggressive Accel + Tailgating
    RECKLESS_DRIVING = "reckless_driving" # Speeding + Swerving
    PANIC_MANEUVER = "panic_maneuver"  # Sudden Brake + Swerving


@dataclass
class BehaviorAlert:
    """
    Alert for detected dangerous behavior.
    
    Attributes:
        track_id: Vehicle track ID
        behavior_type: Type of dangerous behavior
        severity: "warning", "violation", "critical"
        confidence: Detection confidence (0-1)
        description: Human-readable description
        location: (x, y) where behavior occurred
        timestamp: Detection time
        evidence: Supporting data for the alert
    """
    track_id: int
    behavior_type: BehaviorType
    severity: str
    confidence: float
    description: str
    location: Tuple[float, float]
    timestamp: float
    evidence: Dict = field(default_factory=dict)


class BehaviorAnalyzer:
    """
    Analyzes vehicle trajectories for dangerous driving patterns.
    
    Uses trajectory analysis, speed changes, and lane position
    to identify potentially dangerous behaviors.
    """
    
    def __init__(
        self,
        profile: Optional['TrafficProfile'] = None,
        expected_flow_direction: float = 0.0,
        fps: float = 30.0
    ):
        """
        Initialize behavior analyzer.
        
        Args:
            profile: Traffic profile configuration (defaults to US if None)
            expected_flow_direction: Expected traffic direction in degrees
            fps: Camera frame rate
        """
        if profile is None:
            # Lazy import to avoid circular dependency if needed, though type checking might need it
            from .traffic_profiles import US_PROFILE
            self.profile = US_PROFILE
        else:
            self.profile = profile
            
        self.expected_flow_direction = expected_flow_direction
        self.lane_width_pixels = self.profile.lane_width_pixels
        self.fps = fps
        
        # Track behavior history
        self._alerts_history: List[BehaviorAlert] = []
        self._track_alerts: Dict[int, List[BehaviorAlert]] = {}
        
        logger.info(f"BehaviorAnalyzer initialized: profile={self.profile.name}, flow={expected_flow_direction}°")
    

    
    def add_external_alert(self, alert: BehaviorAlert):
        """Register an alert detected by another service (e.g. Tailgating)."""
        self._record_alert(alert)
        # Check sequences immediately after adding external alert
        seq_alert = self._detect_sequences(alert.track_id)
        if seq_alert:
            self._record_alert(seq_alert)
            return seq_alert
        return None

    def analyze_trajectory(
        self,
        track_id: int,
        trajectory: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        speed_kmh: Optional[float] = None,
        speed_limit: float = 60.0
    ) -> List[BehaviorAlert]:
        """
        Analyze a trajectory for dangerous behaviors.
        """
        alerts = []
        
        # 1. Check Speeding
        if speed_kmh is not None:
            speed_alert = self._analyze_speed(track_id, speed_kmh, speed_limit, trajectory[-1] if trajectory else (0,0))
            if speed_alert:
                alerts.append(speed_alert)
        
        if len(trajectory) < MIN_HISTORY_LENGTH:
            # Even with short trajectory, we might still return speed alerts
            for alert in alerts:
                self._record_alert(alert)
            return alerts
        
        # 2. Check trajectory-based behaviors
        swerve_alert = self._detect_swerving(track_id, trajectory)
        if swerve_alert:
            alerts.append(swerve_alert)
        
        brake_alert = self._detect_sudden_brake(track_id, velocities, trajectory[-1])
        if brake_alert:
            alerts.append(brake_alert)
        
        accel_alert = self._detect_aggressive_accel(track_id, velocities, trajectory[-1])
        if accel_alert:
            alerts.append(accel_alert)
        
        wrong_way_alert = self._detect_wrong_way(track_id, velocities, trajectory[-1])
        if wrong_way_alert:
            alerts.append(wrong_way_alert)
        
        lane_alert = self._detect_erratic_lane_change(track_id, trajectory)
        if lane_alert:
            alerts.append(lane_alert)
        
        # Record all primary alerts
        for alert in alerts:
            self._record_alert(alert)
            
        # 3. Check for Composite Sequences (e.g. Road Rage)
        seq_alert = self._detect_sequences(track_id)
        if seq_alert:
            alerts.append(seq_alert)
            self._record_alert(seq_alert)
        
        return alerts

    def _analyze_speed(
        self,
        track_id: int,
        speed_kmh: float,
        limit: float,
        location: Tuple[float, float]
    ) -> Optional[BehaviorAlert]:
        """Check for speeding."""
        if speed_kmh > limit * 1.2: # 20% buffer
            ratio = speed_kmh / limit
            severity = "violation" if ratio > 1.5 else "warning"
            
            # Avoid spamming: Check if we recently alerted for speeding for this track
            recent = self.get_track_alerts(track_id)
            last_speeding = next((a for a in reversed(recent) if a.behavior_type == BehaviorType.SPEEDING), None)
            
            if last_speeding and (time.time() - last_speeding.timestamp < 2.0):
                return None
                
            return BehaviorAlert(
                track_id=track_id,
                behavior_type=BehaviorType.SPEEDING,
                severity=severity,
                confidence=min(1.0, (ratio - 1.0)),
                description=f"Speeding: {speed_kmh:.0f} km/h (Limit {limit})",
                location=location,
                timestamp=time.time(),
                evidence={"speed": speed_kmh, "limit": limit}
            )
        return None

    def _detect_sequences(self, track_id: int) -> Optional[BehaviorAlert]:
        """Detect dangerous sequences of behaviors."""
        alerts = self.get_track_alerts(track_id)
        if not alerts:
            return None
        
        # Look at last 5 seconds
        now = time.time()
        recent = [a for a in alerts if now - a.timestamp < 5.0]
        types = {a.behavior_type for a in recent}
        
        # 1. Road Rage: Aggressive Accel + Tailgating
        if BehaviorType.AGGRESSIVE_ACCEL in types and BehaviorType.TAILGATING in types:
            # Check if we haven't already flagged road rage recently
            if BehaviorType.ROAD_RAGE not in types:
                return BehaviorAlert(
                    track_id=track_id,
                    behavior_type=BehaviorType.ROAD_RAGE,
                    severity="critical",
                    confidence=0.9,
                    description="🔥 ROAD RAGE DETECTED: Aggressive driving + Tailgating",
                    location=recent[-1].location,
                    timestamp=now,
                    evidence={"components": ["aggressive_acceleration", "tailgating"]}
                )

        # 2. Reckless Driving: Speeding + Swerving/Erratic Lane Change
        if BehaviorType.SPEEDING in types and (BehaviorType.SWERVING in types or BehaviorType.ERRATIC_LANE_CHANGE in types):
            if BehaviorType.RECKLESS_DRIVING not in types:
                return BehaviorAlert(
                    track_id=track_id,
                    behavior_type=BehaviorType.RECKLESS_DRIVING,
                    severity="critical",
                    confidence=0.85,
                    description="⚠️ RECKLESS DRIVING: High speed swerving",
                    location=recent[-1].location,
                    timestamp=now,
                    evidence={"components": ["speeding", "swerving"]}
                )
                
        # 3. Panic Maneuver: Sudden Brake + Swerving
        if BehaviorType.SUDDEN_BRAKE in types and BehaviorType.SWERVING in types:
             if BehaviorType.PANIC_MANEUVER not in types:
                return BehaviorAlert(
                    track_id=track_id,
                    behavior_type=BehaviorType.PANIC_MANEUVER,
                    severity="warning",
                    confidence=0.75,
                    description="Panic Maneuver Detected",
                    location=recent[-1].location,
                    timestamp=now,
                    evidence={"components": ["sudden_brake", "swerving"]}
                )
        
        return None
    
    def _detect_swerving(
        self,
        track_id: int,
        trajectory: List[Tuple[float, float]]
    ) -> Optional[BehaviorAlert]:
        """Detect swerving/unstable driving."""
        if len(trajectory) < 5:
            return None
        
        recent = trajectory[-10:] if len(trajectory) >= 10 else trajectory
        
        # Calculate lateral deviation from straight-line path
        start = np.array(recent[0])
        end = np.array(recent[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 10:  # Nearly stationary
            return None
        
        line_unit = line_vec / line_len
        
        # Calculate perpendicular distances
        deviations = []
        for point in recent[1:-1]:
            p = np.array(point)
            proj_len = np.dot(p - start, line_unit)
            proj_point = start + proj_len * line_unit
            deviation = np.linalg.norm(p - proj_point)
            deviations.append(deviation)
        
        if not deviations:
            return None
        
        variance = np.var(deviations)
        max_deviation = np.max(deviations)
        
        if variance > self.profile.swerve_variance_threshold or max_deviation > self.lane_width_pixels * 0.5:
            severity = "critical" if max_deviation > self.lane_width_pixels else "warning"
            
            return BehaviorAlert(
                track_id=track_id,
                behavior_type=BehaviorType.SWERVING,
                severity=severity,
                confidence=min(1.0, variance / self.profile.swerve_variance_threshold),
                description=f"Swerving detected: {max_deviation:.0f}px deviation",
                location=recent[-1],
                timestamp=time.time(),
                evidence={
                    "max_deviation": max_deviation,
                    "variance": variance,
                    "trajectory_points": len(recent)
                }
            )
        
        return None
    
    def _detect_sudden_brake(
        self,
        track_id: int,
        velocities: List[Tuple[float, float]],
        current_position: Tuple[float, float]
    ) -> Optional[BehaviorAlert]:
        """Detect sudden braking."""
        if len(velocities) < 3:
            return None
        
        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities[-5:]]
        
        if len(speeds) < 2:
            return None
        
        # Check for sudden deceleration
        for i in range(1, len(speeds)):
            if speeds[i-1] > 5:  # Only if was moving
                decel_ratio = (speeds[i-1] - speeds[i]) / speeds[i-1]
                
                if decel_ratio > self.profile.sudden_brake_ratio:
                    severity = "critical" if decel_ratio > 0.6 else "warning"
                    
                    return BehaviorAlert(
                        track_id=track_id,
                        behavior_type=BehaviorType.SUDDEN_BRAKE,
                        severity=severity,
                        confidence=min(1.0, decel_ratio / 0.5),
                        description=f"Sudden brake: {decel_ratio*100:.0f}% speed drop",
                        location=current_position,
                        timestamp=time.time(),
                        evidence={
                            "speed_before": speeds[i-1],
                            "speed_after": speeds[i],
                            "deceleration_ratio": decel_ratio
                        }
                    )
        
        return None
    
    def _detect_aggressive_accel(
        self,
        track_id: int,
        velocities: List[Tuple[float, float]],
        current_position: Tuple[float, float]
    ) -> Optional[BehaviorAlert]:
        """Detect aggressive acceleration."""
        if len(velocities) < 3:
            return None
        
        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities[-5:]]
        
        if len(speeds) < 2:
            return None
        
        for i in range(1, len(speeds)):
            if speeds[i-1] > 1:  # Was moving
                accel_ratio = (speeds[i] - speeds[i-1]) / (speeds[i-1] + 0.1)
                
                if accel_ratio > self.profile.aggressive_accel_ratio:
                    severity = "warning" if accel_ratio < 0.6 else "violation"
                    
                    return BehaviorAlert(
                        track_id=track_id,
                        behavior_type=BehaviorType.AGGRESSIVE_ACCEL,
                        severity=severity,
                        confidence=min(1.0, accel_ratio / 0.5),
                        description=f"Aggressive acceleration: {accel_ratio*100:.0f}% speed increase",
                        location=current_position,
                        timestamp=time.time(),
                        evidence={
                            "speed_before": speeds[i-1],
                            "speed_after": speeds[i],
                            "acceleration_ratio": accel_ratio
                        }
                    )
        
        return None
    
    def _detect_wrong_way(
        self,
        track_id: int,
        velocities: List[Tuple[float, float]],
        current_position: Tuple[float, float]
    ) -> Optional[BehaviorAlert]:
        """Detect wrong-way driving."""
        if len(velocities) < 3:
            return None
        
        # Average recent velocity direction
        recent_vels = velocities[-5:] if len(velocities) >= 5 else velocities
        avg_vx = np.mean([v[0] for v in recent_vels])
        avg_vy = np.mean([v[1] for v in recent_vels])
        
        speed = np.sqrt(avg_vx**2 + avg_vy**2)
        if speed < 2:  # Too slow to determine direction
            return None
        
        # Calculate direction
        direction = np.degrees(np.arctan2(avg_vy, avg_vx))
        if direction < 0:
            direction += 360
        
        # Compare with expected flow direction
        angle_diff = abs(direction - self.expected_flow_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Wrong way if going opposite direction (±45° of opposite)
        if angle_diff > 135:
            return BehaviorAlert(
                track_id=track_id,
                behavior_type=BehaviorType.WRONG_WAY,
                severity="critical",
                confidence=min(1.0, (angle_diff - 135) / 45),
                description=f"⚠️ WRONG-WAY DRIVER: heading {direction:.0f}° vs expected {self.expected_flow_direction:.0f}°",
                location=current_position,
                timestamp=time.time(),
                evidence={
                    "actual_direction": direction,
                    "expected_direction": self.expected_flow_direction,
                    "angle_difference": angle_diff
                }
            )
        
        return None
    
    def _detect_erratic_lane_change(
        self,
        track_id: int,
        trajectory: List[Tuple[float, float]]
    ) -> Optional[BehaviorAlert]:
        """Detect erratic/sudden lane changes."""
        if len(trajectory) < 8:
            return None
        
        recent = trajectory[-15:] if len(trajectory) >= 15 else trajectory
        
        # Calculate direction changes (angles between consecutive segments)
        angles = []
        for i in range(2, len(recent)):
            v1 = (recent[i-1][0] - recent[i-2][0], recent[i-1][1] - recent[i-2][1])
            v2 = (recent[i][0] - recent[i-1][0], recent[i][1] - recent[i-1][1])
            
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 > 1 and len2 > 1:
                cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
        
        if not angles:
            return None
        
        max_angle = np.max(angles)
        
        if max_angle > self.profile.swerve_angle_threshold:
            severity = "critical" if max_angle > 30 else "warning"
            
            return BehaviorAlert(
                track_id=track_id,
                behavior_type=BehaviorType.ERRATIC_LANE_CHANGE,
                severity=severity,
                confidence=min(1.0, max_angle / 45),
                description=f"Erratic lane change: {max_angle:.0f}° sharp turn",
                location=recent[-1],
                timestamp=time.time(),
                evidence={
                    "max_angle_change": max_angle,
                    "angles": angles[-5:]
                }
            )
        
        return None
    
    def _record_alert(self, alert: BehaviorAlert):
        """Record an alert to history."""
        self._alerts_history.append(alert)
        
        if alert.track_id not in self._track_alerts:
            self._track_alerts[alert.track_id] = []
        self._track_alerts[alert.track_id].append(alert)
        
        # Limit history size
        if len(self._alerts_history) > 500:
            self._alerts_history.pop(0)
        
        logger.warning(
            f"Behavior Alert: {alert.behavior_type.value} - "
            f"Track {alert.track_id} - {alert.severity} - {alert.description}"
        )
    
    def get_recent_alerts(
        self,
        since_seconds: float = 60,
        severity_filter: Optional[str] = None
    ) -> List[BehaviorAlert]:
        """Get recent behavior alerts."""
        cutoff = time.time() - since_seconds
        alerts = [a for a in self._alerts_history if a.timestamp >= cutoff]
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        return alerts
    
    def get_track_alerts(self, track_id: int) -> List[BehaviorAlert]:
        """Get all alerts for a specific track."""
        return self._track_alerts.get(track_id, [])
    
    def get_alert_count_by_type(self) -> Dict[str, int]:
        """Get count of alerts by behavior type."""
        counts = {}
        for alert in self._alerts_history:
            key = alert.behavior_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def set_expected_flow_direction(self, direction: float):
        """
        Set expected traffic flow direction.
        
        Args:
            direction: Direction in degrees (0=right, 90=down, 180=left, 270=up)
        """
        self.expected_flow_direction = direction % 360
        logger.info(f"Expected flow direction set to {self.expected_flow_direction}°")
    
    def reset(self):
        """Reset analyzer state."""
        self._alerts_history.clear()
        self._track_alerts.clear()
        logger.info("BehaviorAnalyzer reset")
