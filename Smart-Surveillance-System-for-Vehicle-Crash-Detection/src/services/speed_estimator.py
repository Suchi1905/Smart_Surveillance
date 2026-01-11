"""
Speed Estimation Service.

This module provides pixel-to-real-world speed conversion using
camera calibration and perspective correction.

Supports multiple estimation methods:
1. Optical flow based
2. Bounding box displacement
3. Calibrated reference points
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SpeedCalibration:
    """
    Camera calibration parameters for speed estimation.
    
    Attributes:
        pixels_per_meter: Conversion factor (calibrated)
        fps: Camera frame rate
        lane_width_meters: Known lane width for auto-calibration
        perspective_matrix: 3x3 transformation matrix
    """
    pixels_per_meter: float = 10.0  # Default: 10 pixels = 1 meter
    fps: float = 30.0
    lane_width_meters: float = 3.5  # Standard lane width
    perspective_matrix: Optional[np.ndarray] = None


@dataclass
class SpeedMeasurement:
    """
    Speed measurement result.
    
    Attributes:
        track_id: Associated track ID
        speed_kmh: Speed in km/h
        speed_mph: Speed in mph
        speed_pixels: Raw speed in pixels/frame
        direction: Movement direction in degrees (0=right, 90=down)
        confidence: Estimation confidence (0-1)
        is_stationary: Whether vehicle is stopped
        timestamp: Measurement time
    """
    track_id: int
    speed_kmh: float
    speed_mph: float
    speed_pixels: float
    direction: float
    confidence: float
    is_stationary: bool
    timestamp: float


class SpeedEstimator:
    """
    Real-time speed estimation from tracked vehicles.
    
    Uses multiple frames of trajectory data to smooth estimates
    and reduce noise from detection jitter.
    """
    
    # Speed thresholds (km/h)
    STATIONARY_THRESHOLD = 5.0
    SPEEDING_URBAN = 50.0
    SPEEDING_HIGHWAY = 120.0
    DANGEROUS_SPEED = 150.0
    
    def __init__(
        self,
        calibration: Optional[SpeedCalibration] = None,
        smoothing_window: int = 5,
        min_track_length: int = 3
    ):
        """
        Initialize speed estimator.
        
        Args:
            calibration: Camera calibration parameters
            smoothing_window: Number of frames for moving average
            min_track_length: Minimum trajectory points needed
        """
        self.calibration = calibration or SpeedCalibration()
        self.smoothing_window = smoothing_window
        self.min_track_length = min_track_length
        
        # Speed history per track for smoothing
        self._speed_history: Dict[int, List[float]] = {}
        
        logger.info(
            f"SpeedEstimator initialized: "
            f"{self.calibration.pixels_per_meter} px/m, "
            f"{self.calibration.fps} fps"
        )
    
    def calibrate_from_lane(
        self,
        lane_points: List[Tuple[int, int]],
        lane_width_meters: float = 3.5
    ):
        """
        Auto-calibrate using known lane width.
        
        Args:
            lane_points: Two points marking lane edges [(x1,y1), (x2,y2)]
            lane_width_meters: Known lane width in meters
        """
        if len(lane_points) < 2:
            logger.warning("Need at least 2 points for lane calibration")
            return
        
        p1, p2 = lane_points[0], lane_points[1]
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        if pixel_distance > 0:
            self.calibration.pixels_per_meter = pixel_distance / lane_width_meters
            self.calibration.lane_width_meters = lane_width_meters
            logger.info(
                f"Calibrated: {self.calibration.pixels_per_meter:.2f} pixels/meter"
            )
    
    def calibrate_from_reference(
        self,
        pixel_distance: float,
        real_distance_meters: float
    ):
        """
        Calibrate using a known reference distance.
        
        Args:
            pixel_distance: Distance in pixels
            real_distance_meters: Actual distance in meters
        """
        if pixel_distance > 0 and real_distance_meters > 0:
            self.calibration.pixels_per_meter = pixel_distance / real_distance_meters
            logger.info(
                f"Calibrated: {self.calibration.pixels_per_meter:.2f} pixels/meter"
            )
    
    def set_fps(self, fps: float):
        """Set camera frame rate for speed calculation."""
        self.calibration.fps = fps
        logger.info(f"FPS set to {fps}")
    
    def estimate_from_trajectory(
        self,
        track_id: int,
        trajectory: List[Tuple[float, float]]
    ) -> Optional[SpeedMeasurement]:
        """
        Estimate speed from trajectory history.
        
        Args:
            track_id: Track identifier
            trajectory: List of (x, y) center positions
        
        Returns:
            SpeedMeasurement or None if insufficient data
        """
        if len(trajectory) < self.min_track_length:
            return None
        
        # Use recent positions for velocity calculation
        recent = trajectory[-self.smoothing_window:] if len(trajectory) >= self.smoothing_window else trajectory
        
        # Calculate average displacement per frame
        total_dx = 0.0
        total_dy = 0.0
        count = 0
        
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            total_dx += dx
            total_dy += dy
            count += 1
        
        if count == 0:
            return None
        
        avg_dx = total_dx / count
        avg_dy = total_dy / count
        
        # Speed in pixels per frame
        speed_pixels = np.sqrt(avg_dx**2 + avg_dy**2)
        
        # Convert to real-world speed
        # pixels/frame -> meters/second -> km/h
        meters_per_frame = speed_pixels / self.calibration.pixels_per_meter
        meters_per_second = meters_per_frame * self.calibration.fps
        speed_kmh = meters_per_second * 3.6  # m/s to km/h
        
        # Apply smoothing from history
        if track_id not in self._speed_history:
            self._speed_history[track_id] = []
        
        self._speed_history[track_id].append(speed_kmh)
        if len(self._speed_history[track_id]) > self.smoothing_window:
            self._speed_history[track_id].pop(0)
        
        # Smoothed speed (moving average)
        smoothed_speed = np.mean(self._speed_history[track_id])
        
        # Calculate direction (0 = right, 90 = down, etc.)
        direction = np.degrees(np.arctan2(avg_dy, avg_dx))
        if direction < 0:
            direction += 360
        
        # Confidence based on trajectory consistency
        if len(self._speed_history[track_id]) >= 3:
            std = np.std(self._speed_history[track_id])
            mean = np.mean(self._speed_history[track_id])
            cv = std / (mean + 1e-6)  # Coefficient of variation
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5
        
        return SpeedMeasurement(
            track_id=track_id,
            speed_kmh=round(smoothed_speed, 1),
            speed_mph=round(smoothed_speed * 0.621371, 1),
            speed_pixels=round(speed_pixels, 2),
            direction=round(direction, 1),
            confidence=round(confidence, 2),
            is_stationary=smoothed_speed < self.STATIONARY_THRESHOLD,
            timestamp=time.time()
        )
    
    def estimate_from_velocity(
        self,
        track_id: int,
        velocity: Tuple[float, float]
    ) -> SpeedMeasurement:
        """
        Estimate speed from velocity vector.
        
        Args:
            track_id: Track identifier
            velocity: (vx, vy) velocity in pixels/frame
        
        Returns:
            SpeedMeasurement
        """
        vx, vy = velocity
        speed_pixels = np.sqrt(vx**2 + vy**2)
        
        # Convert to km/h
        meters_per_frame = speed_pixels / self.calibration.pixels_per_meter
        meters_per_second = meters_per_frame * self.calibration.fps
        speed_kmh = meters_per_second * 3.6
        
        # Direction
        direction = np.degrees(np.arctan2(vy, vx))
        if direction < 0:
            direction += 360
        
        return SpeedMeasurement(
            track_id=track_id,
            speed_kmh=round(speed_kmh, 1),
            speed_mph=round(speed_kmh * 0.621371, 1),
            speed_pixels=round(speed_pixels, 2),
            direction=round(direction, 1),
            confidence=0.7,  # Lower confidence for single-frame estimate
            is_stationary=speed_kmh < self.STATIONARY_THRESHOLD,
            timestamp=time.time()
        )
    
    def is_speeding(
        self,
        speed_kmh: float,
        zone_type: str = "urban"
    ) -> Tuple[bool, str]:
        """
        Check if speed exceeds limit for zone type.
        
        Args:
            speed_kmh: Current speed
            zone_type: "urban", "highway", or "school"
        
        Returns:
            (is_speeding, severity) where severity is "warning", "violation", "dangerous"
        """
        limits = {
            "school": 30.0,
            "urban": 50.0,
            "highway": 120.0
        }
        
        limit = limits.get(zone_type, 50.0)
        
        if speed_kmh >= self.DANGEROUS_SPEED:
            return True, "dangerous"
        elif speed_kmh >= limit * 1.5:
            return True, "violation"
        elif speed_kmh >= limit:
            return True, "warning"
        else:
            return False, "normal"
    
    def get_speed_color(self, speed_kmh: float, zone_type: str = "urban") -> Tuple[int, int, int]:
        """
        Get BGR color for speed visualization.
        
        Returns:
            (B, G, R) color tuple
        """
        is_speeding, severity = self.is_speeding(speed_kmh, zone_type)
        
        if severity == "dangerous":
            return (0, 0, 255)  # Red
        elif severity == "violation":
            return (0, 128, 255)  # Orange
        elif severity == "warning":
            return (0, 255, 255)  # Yellow
        else:
            return (0, 255, 0)  # Green
    
    def clear_track_history(self, track_id: int):
        """Clear speed history for a track (when track is lost)."""
        if track_id in self._speed_history:
            del self._speed_history[track_id]
    
    def reset(self):
        """Reset all speed history."""
        self._speed_history.clear()
        logger.info("SpeedEstimator reset")


class SpeedZone:
    """
    Represents a speed monitoring zone with specific limits.
    """
    
    def __init__(
        self,
        name: str,
        polygon: List[Tuple[int, int]],
        speed_limit: float,
        zone_type: str = "urban"
    ):
        """
        Initialize speed zone.
        
        Args:
            name: Zone name (e.g., "School Zone A")
            polygon: List of (x, y) points defining zone boundary
            speed_limit: Speed limit in km/h
            zone_type: Zone type for severity calculation
        """
        self.name = name
        self.polygon = np.array(polygon)
        self.speed_limit = speed_limit
        self.zone_type = zone_type
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside this zone."""
        # Ray casting algorithm
        x, y = point
        n = len(self.polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def check_violation(
        self,
        point: Tuple[float, float],
        speed_kmh: float
    ) -> Optional[Dict]:
        """
        Check for speed violation in this zone.
        
        Returns:
            Violation dict or None
        """
        if not self.contains_point(point):
            return None
        
        if speed_kmh > self.speed_limit:
            excess = speed_kmh - self.speed_limit
            severity = "warning" if excess < 10 else "violation" if excess < 30 else "dangerous"
            
            return {
                "zone_name": self.name,
                "zone_type": self.zone_type,
                "speed_limit": self.speed_limit,
                "actual_speed": speed_kmh,
                "excess_speed": round(excess, 1),
                "severity": severity
            }
        
        return None
