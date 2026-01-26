"""
Traffic Profile Configuration.

Defines adaptable traffic parameters for different regions (e.g., US vs India).
Indian traffic is characterized by:
- Higher density (shorter following distances)
- More chaotic movement (higher swerve tolerance)
- Lower average speeds but rapid changes
- Mixed vehicle types
"""

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class TrafficProfile:
    """Configuration params for traffic behavior analysis."""
    name: str
    
    # Collision Thresholds
    min_safe_following_time: float  # seconds
    ttc_critical: float  # seconds
    ttc_high: float      # seconds
    ttc_medium: float    # seconds
    
    # Behavior Thresholds
    swerve_angle_threshold: float    # degrees
    swerve_variance_threshold: float # pixel variance
    sudden_brake_ratio: float        # ratio of speed drop
    aggressive_accel_ratio: float    # ratio of speed increase
    
    # Lane Discipline (pixels)
    lane_width_pixels: float         # expected lane width
    erratic_lane_change_angle: float # degrees

    # Risk Scoring Weights
    weight_collision: float
    weight_behavior: float
    weight_speed: float


# Standard/US Traffic Profile (Structured, clear lanes, higher speeds)
US_PROFILE = TrafficProfile(
    name="us",
    min_safe_following_time=2.0,
    ttc_critical=1.5,
    ttc_high=2.5,
    ttc_medium=5.0,
    swerve_angle_threshold=15.0,
    swerve_variance_threshold=50.0,
    sudden_brake_ratio=0.3,
    aggressive_accel_ratio=0.4,
    lane_width_pixels=100.0,
    erratic_lane_change_angle=30.0,
    weight_collision=0.5,
    weight_behavior=0.3,
    weight_speed=0.2
)

# Indian Traffic Profile (Chaotic, high density, tight gaps)
INDIAN_PROFILE = TrafficProfile(
    name="indian",
    min_safe_following_time=1.0,     # Shorter following distance is normal
    ttc_critical=1.0,                # Lower threshold due to tighter maneuvering
    ttc_high=1.8,
    ttc_medium=3.5,
    swerve_angle_threshold=25.0,     # Higher tolerance for lateral movement
    swerve_variance_threshold=80.0,  # More zigzagging allowed
    sudden_brake_ratio=0.4,          # More intense braking is common
    aggressive_accel_ratio=0.5,      # Rapid acceleration in gaps
    lane_width_pixels=90.0,          # Often narrower effective lanes
    erratic_lane_change_angle=45.0,  # Sharper cuts allowed
    weight_collision=0.6,            # Collision risk is prioritized over erratic behavior
    weight_behavior=0.2,             # Erratic behavior is less penalized
    weight_speed=0.2
)

PARAMS: Dict[str, TrafficProfile] = {
    "us": US_PROFILE,
    "indian": INDIAN_PROFILE,
    "european": US_PROFILE  # Default alias
}

def get_traffic_profile(profile_name: str) -> TrafficProfile:
    """Get traffic profile by name, default to 'indian'."""
    return PARAMS.get(profile_name.lower(), INDIAN_PROFILE)
