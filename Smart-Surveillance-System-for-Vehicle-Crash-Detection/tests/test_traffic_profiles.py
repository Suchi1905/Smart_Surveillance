"""
Tests for Traffic Profile Configuration.
"""

import pytest
from src.services.traffic_profiles import get_traffic_profile, US_PROFILE, INDIAN_PROFILE
from src.services.behavior import BehaviorAnalyzer
from src.services.collision import CollisionPredictor

def test_get_traffic_profile():
    """Test retrieving profiles by name."""
    # Test valid names
    assert get_traffic_profile("us").name == "us"
    assert get_traffic_profile("indian").name == "indian"
    assert get_traffic_profile("european").name == "us"  # check alias
    
    # Test case insensitivity
    assert get_traffic_profile("INDIAN").name == "indian"
    
    # Test default
    assert get_traffic_profile("unknown").name == "indian"

def test_indian_profile_values():
    """Verify Indian profile values are different from US."""
    us = US_PROFILE
    indian = INDIAN_PROFILE
    
    # Indian traffic allows closer following
    assert indian.min_safe_following_time < us.min_safe_following_time
    
    # Indian traffic tolerates more swerving
    assert indian.swerve_angle_threshold > us.swerve_angle_threshold
    assert indian.swerve_variance_threshold > us.swerve_variance_threshold

def test_behavior_analyzer_with_profile():
    """Test BehaviorAnalyzer initialization with profile."""
    # Default (US)
    analyzer_us = BehaviorAnalyzer()
    assert analyzer_us.profile == US_PROFILE
    assert analyzer_us.lane_width_pixels == 100.0
    
    # Indian
    analyzer_ind = BehaviorAnalyzer(profile=INDIAN_PROFILE)
    assert analyzer_ind.profile == INDIAN_PROFILE
    assert analyzer_ind.lane_width_pixels == 90.0  # From profile

def test_collision_predictor_with_profile():
    """Test CollisionPredictor initialization with profile."""
    # Default (US)
    predictor_us = CollisionPredictor()
    assert predictor_us.profile == US_PROFILE
    assert predictor_us.safe_following_time == 2.0
    
    # Indian
    predictor_ind = CollisionPredictor(profile=INDIAN_PROFILE)
    assert predictor_ind.profile == INDIAN_PROFILE
    assert predictor_ind.safe_following_time == 1.0  # From profile
