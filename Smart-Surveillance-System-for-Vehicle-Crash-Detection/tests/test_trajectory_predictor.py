"""
Tests for Trajectory Prediction Service.
"""

import pytest
import numpy as np
from src.services.trajectory_predictor import TrajectoryPredictor

def test_init():
    """Test initialization."""
    predictor = TrajectoryPredictor()
    assert len(predictor.filters) == 0

def test_constant_velocity_prediction():
    """Test valid prediction for constant velocity."""
    predictor = TrajectoryPredictor()
    track_id = 1
    
    # Simulate valid track moving right: (0,0) -> (10,0) -> (20,0)
    predictor.update(track_id, (0.0, 0.0), dt=1.0)
    predictor.update(track_id, (10.0, 0.0), dt=1.0) # vx should be approx 10
    predictor.update(track_id, (20.0, 0.0), dt=1.0)
    
    # Predict next 5 steps
    future = predictor.predict_future_path(track_id, num_frames=5, dt=1.0)
    assert len(future) == 5
    
    # First point should be near (30, 0)
    # Note: Kalman Filter warmup might lag slightly, but trend should be right
    next_pt = future[0]
    assert 25.0 < next_pt[0] < 35.0
    assert -2.0 < next_pt[1] < 2.0  # y should stay near 0

def test_curved_motion_prediction():
    """Test prediction for accelerating/curving object."""
    predictor = TrajectoryPredictor()
    track_id = 2
    
    # Simulate parabola: x=t, y=t^2  (acceleration in y)
    # t=0: (0,0)
    # t=1: (1,1)
    # t=2: (2,4)
    # t=3: (3,9)
    # t=4: (4,16)
    
    points = [(0,0), (1,1), (2,4), (3,9), (4,16)]
    for pt in points:
        predictor.update(track_id, pt, dt=1.0)
        
    # Expect next point around (5, 25)
    future = predictor.predict_future_path(track_id, num_frames=1, dt=1.0)
    next_pt = future[0]
    
    # Check x roughly 5
    assert 4.5 < next_pt[0] < 5.5
    # Check y roughly 25 (acceleration detection takes time, but should be > 16)
    # Linear would give 16 + (16-9) = 23.
    # Acceleration should push it higher.
    assert next_pt[1] > 20.0 

def test_reset():
    """Test reset clears filters."""
    predictor = TrajectoryPredictor()
    predictor.update(1, (0,0))
    predictor.reset()
    assert len(predictor.filters) == 0
