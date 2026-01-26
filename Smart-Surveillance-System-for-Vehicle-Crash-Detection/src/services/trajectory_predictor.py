"""
Trajectory Prediction Service.

Uses Kalman Filtering (Constant Acceleration Model) to predict future vehicle paths.
This handles non-linear motion (curves, acceleration) better than linear extrapolation.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class TrajectoryPredictor:
    """
    Predicts future trajectories of vehicles using Kalman Filtering.
    """
    
    def __init__(self):
        # Map track_id -> KalmanFilter
        self.filters: Dict[int, cv2.KalmanFilter] = {}
        # Map track_id -> last update time (to clean up old filters)
        self.last_update: Dict[int, float] = {}
        
        # Kalman Filter Parameters
        # State: [x, y, vx, vy, ax, ay]
        self.state_num = 6
        self.measure_num = 2  # [x, y]
        
    def _create_filter(self) -> cv2.KalmanFilter:
        """Create and configure a new Kalman Filter."""
        kf = cv2.KalmanFilter(self.state_num, self.measure_num, 0)
        
        # Transition Matrix (F) - update with dt later
        # [1 0 dt 0 0.5*dt^2 0]
        # [0 1 0 dt 0 0.5*dt^2]
        # ...
        kf.transitionMatrix = np.eye(self.state_num, dtype=np.float32)
        
        # Measurement Matrix (H) - we observe x, y
        kf.measurementMatrix = np.zeros((self.measure_num, self.state_num), np.float32)
        kf.measurementMatrix[0, 0] = 1
        kf.measurementMatrix[1, 1] = 1
        
        # Process Noise Covariance (Q)
        # Determines how much we trust the model vs how much we expect "jerk"
        cv2.setIdentity(kf.processNoiseCov, 1e-2)
        
        # Measurement Noise Covariance (R)
        # Trust measurement quite a bit
        cv2.setIdentity(kf.measurementNoiseCov, 1e-1)
        
        # Error Covariance (P) - initial uncertainty
        cv2.setIdentity(kf.errorCovPost, 1.0)
        
        return kf

    def update(self, track_id: int, center: Tuple[float, float], dt: float = 1.0/30.0):
        """
        Update the filter with a new measurement.
        
        Args:
            track_id: Vehicle ID
            center: Measured (x, y) center
            dt: Time since last frame (seconds)
        """
        if track_id not in self.filters:
            # Initialize new filter
            kf = self._create_filter()
            # Set initial state
            kf.statePost = np.array([
                [center[0]], [center[1]], 
                [0], [0], 
                [0], [0]
            ], dtype=np.float32)
            self.filters[track_id] = kf
        
        kf = self.filters[track_id]
        
        # Update Transition Matrix with actual dt
        # x = x + vx*dt + 0.5*ax*dt^2
        # v = v + ax*dt
        # a = a
        F = np.eye(self.state_num, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt**2
        F[1, 5] = 0.5 * dt**2
        F[2, 4] = dt
        F[3, 5] = dt
        
        kf.transitionMatrix = F
        
        # Predict phase (to update state estimate before measurement)
        kf.predict()
        
        # Correct phase (incorporate new measurement)
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
        kf.correct(measurement)
        
        # Cleanup logic (optional, managed by last_update)
        # We manually track active IDs in main service usually, 
        # but let's keep a cleanup mechanism if tracking stops.
        
    def predict_future_path(
        self, 
        track_id: int, 
        num_frames: int = 30, 
        dt: float = 1.0/30.0
    ) -> List[Tuple[float, float]]:
        """
        Predict future path for a track.
        
        Args:
            track_id: Vehicle ID
            num_frames: How many frames ahead to predict
            dt: Time step per frame
            
        Returns:
            List of (x, y) predicted points
        """
        if track_id not in self.filters:
            return []
            
        kf = self.filters[track_id]
        
        # We must NOT modify the actual filter state during lookahead.
        # So we clone the state.
        # cv2.KalmanFilter doesn't have a deepcopy method that works easily for internal C++ state.
        # We will manually simulate the projection using the current state estimate.
        
        current_state = kf.statePost.copy()
        x, y = current_state[0, 0], current_state[1, 0]
        vx, vy = current_state[2, 0], current_state[3, 0]
        ax, ay = current_state[4, 0], current_state[5, 0]
        
        path = []
        
        for _ in range(num_frames):
            # Constant Acceleration Model Step
            # x_new = x + vx*dt + 0.5*ax*dt^2
            x = x + vx * dt + 0.5 * ax * dt**2
            y = y + vy * dt + 0.5 * ay * dt**2
            
            # vx_new = vx + ax*dt
            vx = vx + ax * dt
            vy = vy + ay * dt
            
            # ax remains constant (or we could decay it)
            
            path.append((float(x), float(y)))
            
        return path
    
    def reset(self):
        """Reset predictor."""
        self.filters.clear()
