"""
Severity Triage System for accident severity classification.

This module implements temporal vehicle tracking and severity analysis
using IoU-based track assignment and velocity drop detection.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class Detection:
    """Detection data structure."""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    frame_number: int


@dataclass
class SeverityResult:
    """Severity analysis result."""
    track_id: int
    severity_index: float
    severity_category: str
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]


class SeverityTriageSystem:
    """
    Temporal vehicle tracking and severity classification system.
    
    Uses IoU-based track assignment and velocity analysis to determine
    crash severity categories: Severe, Moderate, Mild, or Monitoring.
    
    Attributes:
        buffer_size: Number of frames to retain per track (default: 10)
        iou_threshold: Minimum IoU for severe classification (default: 0.3)
    
    Example:
        >>> triage = SeverityTriageSystem()
        >>> detections = [(box, conf, class_name), ...]
        >>> results = triage.analyze_accident(detections, frame_number)
        >>> for result in results:
        ...     print(f"{result.severity_category}: {result.severity_index:.2f}")
    """
    
    def __init__(self, buffer_size: int = 10, iou_threshold: float = 0.3):
        """
        Initialize the severity triage system.
        
        Args:
            buffer_size: Number of frames to retain per vehicle track
            iou_threshold: Minimum IoU threshold for severe classification
        """
        self.vehicle_tracks: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.next_track_id = 0
        self.track_assignments: Dict[int, int] = {}
    
    def calculate_iou(
        self, 
        box1: Tuple[int, int, int, int], 
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_box_center(self, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Get center point of bounding box.
        
        Args:
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Center point (cx, cy)
        """
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def calculate_velocity(self, track_history: List[dict]) -> float:
        """
        Calculate pixel displacement velocity from track history.
        
        Args:
            track_history: List of track history entries
            
        Returns:
            Velocity in pixels per frame
        """
        if len(track_history) < 2:
            return 0.0
        
        recent = track_history[-1]
        previous = track_history[-2]
        
        center_recent = self.get_box_center(recent['box'])
        center_previous = self.get_box_center(previous['box'])
        
        # Euclidean distance
        displacement = np.sqrt(
            (center_recent[0] - center_previous[0])**2 + 
            (center_recent[1] - center_previous[1])**2
        )
        
        return float(displacement)
    
    def assign_track_id(
        self, 
        current_box: Tuple[int, int, int, int], 
        frame_number: int
    ) -> int:
        """
        Assign track ID to detection based on IoU with previous tracks.
        
        Args:
            current_box: Current detection bounding box
            frame_number: Current frame number
            
        Returns:
            Assigned track ID
        """
        best_iou = 0.0
        best_track_id = None
        
        # Try to match with existing tracks
        for track_id, history in self.vehicle_tracks.items():
            if len(history) > 0:
                last_detection = history[-1]
                # Only match if detection is recent (within 5 frames)
                if frame_number - last_detection['frame'] < 5:
                    iou = self.calculate_iou(current_box, last_detection['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
        
        # If good match found, use existing track
        if best_iou > 0.3:
            return best_track_id
        
        # Otherwise create new track
        new_id = self.next_track_id
        self.next_track_id += 1
        return new_id
    
    def update_track(
        self, 
        track_id: int, 
        box: Tuple[int, int, int, int], 
        frame_number: int, 
        class_name: str, 
        confidence: float
    ) -> None:
        """
        Update vehicle track with new detection.
        
        Args:
            track_id: Track identifier
            box: Detection bounding box
            frame_number: Current frame number
            class_name: Detection class name
            confidence: Detection confidence score
        """
        self.vehicle_tracks[track_id].append({
            'box': box,
            'frame': frame_number,
            'class': class_name,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def calculate_severity_index(self, track_id: int) -> Tuple[float, str]:
        """
        Calculate severity index based on velocity drop and IoU consistency.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Tuple of (severity_index, severity_category)
        """
        track_history = list(self.vehicle_tracks[track_id])
        
        if len(track_history) < 3:
            return 0.0, "Insufficient Data"
        
        # Calculate velocity over last few frames
        velocities = []
        for i in range(len(track_history) - 1, max(0, len(track_history) - 4), -1):
            if i > 0:
                vel = self.calculate_velocity(track_history[i-1:i+1])
                velocities.append(vel)
        
        if len(velocities) < 2:
            return 0.0, "Insufficient Data"
        
        # Check for sudden velocity drop (indicating crash)
        avg_velocity = np.mean(velocities[1:]) if len(velocities) > 1 else velocities[0]
        current_velocity = velocities[0]
        velocity_drop_ratio = (avg_velocity - current_velocity) / (avg_velocity + 1e-6)
        
        # Calculate IoU consistency (vehicle staying in same location)
        recent_boxes = [d['box'] for d in track_history[-3:]]
        iou_values = []
        for i in range(len(recent_boxes) - 1):
            iou = self.calculate_iou(recent_boxes[i], recent_boxes[i+1])
            iou_values.append(iou)
        
        avg_iou = np.mean(iou_values) if iou_values else 0.0
        
        # Severity Index: High IoU + Sudden velocity drop = Severe crash
        # BUT require minimum prior velocity (object must have been moving first)
        severity_index = 0.0
        severity_category = "Monitoring"
        
        # Minimum velocity required before considering it a "crash" (object must have been moving)
        min_prior_velocity = 5.0  # pixels per frame
        
        # Check if object was ever moving (to distinguish from always-stationary objects)
        was_moving = avg_velocity > min_prior_velocity
        is_now_slow = current_velocity < min_prior_velocity
        
        if was_moving and is_now_slow and avg_iou > self.iou_threshold and velocity_drop_ratio > 0.7:
            severity_index = min(1.0, avg_iou * velocity_drop_ratio)
            severity_category = "Severe"
        elif was_moving and avg_iou > 0.2 and velocity_drop_ratio > 0.5:
            severity_index = 0.5
            severity_category = "Moderate"
        elif was_moving and velocity_drop_ratio > 0.3:
            severity_index = 0.3
            severity_category = "Mild"
        
        return float(severity_index), severity_category
    
    def analyze_accident(
        self, 
        detections: List[Tuple[Tuple[int, int, int, int], float, str]], 
        frame_number: int
    ) -> List[SeverityResult]:
        """
        Analyze detections for accident severity.
        
        Args:
            detections: List of (box, confidence, class_name) tuples
            frame_number: Current frame number
            
        Returns:
            List of SeverityResult objects for accident detections
        """
        severity_results = []
        
        for box, conf, cls_name in detections:
            track_id = self.assign_track_id(box, frame_number)
            self.update_track(track_id, box, frame_number, cls_name, conf)
            
            # Calculate severity ONLY if class indicates an accident/crash
            # Filter: must contain 'accident', 'crash', 'collision' or be severity labels
            cls_lower = cls_name.lower()
            is_crash_class = any(keyword in cls_lower for keyword in [
                'accident', 'crash', 'collision', 'severe', 'moderate', 'mild', 'wreck', 'impact'
            ])
            
            # Skip non-crash detections (like 'person', 'car', 'truck' from COCO)
            if not is_crash_class:
                continue
            
            severity_index, severity_category = self.calculate_severity_index(track_id)
            severity_results.append(SeverityResult(
                track_id=track_id,
                severity_index=severity_index,
                severity_category=severity_category,
                class_name=cls_name,
                confidence=conf,
                box=box
            ))
        
        return severity_results
    
    def reset(self) -> None:
        """Reset all tracks and state."""
        self.vehicle_tracks.clear()
        self.next_track_id = 0
        self.track_assignments.clear()
