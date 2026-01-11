"""
Multi-Object Tracking (MOT) service using ByteTrack algorithm.

This module provides robust vehicle tracking across frames using IoU-based
track association with byte-level confidence handling.

Based on ByteTrack: Multi-Object Tracking by Associating Every Detection Box
https://arxiv.org/abs/2110.06864
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Enumeration of track states."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    """
    Represents a tracked object across frames.
    
    Attributes:
        track_id: Unique identifier for this track
        bbox: Current bounding box (x1, y1, x2, y2)
        score: Detection confidence score
        class_id: Object class identifier
        class_name: Human-readable class name
        state: Current track state
        history: List of historical positions for trajectory
        velocity: Estimated velocity (vx, vy) in pixels/frame
        age: Number of frames since track creation
        hits: Number of successful associations
        time_since_update: Frames since last detection match
        features: Optional appearance features for Re-ID
    """
    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int = 0
    class_name: str = "vehicle"
    state: int = TrackState.NEW
    history: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize history with current center position."""
        if not self.history:
            center = self.get_center()
            self.history.append(center)
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))
    
    def update(self, bbox: np.ndarray, score: float):
        """Update track with new detection."""
        old_center = self.get_center()
        self.bbox = bbox
        self.score = score
        new_center = self.get_center()
        
        # Calculate velocity
        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        
        # Update history (keep last 30 positions for trajectory)
        self.history.append(new_center)
        if len(self.history) > 30:
            self.history.pop(0)
        
        self.state = TrackState.TRACKED
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
    
    def predict(self) -> np.ndarray:
        """Predict next position using constant velocity model."""
        predicted_bbox = self.bbox.copy()
        vx, vy = self.velocity
        predicted_bbox[0] += vx
        predicted_bbox[1] += vy
        predicted_bbox[2] += vx
        predicted_bbox[3] += vy
        return predicted_bbox
    
    def get_speed_pixels(self) -> float:
        """Get speed in pixels per frame."""
        vx, vy = self.velocity
        return np.sqrt(vx**2 + vy**2)
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get full trajectory history."""
        return self.history.copy()
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.LOST
        self.time_since_update += 1
        self.age += 1
    
    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.REMOVED


class ByteTracker:
    """
    ByteTrack-style multi-object tracker.
    
    Uses two-stage association:
    1. Match high-confidence detections with existing tracks
    2. Match remaining low-confidence detections with unmatched tracks
    
    This approach recovers occluded objects that may have lower confidence.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,
        match_thresh: float = 0.8,
        max_time_lost: int = 30,
        min_hits: int = 3
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Minimum confidence to create new track
            high_thresh: Threshold for high-confidence detections
            match_thresh: IoU threshold for matching
            max_time_lost: Max frames before removing lost track
            min_hits: Min hits before track is confirmed
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        
        self.tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.removed_tracks: Dict[int, Track] = {}
        
        self._next_id = 0
        self.frame_count = 0
        
        logger.info(f"ByteTracker initialized with thresh={track_thresh}, max_lost={max_time_lost}")
    
    def _get_next_id(self) -> int:
        """Get next unique track ID."""
        track_id = self._next_id
        self._next_id += 1
        return track_id
    
    def update(
        self, 
        detections: List[Tuple[np.ndarray, float, int, str]]
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (bbox, score, class_id, class_name) tuples
                       where bbox is [x1, y1, x2, y2]
        
        Returns:
            List of active Track objects
        """
        self.frame_count += 1
        
        if not detections:
            # No detections - mark all tracks as lost
            for track in list(self.tracks.values()):
                track.mark_lost()
                if track.time_since_update > self.max_time_lost:
                    self._remove_track(track)
                else:
                    self.lost_tracks[track.track_id] = track
                    del self.tracks[track.track_id]
            return list(self.tracks.values())
        
        # Separate high and low confidence detections
        high_dets = []
        low_dets = []
        
        for det in detections:
            bbox, score, class_id, class_name = det
            if score >= self.high_thresh:
                high_dets.append(det)
            elif score >= self.track_thresh:
                low_dets.append(det)
        
        # Step 1: Match high-confidence detections with tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            list(self.tracks.values()), high_dets, self.match_thresh
        )
        
        # Update matched tracks
        for track, det in matched_tracks:
            bbox, score, class_id, class_name = det
            track.update(np.array(bbox), score)
        
        # Step 2: Match low-confidence detections with remaining tracks
        remaining_tracks = [self.tracks[tid] for tid in unmatched_tracks]
        matched_low, still_unmatched, unmatched_low = self._associate(
            remaining_tracks, low_dets, self.match_thresh
        )
        
        # Update tracks matched with low-confidence detections
        for track, det in matched_low:
            bbox, score, class_id, class_name = det
            track.update(np.array(bbox), score)
        
        # Step 3: Try to recover lost tracks with unmatched high detections
        recovered = []
        for det in unmatched_dets:
            bbox, score, class_id, class_name = det
            best_iou = 0
            best_lost = None
            
            for lost_track in self.lost_tracks.values():
                iou = self._iou(np.array(bbox), lost_track.predict())
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_lost = lost_track
            
            if best_lost:
                best_lost.update(np.array(bbox), score)
                best_lost.state = TrackState.TRACKED
                self.tracks[best_lost.track_id] = best_lost
                del self.lost_tracks[best_lost.track_id]
                recovered.append(best_lost.track_id)
            else:
                unmatched_dets.remove(det)
                # Create new track
                new_track = Track(
                    track_id=self._get_next_id(),
                    bbox=np.array(bbox),
                    score=score,
                    class_id=class_id,
                    class_name=class_name,
                    state=TrackState.NEW
                )
                self.tracks[new_track.track_id] = new_track
        
        # Step 4: Handle unmatched tracks (mark as lost)
        for tid in still_unmatched:
            track = self.tracks.get(tid)
            if track:
                track.mark_lost()
                if track.time_since_update > self.max_time_lost:
                    self._remove_track(track)
                else:
                    self.lost_tracks[track.track_id] = track
                    del self.tracks[track.track_id]
        
        # Step 5: Create new tracks for unmatched high detections
        for det in unmatched_dets:
            bbox, score, class_id, class_name = det
            new_track = Track(
                track_id=self._get_next_id(),
                bbox=np.array(bbox),
                score=score,
                class_id=class_id,
                class_name=class_name,
                state=TrackState.NEW
            )
            self.tracks[new_track.track_id] = new_track
        
        # Clean up old lost tracks
        for tid in list(self.lost_tracks.keys()):
            self.lost_tracks[tid].time_since_update += 1
            if self.lost_tracks[tid].time_since_update > self.max_time_lost:
                self._remove_track(self.lost_tracks[tid])
                del self.lost_tracks[tid]
        
        # Return confirmed tracks only
        return [t for t in self.tracks.values() 
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits]
    
    def _associate(
        self, 
        tracks: List[Track], 
        detections: List[Tuple],
        thresh: float
    ) -> Tuple[List[Tuple[Track, Tuple]], List[int], List[Tuple]]:
        """
        Associate detections with tracks using IoU.
        
        Returns:
            matched: List of (track, detection) pairs
            unmatched_tracks: List of unmatched track IDs
            unmatched_dets: List of unmatched detections
        """
        if not tracks or not detections:
            return [], [t.track_id for t in tracks], detections
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                bbox = np.array(det[0])
                iou_matrix[i, j] = self._iou(track.bbox, bbox)
        
        # Hungarian algorithm (simplified greedy matching)
        matched = []
        matched_track_ids = set()
        matched_det_indices = set()
        
        # Sort by IoU (highest first)
        sorted_pairs = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] >= thresh:
                    sorted_pairs.append((iou_matrix[i, j], i, j))
        
        sorted_pairs.sort(reverse=True)
        
        for iou_val, i, j in sorted_pairs:
            if i not in matched_track_ids and j not in matched_det_indices:
                matched.append((tracks[i], detections[j]))
                matched_track_ids.add(i)
                matched_det_indices.add(j)
        
        unmatched_tracks = [tracks[i].track_id for i in range(len(tracks)) 
                           if i not in matched_track_ids]
        unmatched_dets = [detections[j] for j in range(len(detections)) 
                         if j not in matched_det_indices]
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _remove_track(self, track: Track):
        """Remove track from active tracking."""
        track.mark_removed()
        self.removed_tracks[track.track_id] = track
        # Keep limited history
        if len(self.removed_tracks) > 100:
            oldest = min(self.removed_tracks.keys())
            del self.removed_tracks[oldest]
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id) or self.lost_tracks.get(track_id)
    
    def get_all_tracks(self) -> List[Track]:
        """Get all active and lost tracks."""
        return list(self.tracks.values()) + list(self.lost_tracks.values())
    
    def get_active_count(self) -> int:
        """Get count of currently active tracks."""
        return len(self.tracks)
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self._next_id = 0
        self.frame_count = 0
        logger.info("ByteTracker reset")
