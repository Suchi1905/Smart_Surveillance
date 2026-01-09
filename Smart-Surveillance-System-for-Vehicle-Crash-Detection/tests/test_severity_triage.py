"""
Unit tests for SeverityTriageSystem.

Tests cover:
- IoU calculation
- Velocity estimation
- Track ID assignment
- Severity index calculation
- Accident analysis
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.severity_triage import SeverityTriageSystem, SeverityResult


class TestIoUCalculation:
    """Tests for IoU calculation method."""
    
    def test_perfect_overlap(self, severity_triage_system):
        """Test IoU for perfectly overlapping boxes."""
        box = (100, 100, 200, 200)
        iou = severity_triage_system.calculate_iou(box, box)
        assert iou == 1.0
    
    def test_no_overlap(self, severity_triage_system):
        """Test IoU for non-overlapping boxes."""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        iou = severity_triage_system.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_partial_overlap(self, severity_triage_system):
        """Test IoU for partially overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = severity_triage_system.calculate_iou(box1, box2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.001
    
    def test_contained_box(self, severity_triage_system):
        """Test IoU when one box contains another."""
        box1 = (0, 0, 200, 200)
        box2 = (50, 50, 150, 150)
        iou = severity_triage_system.calculate_iou(box1, box2)
        # Intersection: 100x100 = 10000
        # Union: 40000 + 10000 - 10000 = 40000
        expected_iou = 10000 / 40000
        assert abs(iou - expected_iou) < 0.001


class TestBoxCenter:
    """Tests for bounding box center calculation."""
    
    def test_simple_box(self, severity_triage_system):
        """Test center calculation for simple box."""
        box = (0, 0, 100, 100)
        center = severity_triage_system.get_box_center(box)
        assert center == (50.0, 50.0)
    
    def test_offset_box(self, severity_triage_system):
        """Test center calculation for offset box."""
        box = (100, 200, 200, 400)
        center = severity_triage_system.get_box_center(box)
        assert center == (150.0, 300.0)


class TestVelocityCalculation:
    """Tests for velocity calculation."""
    
    def test_no_history(self, severity_triage_system):
        """Test velocity with empty history."""
        velocity = severity_triage_system.calculate_velocity([])
        assert velocity == 0.0
    
    def test_single_frame(self, severity_triage_system):
        """Test velocity with single frame."""
        history = [{'box': (0, 0, 100, 100), 'frame': 1}]
        velocity = severity_triage_system.calculate_velocity(history)
        assert velocity == 0.0
    
    def test_stationary_object(self, severity_triage_system):
        """Test velocity for stationary object."""
        history = [
            {'box': (100, 100, 200, 200), 'frame': 1},
            {'box': (100, 100, 200, 200), 'frame': 2},
        ]
        velocity = severity_triage_system.calculate_velocity(history)
        assert velocity == 0.0
    
    def test_moving_object(self, severity_triage_system):
        """Test velocity for moving object."""
        history = [
            {'box': (0, 0, 100, 100), 'frame': 1},
            {'box': (100, 0, 200, 100), 'frame': 2},
        ]
        velocity = severity_triage_system.calculate_velocity(history)
        # Centers: (50, 50) -> (150, 50), displacement = 100
        assert abs(velocity - 100.0) < 0.001


class TestTrackAssignment:
    """Tests for track ID assignment."""
    
    def test_new_track(self, severity_triage_system):
        """Test assignment of new track ID."""
        box = (100, 100, 200, 200)
        track_id = severity_triage_system.assign_track_id(box, frame_number=1)
        assert track_id == 0
    
    def test_second_track(self, severity_triage_system):
        """Test assignment of second new track."""
        box1 = (100, 100, 200, 200)
        box2 = (500, 500, 600, 600)  # Far away, different track
        
        track_id1 = severity_triage_system.assign_track_id(box1, frame_number=1)
        severity_triage_system.update_track(track_id1, box1, 1, "car", 0.9)
        
        track_id2 = severity_triage_system.assign_track_id(box2, frame_number=2)
        
        assert track_id1 == 0
        assert track_id2 == 1
    
    def test_matching_existing_track(self, severity_triage_system):
        """Test matching to existing track with high IoU."""
        box1 = (100, 100, 200, 200)
        box2 = (105, 105, 205, 205)  # Slightly moved, high IoU
        
        track_id1 = severity_triage_system.assign_track_id(box1, frame_number=1)
        severity_triage_system.update_track(track_id1, box1, 1, "car", 0.9)
        
        track_id2 = severity_triage_system.assign_track_id(box2, frame_number=2)
        
        # Should match existing track
        assert track_id2 == track_id1


class TestSeverityIndex:
    """Tests for severity index calculation."""
    
    def test_insufficient_data(self, severity_triage_system):
        """Test severity with insufficient history."""
        track_id = 0
        severity_triage_system.update_track(track_id, (100, 100, 200, 200), 1, "Accident", 0.9)
        
        severity_index, category = severity_triage_system.calculate_severity_index(track_id)
        assert category == "Insufficient Data"
        assert severity_index == 0.0
    
    def test_monitoring_category(self, severity_triage_system):
        """Test monitoring category for normal movement."""
        track_id = 0
        # Simulate normal movement (different positions)
        for i in range(5):
            box = (i * 100, 100, i * 100 + 100, 200)
            severity_triage_system.update_track(track_id, box, i, "car", 0.9)
        
        severity_index, category = severity_triage_system.calculate_severity_index(track_id)
        # Should be monitoring since no velocity drop
        assert category in ["Monitoring", "Insufficient Data", "Mild"]


class TestAccidentAnalysis:
    """Tests for accident analysis."""
    
    def test_empty_detections(self, severity_triage_system):
        """Test analysis with no detections."""
        results = severity_triage_system.analyze_accident([], frame_number=1)
        assert results == []
    
    def test_non_accident_detection(self, severity_triage_system):
        """Test analysis with non-accident detection."""
        detections = [((100, 100, 200, 200), 0.9, "car")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        # No severity results for non-accident classes
        assert results == []
    
    def test_accident_detection(self, severity_triage_system):
        """Test analysis with accident detection."""
        detections = [((100, 100, 200, 200), 0.9, "Accident")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        
        assert len(results) == 1
        assert isinstance(results[0], SeverityResult)
        assert results[0].class_name == "Accident"
        assert results[0].confidence == 0.9
    
    def test_multiple_detections(self, severity_triage_system, sample_detections):
        """Test analysis with multiple detections."""
        results = severity_triage_system.analyze_accident(sample_detections, frame_number=1)
        
        # Should have results for "Accident" and "severe"
        assert len(results) == 2
        class_names = [r.class_name for r in results]
        assert "Accident" in class_names
        assert "severe" in class_names


class TestReset:
    """Tests for system reset."""
    
    def test_reset_clears_tracks(self, severity_triage_system):
        """Test that reset clears all tracks."""
        detections = [((100, 100, 200, 200), 0.9, "Accident")]
        severity_triage_system.analyze_accident(detections, frame_number=1)
        
        assert len(severity_triage_system.vehicle_tracks) > 0
        
        severity_triage_system.reset()
        
        assert len(severity_triage_system.vehicle_tracks) == 0
        assert severity_triage_system.next_track_id == 0
