"""
Unit tests for anonymization functions.

Tests cover:
- Face blurring
- License plate detection
- Privacy metrics
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.anonymization import (
    anonymize_frame,
    _blur_faces,
    _blur_license_plates,
    get_privacy_metrics,
    set_face_model
)


class TestAnonymizeFrame:
    """Tests for frame anonymization."""
    
    def test_no_model_returns_original(self, sample_frame):
        """Test that without face model, original frame is returned."""
        set_face_model(None)
        result = anonymize_frame(sample_frame)
        assert np.array_equal(result, sample_frame)
    
    def test_returns_same_shape(self, sample_frame):
        """Test that output has same shape as input."""
        set_face_model(None)
        result = anonymize_frame(sample_frame)
        assert result.shape == sample_frame.shape
    
    def test_frame_is_copied(self, sample_frame):
        """Test that original frame is not modified."""
        original = sample_frame.copy()
        set_face_model(None)
        _ = anonymize_frame(sample_frame)
        assert np.array_equal(sample_frame, original)


class TestBlurLicensePlates:
    """Tests for license plate blurring."""
    
    def test_white_region_detection(self):
        """Test detection of white rectangular regions."""
        # Create frame with white rectangle (simulating plate)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add white rectangle with plate-like aspect ratio
        frame[200:240, 300:420] = [255, 255, 255]  # 40x120 = 3:1 ratio
        
        result = _blur_license_plates(frame.copy(), frame)
        
        # The white region should have been modified (blurred)
        # Check that the region is no longer uniform white
        roi = result[200:240, 300:420]
        # Due to blur, the pixel values should vary
        assert roi.std() < frame[200:240, 300:420].std() or not np.array_equal(
            result[200:240, 300:420], 
            frame[200:240, 300:420]
        )
    
    def test_non_plate_regions_preserved(self):
        """Test that non-plate regions are preserved."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a square white region (wrong aspect ratio for plate)
        frame[100:200, 100:200] = [255, 255, 255]
        
        result = _blur_license_plates(frame.copy(), frame)
        
        # Square should not be modified (aspect ratio 1:1)
        assert np.array_equal(result[100:200, 100:200], frame[100:200, 100:200])


class TestPrivacyMetrics:
    """Tests for privacy metrics function."""
    
    def test_metrics_without_model(self, sample_frame):
        """Test metrics when no face model loaded."""
        set_face_model(None)
        metrics = get_privacy_metrics(sample_frame)
        
        assert 'faces_detected' in metrics
        assert 'plates_detected' in metrics
        assert 'anonymization_enabled' in metrics
        assert metrics['anonymization_enabled'] is False
    
    def test_metrics_structure(self, sample_frame):
        """Test that metrics have correct structure."""
        set_face_model(None)
        metrics = get_privacy_metrics(sample_frame)
        
        assert isinstance(metrics['faces_detected'], int)
        assert isinstance(metrics['plates_detected'], int)
        assert isinstance(metrics['anonymization_enabled'], bool)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_frame(self):
        """Test with empty frame."""
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        set_face_model(None)
        # Should not raise error
        result = anonymize_frame(frame)
        assert result.shape == frame.shape
    
    def test_single_pixel(self):
        """Test with single pixel frame."""
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        set_face_model(None)
        result = anonymize_frame(frame)
        assert result.shape == (1, 1, 3)
    
    def test_grayscale_handling(self):
        """Test that grayscale frames work."""
        # Create grayscale-like frame (but still 3 channel for BGR)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        set_face_model(None)
        result = anonymize_frame(frame)
        assert result.shape == frame.shape
