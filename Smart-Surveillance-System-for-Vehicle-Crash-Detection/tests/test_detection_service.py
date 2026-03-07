"""
Detection Service unit tests.

Tests core DetectionService methods in isolation:
- Model loading, IoU calculation, cooldown logic,
  frame encoding, error frames, and severity config.
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.unit
class TestDetectionServiceInit:
    """Test DetectionService initialization."""

    def test_service_instantiation(self, mock_settings):
        """Test service can be created with settings."""
        from services.detection import DetectionService
        service = DetectionService()
        assert service is not None

    def test_initial_state(self, mock_detection_service):
        """Test initial state of service."""
        service = mock_detection_service
        assert service.frame_counter == 0
        assert service._last_alert_time == 0.0
        assert service._alert_callback is None


@pytest.mark.unit
class TestIoUCalculation:
    """Test the internal IoU calculation in DetectionService."""

    def test_perfect_overlap(self, mock_detection_service):
        """Test IoU = 1.0 for identical boxes."""
        box = (100, 100, 200, 200)
        iou = mock_detection_service._calculate_iou(box, box)
        assert iou == pytest.approx(1.0, abs=0.001)

    def test_no_overlap(self, mock_detection_service):
        """Test IoU = 0.0 for non-overlapping boxes."""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        iou = mock_detection_service._calculate_iou(box1, box2)
        assert iou == pytest.approx(0.0, abs=0.001)

    def test_partial_overlap(self, mock_detection_service):
        """Test IoU for partially overlapping boxes."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = mock_detection_service._calculate_iou(box1, box2)
        expected = 2500 / 17500  # 50x50 intersection / union
        assert iou == pytest.approx(expected, abs=0.01)

    def test_contained_box(self, mock_detection_service):
        """Test IoU when one box contains the other."""
        box1 = (0, 0, 200, 200)
        box2 = (50, 50, 150, 150)
        iou = mock_detection_service._calculate_iou(box1, box2)
        expected = 10000 / 40000
        assert iou == pytest.approx(expected, abs=0.01)


@pytest.mark.unit
class TestAlertCooldown:
    """Test alert cooldown timing logic."""

    def test_can_alert_initially(self, mock_detection_service):
        """Test alert is allowed on first call (last_alert_time=0)."""
        service = mock_detection_service
        assert service._can_send_alert() is True

    def test_cooldown_blocks_alert(self, mock_detection_service):
        """Test alert is blocked during cooldown."""
        service = mock_detection_service
        service._last_alert_time = time.time()
        assert service._can_send_alert() is False

    def test_cooldown_expires(self, mock_detection_service):
        """Test alert is allowed after cooldown expires."""
        service = mock_detection_service
        service._last_alert_time = time.time() - service._alert_cooldown - 1
        assert service._can_send_alert() is True

    def test_cooldown_boundary(self, mock_detection_service):
        """Test alert exactly at cooldown edge."""
        service = mock_detection_service
        service._last_alert_time = time.time() - service._alert_cooldown
        # At exactly the cooldown boundary, should pass (>=)
        assert service._can_send_alert() is True


@pytest.mark.unit
class TestFrameEncoding:
    """Test MJPEG frame encoding."""

    def test_encode_frame_produces_bytes(self, mock_detection_service, sample_frame):
        """Test that encode_frame returns bytes."""
        encoded = mock_detection_service._encode_frame(sample_frame)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_frame_is_jpeg(self, mock_detection_service, sample_frame):
        """Test encoded frame starts with JPEG magic bytes."""
        encoded = mock_detection_service._encode_frame(sample_frame)
        # JPEG files start with 0xFF 0xD8
        # The encoded output includes the MJPEG boundary header
        assert b"--frame" in encoded or b"\xff\xd8" in encoded

    def test_encode_colored_frame(self, mock_detection_service, sample_frame_with_content):
        """Test encoding a frame with actual pixel content."""
        encoded = mock_detection_service._encode_frame(sample_frame_with_content)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 100  # Colored frame should produce more bytes


@pytest.mark.unit
class TestErrorFrame:
    """Test error frame creation."""

    def test_create_error_frame_shape(self, mock_detection_service):
        """Test error frame has correct dimensions."""
        frame = mock_detection_service._create_error_frame("Test Error")
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # BGR

    def test_error_frame_not_all_zeros(self, mock_detection_service):
        """Test error frame has some content (text drawn on it)."""
        frame = mock_detection_service._create_error_frame("Error: No Camera")
        # Frame should have some non-zero pixels from the text
        assert np.any(frame > 0)


@pytest.mark.unit
class TestSeverityLevelsConfig:
    """Test alert severity levels configuration."""

    def test_default_severity_levels(self, mock_detection_service):
        """Test default severity levels include Severe and Moderate."""
        levels = mock_detection_service.settings.alert_severity_levels
        assert "Severe" in levels
        assert "Moderate" in levels

    def test_monitoring_not_in_levels(self, mock_detection_service):
        """Test Monitoring is not in alert severity levels."""
        levels = mock_detection_service.settings.alert_severity_levels
        assert "Monitoring" not in levels

    def test_insufficient_data_not_in_levels(self, mock_detection_service):
        """Test 'Insufficient Data' is not in alert severity levels."""
        levels = mock_detection_service.settings.alert_severity_levels
        assert "Insufficient Data" not in levels
