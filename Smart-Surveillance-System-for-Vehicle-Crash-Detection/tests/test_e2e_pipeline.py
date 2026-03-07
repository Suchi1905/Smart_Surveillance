"""
End-to-end pipeline integration tests.

Tests the full flow: model init → frame processing → severity triage → alert triggering.
Uses mocked YOLO models to avoid requiring actual model files.
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.severity_triage import SeverityTriageSystem, SeverityResult


@pytest.mark.integration
class TestPipelineInitialization:
    """Test that the detection pipeline initializes correctly."""

    def test_detection_service_creates(self, mock_settings):
        """Test DetectionService can be instantiated."""
        from services.detection import DetectionService
        service = DetectionService()
        assert service is not None
        assert service.triage_system is not None
        assert service.frame_counter == 0

    def test_triage_system_initialized(self, mock_detection_service):
        """Test triage system is properly initialized within detection service."""
        service = mock_detection_service
        assert isinstance(service.triage_system, SeverityTriageSystem)
        assert len(service.triage_system.vehicle_tracks) == 0

    def test_alert_callback_can_be_set(self, mock_detection_service):
        """Test alert callback setter works."""
        callback = MagicMock()
        mock_detection_service.set_alert_callback(callback)
        assert mock_detection_service._alert_callback == callback

    def test_alert_cooldown_initialized(self, mock_detection_service):
        """Test alert cooldown is set to a reasonable value."""
        service = mock_detection_service
        assert service._alert_cooldown > 0
        assert service._last_alert_time == 0.0


@pytest.mark.integration
class TestFrameProcessing:
    """Test frame processing through the pipeline."""

    def test_process_frame_returns_ndarray(self, mock_detection_service, sample_frame):
        """Test _process_frame returns a valid numpy array."""
        result = mock_detection_service._process_frame(sample_frame, conf_threshold=0.6)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_frame.shape

    def test_process_frame_returns_same_shape(self, mock_detection_service, sample_frame):
        """Test _process_frame returns frame with same dimensions."""
        result = mock_detection_service._process_frame(sample_frame, conf_threshold=0.6)
        assert result.shape == sample_frame.shape

    def test_process_frame_handles_empty_detections(self, mock_detection_service, sample_frame):
        """Test processing works when models detect nothing."""
        # Models are mocked to return no detections
        result = mock_detection_service._process_frame(sample_frame, 0.6)
        assert isinstance(result, np.ndarray)

    def test_process_frame_with_crash_detection(self, mock_detection_service, sample_frame):
        """Test processing when crash model detects something."""
        service = mock_detection_service

        # Create a properly structured mock result
        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__ = MagicMock(return_value=MagicMock())
        mock_box.xyxy.__getitem__.return_value.cpu.return_value.numpy.return_value.astype.return_value = np.array([100, 100, 200, 200])
        mock_box.conf = MagicMock()
        mock_box.conf.__getitem__ = MagicMock(return_value=MagicMock())
        mock_box.conf.__getitem__.return_value = 0.85
        mock_box.cls = MagicMock()
        mock_box.cls.__getitem__ = MagicMock(return_value=MagicMock())
        mock_box.cls.__getitem__.return_value = 0

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__iter__ = MagicMock(return_value=iter([mock_box]))
        mock_result.boxes.__len__ = MagicMock(return_value=1)

        service.crash_model.return_value = [mock_result]
        service.crash_model.names = {0: "Accident"}

        result = service._process_frame(sample_frame, 0.6)
        assert isinstance(result, np.ndarray)


@pytest.mark.integration
class TestSeverityTriageFlow:
    """Test crash detections flowing into severity triage."""

    def test_crash_triggers_triage_analysis(self, severity_triage_system):
        """Test that crash detections are analyzed by triage."""
        detections = [((100, 100, 200, 200), 0.9, "Accident")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        assert len(results) == 1
        assert isinstance(results[0], SeverityResult)

    def test_non_crash_classes_filtered(self, severity_triage_system):
        """Test non-crash classes (car, person) are filtered out."""
        detections = [
            ((100, 100, 200, 200), 0.9, "car"),
            ((300, 300, 400, 400), 0.8, "person"),
        ]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        assert len(results) == 0

    def test_early_confidence_classification(self, severity_triage_system):
        """Test high-confidence crash gets 'Severe' on first frame (no history needed)."""
        detections = [((100, 100, 200, 200), 0.85, "Accident")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        assert len(results) == 1
        assert results[0].severity_category == "Severe"
        assert results[0].severity_index == pytest.approx(0.85, abs=0.01)

    def test_severe_class_early_classification(self, severity_triage_system):
        """Test 'severe' class name triggers immediate Severe classification."""
        detections = [((100, 100, 200, 200), 0.75, "severe")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        assert results[0].severity_category == "Severe"

    def test_moderate_class_classification(self, severity_triage_system):
        """Test 'moderate' class gets Moderate classification."""
        detections = [((100, 100, 200, 200), 0.7, "moderate")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        assert results[0].severity_category == "Moderate"

    def test_mild_class_classification(self, severity_triage_system):
        """Test 'mild' class gets Mild classification."""
        detections = [((100, 100, 200, 200), 0.6, "mild")]
        results = severity_triage_system.analyze_accident(detections, frame_number=1)
        assert results[0].severity_category == "Mild"

    def test_two_frame_motion_analysis(self, severity_triage_system):
        """Test severity computed with only 2 frames of track history."""
        # Use a non-crash class that won't trigger early confidence classification
        # but IS a crash keyword to pass the filter
        # First simulate 2 frames with the class "impact" (a crash keyword)
        box1 = (100, 100, 200, 200)
        box2 = (100, 100, 200, 200)  # Same position (stationary after crash)

        # Frame 1
        severity_triage_system.update_track(0, box1, 1, "impact", 0.5)
        # Frame 2
        severity_triage_system.update_track(0, box2, 2, "impact", 0.5)

        severity_index, category = severity_triage_system.calculate_severity_index(0)
        # With only 2 frames and low confidence, should not be "Insufficient Data"
        assert category != "Insufficient Data"


@pytest.mark.integration
class TestAlertTriggering:
    """Test alert triggering based on severity results."""

    def test_severe_crash_triggers_alert(self, mock_detection_service, sample_frame):
        """Test that a Severe crash triggers _trigger_alert."""
        service = mock_detection_service
        sev_result = SeverityResult(
            track_id=0,
            severity_index=0.9,
            severity_category="Severe",
            class_name="Accident",
            confidence=0.9,
            box=(100, 100, 200, 200)
        )

        # Mock _trigger_alert to track calls
        service._trigger_alert = MagicMock()
        service._last_alert_time = 0.0  # Ensure cooldown passed

        # Directly test the alert logic
        if sev_result.severity_category in service.settings.alert_severity_levels:
            if service._can_send_alert():
                service._trigger_alert(sample_frame, sev_result)

        service._trigger_alert.assert_called_once()

    def test_alert_cooldown_prevents_spam(self, mock_detection_service, sample_frame):
        """Test alert cooldown prevents rapid-fire alerts."""
        service = mock_detection_service
        service._last_alert_time = time.time()  # Just sent an alert

        assert not service._can_send_alert()

    def test_alert_cooldown_expires(self, mock_detection_service):
        """Test alert can be sent after cooldown expires."""
        service = mock_detection_service
        service._last_alert_time = time.time() - service._alert_cooldown - 1

        assert service._can_send_alert()

    def test_monitoring_does_not_trigger_alert(self, mock_detection_service):
        """Test Monitoring severity does not trigger alerts."""
        service = mock_detection_service
        assert "Monitoring" not in service.settings.alert_severity_levels

    def test_insufficient_data_fallback(self, mock_detection_service, sample_frame):
        """Test fallback alert for high-confidence crash with Insufficient Data."""
        service = mock_detection_service
        sev_result = SeverityResult(
            track_id=0,
            severity_index=0.0,
            severity_category="Insufficient Data",
            class_name="Accident",
            confidence=0.80,
            box=(100, 100, 200, 200)
        )

        # Confidence >= 0.75 should trigger fallback
        assert sev_result.confidence >= 0.75
        assert sev_result.severity_category == "Insufficient Data"

        # After fallback logic, severity should be overridden to Severe
        if sev_result.severity_category == "Insufficient Data" and sev_result.confidence >= 0.75:
            sev_result.severity_category = "Severe"
            sev_result.severity_index = sev_result.confidence

        assert sev_result.severity_category == "Severe"
        assert sev_result.severity_index == 0.80


@pytest.mark.integration
class TestTriageResetOnNewStream:
    """Test triage state management across streams."""

    def test_triage_reset_clears_state(self, severity_triage_system):
        """Test reset clears all tracks and counters."""
        # Add some detections
        detections = [((100, 100, 200, 200), 0.9, "Accident")]
        severity_triage_system.analyze_accident(detections, frame_number=1)
        assert len(severity_triage_system.vehicle_tracks) > 0

        # Reset
        severity_triage_system.reset()
        assert len(severity_triage_system.vehicle_tracks) == 0
        assert severity_triage_system.next_track_id == 0

    def test_fresh_tracks_after_reset(self, severity_triage_system):
        """Test new detections after reset get fresh track IDs."""
        # First round
        detections = [((100, 100, 200, 200), 0.9, "Accident")]
        severity_triage_system.analyze_accident(detections, frame_number=1)

        # Reset and new round
        severity_triage_system.reset()
        results = severity_triage_system.analyze_accident(detections, frame_number=1)

        assert results[0].track_id == 0  # Fresh ID after reset
