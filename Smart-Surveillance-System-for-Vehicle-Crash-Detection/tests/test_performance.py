"""
Performance benchmark tests.

Validates that critical operations meet latency requirements:
- Frame processing inference speed
- Severity triage throughput
- MJPEG encoding speed
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.slow
class TestInferenceSpeed:
    """Benchmark model inference speed."""

    def test_frame_processing_latency(self, mock_detection_service, sample_frame, timer):
        """Test single frame processing completes within 500ms.
        
        Uses mocked models, so this measures pipeline overhead
        (pre/post-processing, triage, drawing) without model inference.
        """
        service = mock_detection_service

        with timer() as t:
            service._process_frame(sample_frame, 0.6)

        assert t.elapsed < 0.5, f"Frame processing took {t.elapsed:.3f}s (limit: 0.5s)"

    def test_batch_frame_processing(self, mock_detection_service, sample_frame, timer):
        """Test processing 30 frames (1 second of video at 30fps)."""
        service = mock_detection_service

        with timer() as t:
            for _ in range(30):
                service._process_frame(sample_frame, 0.6)

        fps = 30 / t.elapsed
        assert fps > 10, f"Pipeline FPS: {fps:.1f} (minimum: 10)"


class TestTriageThroughput:
    """Benchmark severity triage system throughput."""

    def test_triage_1000_calls(self, severity_triage_system, timer):
        """Test 1000 triage analysis calls complete within 1 second."""
        detections = [((100, 100, 200, 200), 0.9, "Accident")]

        with timer() as t:
            for i in range(1000):
                severity_triage_system.analyze_accident(detections, frame_number=i)

        assert t.elapsed < 1.0, f"1000 triage calls took {t.elapsed:.3f}s (limit: 1.0s)"

    def test_triage_with_multiple_detections(self, severity_triage_system, timer):
        """Test triage throughput with 5 detections per frame."""
        detections = [
            ((100 + i * 100, 100, 200 + i * 100, 200), 0.8, "Accident")
            for i in range(5)
        ]

        with timer() as t:
            for i in range(200):
                severity_triage_system.analyze_accident(detections, frame_number=i)

        calls = 200
        throughput = calls / t.elapsed
        assert throughput > 100, f"Triage throughput: {throughput:.0f}/s (minimum: 100/s)"

    def test_iou_calculation_speed(self, severity_triage_system, timer):
        """Test IoU calculation speed (10000 calls)."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)

        with timer() as t:
            for _ in range(10000):
                severity_triage_system.calculate_iou(box1, box2)

        assert t.elapsed < 0.5, f"10000 IoU calls took {t.elapsed:.3f}s (limit: 0.5s)"


class TestEncodingSpeed:
    """Benchmark frame encoding performance."""

    def test_single_frame_encoding(self, mock_detection_service, sample_frame, timer):
        """Test single MJPEG encoding takes <10ms."""
        service = mock_detection_service

        with timer() as t:
            service._encode_frame(sample_frame)

        assert t.elapsed < 0.01, f"Encoding took {t.elapsed * 1000:.1f}ms (limit: 10ms)"

    def test_batch_encoding_30fps(self, mock_detection_service, sample_frame_with_content, timer):
        """Test encoding 30 frames (1s of video) under 300ms."""
        service = mock_detection_service

        with timer() as t:
            for _ in range(30):
                service._encode_frame(sample_frame_with_content)

        assert t.elapsed < 0.3, f"30 frames encoding took {t.elapsed:.3f}s (limit: 0.3s)"

    def test_hd_frame_encoding(self, mock_detection_service, timer):
        """Test encoding 1080p frame within acceptable time."""
        hd_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        with timer() as t:
            mock_detection_service._encode_frame(hd_frame)

        assert t.elapsed < 0.1, f"HD encoding took {t.elapsed * 1000:.1f}ms (limit: 100ms)"


class TestSeverityCalculationSpeed:
    """Benchmark severity index calculation."""

    def test_severity_calculation_with_history(self, severity_triage_system, timer):
        """Test severity index calculation speed with full track history."""
        track_id = 0
        # Build up 10 frames of history
        for i in range(10):
            box = (100 + i * 10, 100, 200 + i * 10, 200)
            severity_triage_system.update_track(track_id, box, i, "Accident", 0.9)

        with timer() as t:
            for _ in range(1000):
                severity_triage_system.calculate_severity_index(track_id)

        assert t.elapsed < 0.5, f"1000 severity calcs took {t.elapsed:.3f}s (limit: 0.5s)"
