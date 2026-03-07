"""
Pytest configuration and fixtures for Smart Surveillance System tests.

Provides shared fixtures for:
- Sample data (bounding boxes, detections, frames)
- Service instances (SeverityTriageSystem, DetectionService)
- Mock settings and database
- Performance timing helpers
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Sample Data Fixtures ──────────────────────────────────────────────

@pytest.fixture
def sample_bounding_box():
    """Sample bounding box for testing."""
    return (100, 100, 200, 200)


@pytest.fixture
def sample_detection():
    """Sample detection tuple (box, confidence, class_name)."""
    return ((100, 100, 200, 200), 0.85, "Accident")


@pytest.fixture
def sample_detections():
    """List of sample detections with mixed classes."""
    return [
        ((100, 100, 200, 200), 0.9, "Accident"),
        ((300, 100, 400, 200), 0.75, "car"),
        ((500, 300, 600, 400), 0.8, "severe"),
    ]


@pytest.fixture
def sample_frame():
    """Sample 480x640 BGR frame (numpy array)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_with_content():
    """Sample frame with some non-zero pixel data for realistic testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def crash_frame_sequence():
    """Sequence of frames simulating a crash scene (5 frames).
    
    Returns list of (frame, detections) tuples where detections
    simulate a vehicle moving then stopping (crash).
    """
    frames = []
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Vehicle moves right, then stops at frame 3 (crash)
        if i < 3:
            x_offset = i * 50
            box = (100 + x_offset, 200, 200 + x_offset, 300)
        else:
            box = (200, 200, 300, 300)  # Stopped
        detections = [(box, 0.85, "Accident")]
        frames.append((frame, detections))
    return frames


# ── Service Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def severity_triage_system():
    """SeverityTriageSystem instance for testing."""
    from services.severity_triage import SeverityTriageSystem
    return SeverityTriageSystem(buffer_size=10, iou_threshold=0.3)


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock environment settings for testing."""
    monkeypatch.setenv("BOT_TOKEN", "test_token_123")
    monkeypatch.setenv("CHAT_ID", "test_chat_456")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./test_db.db")
    monkeypatch.setenv("ALERT_SEVERITY_LEVELS", "Severe,Moderate")


@pytest.fixture
def mock_detection_service(mock_settings):
    """DetectionService with mocked YOLO models.
    
    Models are replaced with MagicMocks that return empty results,
    allowing testing of the pipeline without actual model files.
    """
    from services.detection import DetectionService
    
    service = DetectionService()
    
    # Mock crash model — returns no detections by default
    mock_crash_model = MagicMock()
    mock_crash_result = MagicMock()
    mock_crash_result.boxes = MagicMock()
    mock_crash_result.boxes.__len__ = lambda self: 0
    mock_crash_result.boxes.__iter__ = lambda self: iter([])
    mock_crash_model.return_value = [mock_crash_result]
    service.crash_model = mock_crash_model
    
    # Mock object model — returns no detections by default
    mock_obj_model = MagicMock()
    mock_obj_result = MagicMock()
    mock_obj_result.boxes = MagicMock()
    mock_obj_result.boxes.__len__ = lambda self: 0
    mock_obj_result.boxes.__iter__ = lambda self: iter([])
    mock_obj_model.return_value = [mock_obj_result]
    mock_obj_model.names = {0: "person", 1: "car", 2: "truck"}
    service.object_model = mock_obj_model
    
    return service


@pytest.fixture
def mock_telegram_service():
    """Mocked TelegramAlertService that tracks calls without HTTP."""
    mock = MagicMock()
    mock.enabled = True
    mock.send_alert = MagicMock(return_value=True)
    return mock


# ── Environment Fixtures ──────────────────────────────────────────────

@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database path."""
    db_path = tmp_path / "test_crashes.db"
    return f"sqlite:///{db_path}"


# ── Performance Helpers ───────────────────────────────────────────────

@pytest.fixture
def timer():
    """Simple timer context manager for performance tests.
    
    Usage:
        with timer() as t:
            do_work()
        assert t.elapsed < 1.0
    """
    class Timer:
        def __init__(self):
            self.start = None
            self.elapsed = None
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
    
    return Timer
