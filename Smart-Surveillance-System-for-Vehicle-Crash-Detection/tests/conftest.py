"""
Pytest configuration and fixtures for Smart Surveillance System tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_bounding_box():
    """Sample bounding box for testing."""
    return (100, 100, 200, 200)


@pytest.fixture
def sample_detection():
    """Sample detection tuple."""
    return ((100, 100, 200, 200), 0.85, "Accident")


@pytest.fixture
def sample_detections():
    """List of sample detections."""
    return [
        ((100, 100, 200, 200), 0.9, "Accident"),
        ((300, 100, 400, 200), 0.75, "car"),
        ((500, 300, 600, 400), 0.8, "severe"),
    ]


@pytest.fixture
def sample_frame():
    """Sample frame (numpy array) for testing."""
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def severity_triage_system():
    """SeverityTriageSystem instance for testing."""
    from services.severity_triage import SeverityTriageSystem
    return SeverityTriageSystem(buffer_size=10, iou_threshold=0.3)


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    monkeypatch.setenv("BOT_TOKEN", "test_token")
    monkeypatch.setenv("CHAT_ID", "test_chat")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./test_db.db")


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    db_path = tmp_path / "test_crashes.db"
    return f"sqlite:///{db_path}"
