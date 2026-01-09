"""
Integration tests for FastAPI endpoints.

Tests cover:
- Health check endpoint
- System status endpoints
- Crash event CRUD endpoints
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Skip tests if FastAPI not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    # Import here to avoid issues if dependencies not installed
    from main import app
    from database import init_db
    
    # Initialize test database
    init_db()
    
    return TestClient(app)


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, test_client):
        """Test health endpoint returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, test_client):
        """Test health response has correct structure."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_health_status_value(self, test_client):
        """Test health status is 'healthy'."""
        response = test_client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"


@pytest.mark.integration
class TestSystemEndpoints:
    """Tests for system status endpoints."""
    
    def test_system_status_v1(self, test_client):
        """Test v1 system status endpoint."""
        response = test_client.get("/api/v1/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "ml_service" in data
        assert "database" in data
        assert "triage" in data
        assert "anonymization" in data
    
    def test_legacy_system_status(self, test_client):
        """Test legacy system status endpoint."""
        response = test_client.get("/api/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "ml_service" in data
    
    def test_config_endpoint(self, test_client):
        """Test config endpoint."""
        response = test_client.get("/api/v1/system/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "confidence_threshold" in data
        assert "anonymization_enabled" in data


@pytest.mark.integration
class TestCrashEndpoints:
    """Tests for crash event CRUD endpoints."""
    
    def test_list_crashes_empty(self, test_client):
        """Test listing crashes when empty."""
        response = test_client.get("/api/v1/crashes")
        assert response.status_code == 200
        
        data = response.json()
        assert "total" in data
        assert "events" in data
        assert isinstance(data["events"], list)
    
    def test_create_crash(self, test_client):
        """Test creating a crash event."""
        crash_data = {
            "confidence": 0.85,
            "class_name": "Accident",
            "severity_category": "Severe",
            "severity_index": 0.75,
            "camera_id": "CAM-01",
            "location": "Test Location"
        }
        
        response = test_client.post("/api/v1/crashes", json=crash_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["confidence"] == 0.85
        assert data["class_name"] == "Accident"
        assert data["severity_category"] == "Severe"
        assert "id" in data
        assert "timestamp" in data
    
    def test_get_crash_by_id(self, test_client):
        """Test getting a specific crash by ID."""
        # First create a crash
        crash_data = {
            "confidence": 0.9,
            "class_name": "Accident",
            "severity_category": "Moderate",
            "severity_index": 0.5
        }
        create_response = test_client.post("/api/v1/crashes", json=crash_data)
        crash_id = create_response.json()["id"]
        
        # Now get it
        response = test_client.get(f"/api/v1/crashes/{crash_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == crash_id
        assert data["confidence"] == 0.9
    
    def test_get_nonexistent_crash(self, test_client):
        """Test getting a crash that doesn't exist."""
        response = test_client.get("/api/v1/crashes/99999")
        assert response.status_code == 404
    
    def test_crash_stats(self, test_client):
        """Test crash statistics endpoint."""
        response = test_client.get("/api/v1/crashes/stats/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_events" in data
        assert "events_today" in data
        assert "severity_breakdown" in data
        assert "average_confidence" in data
    
    def test_filter_by_severity(self, test_client):
        """Test filtering crashes by severity."""
        # Create crashes with different severities
        for severity in ["Severe", "Moderate", "Mild"]:
            test_client.post("/api/v1/crashes", json={
                "confidence": 0.8,
                "class_name": "Accident",
                "severity_category": severity,
                "severity_index": 0.5
            })
        
        # Filter by severity
        response = test_client.get("/api/v1/crashes?severity=Severe")
        assert response.status_code == 200
        
        data = response.json()
        for event in data["events"]:
            assert event["severity_category"] == "Severe"
    
    def test_pagination(self, test_client):
        """Test crash list pagination."""
        # Create several crashes
        for i in range(5):
            test_client.post("/api/v1/crashes", json={
                "confidence": 0.8,
                "class_name": "Accident",
                "severity_category": "Moderate",
                "severity_index": 0.5
            })
        
        # Test limit
        response = test_client.get("/api/v1/crashes?limit=2")
        data = response.json()
        assert len(data["events"]) <= 2
        
        # Test skip
        response = test_client.get("/api/v1/crashes?skip=2&limit=2")
        data = response.json()
        assert len(data["events"]) <= 2


@pytest.mark.integration
class TestVideoEndpoint:
    """Tests for video streaming endpoint."""
    
    def test_video_endpoint_exists(self, test_client):
        """Test video endpoint is accessible."""
        # Just test that endpoint exists (streaming test needs special handling)
        response = test_client.get("/video?conf=0.6", timeout=1)
        # Should return streaming response or error
        assert response.status_code in [200, 500]  # 500 if no camera


@pytest.mark.integration
class TestLegacyEndpoints:
    """Tests for legacy API compatibility."""
    
    def test_legacy_status(self, test_client):
        """Test legacy /api/status endpoint."""
        response = test_client.get("/api/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "detection" in data
        assert "triage" in data
        assert "anonymization" in data
    
    def test_legacy_config(self, test_client):
        """Test legacy /api/config endpoint."""
        response = test_client.get("/api/config")
        assert response.status_code == 200
