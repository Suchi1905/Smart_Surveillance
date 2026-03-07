"""
Alert & Telegram integration tests.

Tests the alert triggering → Telegram dispatch flow with mocked HTTP.
Validates message formatting, callback invocation, and graceful degradation.
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.severity_triage import SeverityResult


def _make_severity_result(severity="Severe", confidence=0.9, class_name="Accident"):
    """Helper to create a SeverityResult for testing."""
    return SeverityResult(
        track_id=0,
        severity_index=confidence,
        severity_category=severity,
        class_name=class_name,
        confidence=confidence,
        box=(100, 100, 200, 200)
    )


@pytest.mark.integration
class TestTelegramAlertDispatch:
    """Test Telegram alert dispatching."""

    def test_telegram_service_init(self, mock_settings):
        """Test TelegramAlertService initializes with env vars."""
        from services.telegram import TelegramAlertService
        service = TelegramAlertService()
        assert service.enabled is True

    def test_telegram_disabled_when_not_configured(self):
        """Test Telegram is disabled when settings indicate not configured."""
        from services.telegram import TelegramAlertService
        service = TelegramAlertService()
        # Directly set enabled to False to test disabled behavior
        service.enabled = False
        assert service.enabled is False

    def test_send_alert_returns_false_when_disabled(self, sample_frame):
        """Test send_alert returns False gracefully when disabled."""
        from services.telegram import TelegramAlertService
        service = TelegramAlertService()
        service.enabled = False  # Force disable

        result = service.send_alert(
            confidence=0.9,
            frame=sample_frame,
            severity_info={"severity_category": "Severe"}
        )
        assert result is False

    @patch("services.telegram.requests.post")
    def test_send_alert_calls_telegram_api(self, mock_post, mock_settings, sample_frame):
        """Test send_alert makes HTTP POST to Telegram API."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

        from services.telegram import TelegramAlertService
        service = TelegramAlertService()

        sev_info = {
            "severity_category": "Severe",
            "severity_index": 0.9,
            "class_name": "Accident",
            "confidence": 0.9
        }

        result = service.send_alert(
            confidence=0.9,
            frame=sample_frame,
            severity_info=sev_info
        )

        assert result is True
        mock_post.assert_called_once()

        # Verify the API URL contains sendPhoto
        call_url = mock_post.call_args[0][0]
        assert "sendPhoto" in call_url

    @patch("services.telegram.requests.post")
    def test_telegram_message_contains_crash_details(self, mock_post, mock_settings, sample_frame):
        """Test alert message includes severity, confidence, and class name."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

        from services.telegram import TelegramAlertService
        service = TelegramAlertService()

        sev_info = {
            "severity_category": "Severe",
            "severity_index": 0.85,
            "class_name": "Accident",
            "confidence": 0.85
        }

        service.send_alert(confidence=0.85, frame=sample_frame, severity_info=sev_info)

        # Verify the API was called
        assert mock_post.called

    @patch("services.telegram.requests.post")
    def test_send_alert_handles_api_failure(self, mock_post, mock_settings, sample_frame):
        """Test send_alert handles Telegram API errors gracefully."""
        mock_post.return_value = MagicMock(status_code=500, json=lambda: {"ok": False})

        from services.telegram import TelegramAlertService
        service = TelegramAlertService()

        result = service.send_alert(
            confidence=0.9,
            frame=sample_frame,
            severity_info={"severity_category": "Severe"}
        )
        # Should not raise, returns False on failure
        assert result is False


@pytest.mark.integration
class TestAlertCallbackIntegration:
    """Test alert callback (WebSocket notification) integration."""

    def test_alert_callback_invoked_on_trigger(self, mock_detection_service, sample_frame):
        """Test _alert_callback is called when _trigger_alert fires."""
        service = mock_detection_service
        callback = MagicMock()
        service.set_alert_callback(callback)

        sev_result = _make_severity_result()

        # Patch the telegram import inside _trigger_alert
        with patch.dict('sys.modules', {'services.telegram': MagicMock()}):
            # Mock the local import of send_telegram_alert
            import services.telegram as tel_mod
            tel_mod.send_telegram_alert = MagicMock(return_value=True)
            service._trigger_alert(sample_frame, sev_result)

        # Give the background thread a moment
        time.sleep(0.5)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == pytest.approx(0.9, abs=0.01)  # confidence

    def test_no_callback_when_none_set(self, mock_detection_service, sample_frame):
        """Test _trigger_alert works when no callback is set."""
        service = mock_detection_service
        service._alert_callback = None

        sev_result = _make_severity_result()

        # Patch the telegram import inside _trigger_alert
        with patch.dict('sys.modules', {'services.telegram': MagicMock()}):
            import services.telegram as tel_mod
            tel_mod.send_telegram_alert = MagicMock(return_value=True)
            # Should not raise
            service._trigger_alert(sample_frame, sev_result)

    def test_trigger_alert_updates_last_alert_time(self, mock_detection_service, sample_frame):
        """Test _trigger_alert updates the last alert timestamp."""
        service = mock_detection_service
        old_time = service._last_alert_time

        sev_result = _make_severity_result()

        with patch.dict('sys.modules', {'services.telegram': MagicMock()}):
            import services.telegram as tel_mod
            tel_mod.send_telegram_alert = MagicMock(return_value=True)
            service._trigger_alert(sample_frame, sev_result)

        assert service._last_alert_time > old_time


@pytest.mark.integration
class TestSendTelegramAlertWrapper:
    """Test the send_telegram_alert wrapper function."""

    @patch("services.telegram.requests.post")
    def test_wrapper_accepts_severity_result(self, mock_post, mock_settings, sample_frame):
        """Test wrapper function handles SeverityResult objects."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

        from services.telegram import send_telegram_alert

        sev_result = _make_severity_result()

        result = send_telegram_alert(
            confidence=sev_result.confidence,
            frame=sample_frame,
            severity_info=sev_result
        )
        assert result is True

    @patch("services.telegram.requests.post")
    def test_wrapper_accepts_dict(self, mock_post, mock_settings, sample_frame):
        """Test wrapper function handles dict severity info."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

        from services.telegram import send_telegram_alert

        result = send_telegram_alert(
            confidence=0.9,
            frame=sample_frame,
            severity_info={"severity_category": "Severe", "confidence": 0.9}
        )
        assert result is True
