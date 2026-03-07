import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.services.telegram import get_alert_service, send_telegram_alert
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_telegram():
    settings = get_settings()
    print(f"Checking Telegram Configuration:")
    print(f"  BOT_TOKEN: {'*' * 5 + settings.bot_token[-5:] if settings.bot_token else 'MISSING'}")
    print(f"  CHAT_ID: {settings.chat_id}")
    print(f"  Configured: {settings.telegram_configured}")
    
    if not settings.telegram_configured:
        print("❌ Telegram not configured in settings.")
        return
    
    print("\nAttempting to send test alert...")
    
    # Create dummy black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST ALERT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Send alert
    success = send_telegram_alert(
        confidence=0.99,
        frame=frame,
        severity_info={'severity_category': 'Severe', 'severity_index': 0.95, 'track_id': 123}
    )
    
    if success:
        print("✅ Alert sent successfully! Check your Telegram.")
    else:
        print("❌ Failed to send alert.")

if __name__ == "__main__":
    test_telegram()
