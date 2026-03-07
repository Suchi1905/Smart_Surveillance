import sys
import os
import logging
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_service_logic():
    print("Importing Telegram Service...")
    try:
        from src.services.telegram import TelegramAlertService
        from src.config import get_settings
    except ImportError as e:
        print(f"Import Error: {e}")
        return

    settings = get_settings()
    print(f"Settings Configured: {settings.telegram_configured}")
    
    service = TelegramAlertService()
    print(f"Service Enabled: {service.enabled}")
    
    if not service.enabled:
        print("Service is disabled via config logic.")
        return

    print("Creating dummy frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "SERVICE TEST", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Sending alert via Service...")
    try:
        success = service.send_alert(
            confidence=0.88,
            frame=frame,
            severity_info={'severity_category': 'Severe', 'severity_index': 0.8, 'track_id': 999}
        )
        
        if success:
            print("✅ Service logic SUCCESS")
        else:
            print("❌ Service logic FAILED")
            
    except Exception as e:
        print(f"❌ Exception during send_alert: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_service_logic()
