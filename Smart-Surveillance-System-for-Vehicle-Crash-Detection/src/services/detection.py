"""
Detection service for YOLO-based crash detection.

This module provides the detection pipeline including model loading,
inference, and result processing.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Generator
import logging
import threading
import time

from .severity_triage import SeverityTriageSystem, SeverityResult
from .anonymization import anonymize_frame
from config import get_settings

logger = logging.getLogger(__name__)


class DetectionService:
    """
    YOLO-based detection service for crash detection.
    
    Handles model loading, video capture, inference, and frame generation
    for the streaming endpoint.
    
    Attributes:
        model: YOLO detection model
        face_model: YOLO face detection model for anonymization
        triage_system: Severity triage system instance
        settings: Application settings
    """
    
    def __init__(self):
        """Initialize the detection service."""
        self.model = None
        self.face_model = None
        self.triage_system = SeverityTriageSystem()
        self.settings = get_settings()
        self.frame_counter = 0
        self.alert_sent = False
        self._alert_callback = None
        self._stop_event = threading.Event()
    
    def stop_stream(self):
        """Signal the frame generator to stop and release resources."""
        logger.info("Stopping detection stream...")
        self._stop_event.set()

    def load_models(self) -> Tuple[bool, bool]:
        """
        Load YOLO detection and face models.
        
        Returns:
            Tuple of (detection_model_loaded, face_model_loaded)
        """
        detection_loaded = False
        face_loaded = False
        
        try:
            from ultralytics import YOLO
            
            # Load detection model
            model_path = self.settings.find_model_path()
            if model_path:
                self.model = YOLO(model_path)
                logger.info(f"Detection model loaded from {model_path}")
                detection_loaded = True
            else:
                # Fallback to default YOLOv8n
                logger.warning("Custom model not found, using YOLOv8n")
                self.model = YOLO("yolov8n.pt")
                detection_loaded = True
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
        
        try:
            from ultralytics import YOLO
            
            # Load face model for anonymization
            self.face_model = YOLO(self.settings.face_model_path)
            logger.info("Face detection model loaded")
            face_loaded = True
            
            # Set model reference for anonymization service
            from .anonymization import set_face_model
            set_face_model(self.face_model)
        except Exception as e:
            logger.warning(f"Face model not loaded: {e}")
        
        return detection_loaded, face_loaded
    
    def set_alert_callback(self, callback):
        """Set callback function for crash alerts."""
        self._alert_callback = callback
    
    def generate_frames(
        self, 
        conf_threshold: float = 0.6
    ) -> Generator[bytes, None, None]:
        """
        Generate MJPEG frames with detection overlay.
        
        Args:
            conf_threshold: Confidence threshold for detections
            
        Yields:
            MJPEG frame bytes
        """
        self._stop_event.clear()
        cap = cv2.VideoCapture(0)
        logger.info("Camera opened for streaming")
        
        try:
            while not self._stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    # Yield error frame
                    error_frame = self._create_error_frame("Camera not available")
                    yield self._encode_frame(error_frame)
                    time.sleep(0.1)
                    continue
                
                self.frame_counter += 1
                
                # Process frame
                if self.model is None:
                    frame = self._draw_error_message(
                        frame, "Model not loaded! Please train a model first"
                    )
                else:
                    frame = self._process_frame(frame, conf_threshold)
                
                yield self._encode_frame(frame)
                
        finally:
            cap.release()
            logger.info("Camera released")
    
    def _process_frame(
        self, 
        frame: np.ndarray, 
        conf_threshold: float
    ) -> np.ndarray:
        """
        Process a single frame through detection and severity analysis.
        
        Args:
            frame: Input BGR frame
            conf_threshold: Confidence threshold
            
        Returns:
            Annotated frame
        """
        results = self.model(frame, verbose=False)
        
        # Collect detections for triage analysis
        detections = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = self.model.names[cls]
                
                if conf > conf_threshold:
                    detections.append((tuple(xyxy), conf, cls_name))
                    
                    # Draw detection box
                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(
                        frame, 
                        (xyxy[0], xyxy[1]), 
                        (xyxy[2], xyxy[3]), 
                        (0, 0, 255), 
                        2
                    )
                    cv2.putText(
                        frame, 
                        label, 
                        (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )
        
        # Perform severity triage analysis
        if detections:
            severity_results = self.triage_system.analyze_accident(
                detections, self.frame_counter
            )
            
            # Draw severity information
            for sev_result in severity_results:
                if sev_result.severity_category != "Monitoring":
                    self._draw_severity_info(frame, sev_result)
                    
                    # Handle severe accidents
                    if (sev_result.severity_category == "Severe" 
                            and not self.alert_sent 
                            and self._alert_callback):
                        self._trigger_alert(frame, sev_result)
        
        return frame
    
    def _draw_severity_info(
        self, 
        frame: np.ndarray, 
        sev_result: SeverityResult
    ) -> None:
        """Draw severity information on frame."""
        sev_text = f"{sev_result.severity_category} (SI: {sev_result.severity_index:.2f})"
        cv2.putText(
            frame, 
            sev_text, 
            (sev_result.box[0], sev_result.box[3] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
    
    def _trigger_alert(
        self, 
        frame: np.ndarray, 
        sev_result: SeverityResult
    ) -> None:
        """Trigger alert for severe crash."""
        # Anonymize frame before sending
        anon_frame = anonymize_frame(frame.copy())
        
        # Send alert in background thread
        if self._alert_callback:
            threading.Thread(
                target=self._alert_callback,
                args=(sev_result.confidence, anon_frame, sev_result)
            ).start()
        
        # Set cooldown
        self.alert_sent = True
        threading.Thread(target=self._reset_alert_flag).start()
    
    def _reset_alert_flag(self) -> None:
        """Reset alert flag after cooldown."""
        time.sleep(self.settings.alert_cooldown_seconds)
        self.alert_sent = False
    
    def _create_error_frame(self, message: str) -> np.ndarray:
        """Create an error frame with message."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return self._draw_error_message(frame, message)
    
    def _draw_error_message(
        self, 
        frame: np.ndarray, 
        message: str
    ) -> np.ndarray:
        """Draw error message on frame."""
        cv2.putText(
            frame, 
            message, 
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2
        )
        return frame
    
    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode frame as MJPEG bytes."""
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        return (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
