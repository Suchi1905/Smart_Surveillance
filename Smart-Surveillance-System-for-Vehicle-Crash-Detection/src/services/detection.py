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
try:
    from ..config import get_settings
except ImportError:
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
        self.crash_model = None  # best.pt - for crash detection
        self.object_model = None  # yolov8n.pt - for general object detection (person, car, etc.)
        self.face_model = None
        self.triage_system = SeverityTriageSystem()
        self.settings = get_settings()
        self.frame_counter = 0
        self.alert_sent = False
        self._alert_callback = None
        self._stop_event = threading.Event()
        # For backwards compatibility
        self.model = None
    
    def stop_stream(self):
        """Signal the frame generator to stop and release resources."""
        logger.info("Stopping detection stream...")
        self._stop_event.set()

    def load_models(self) -> Tuple[bool, bool]:
        """
        Load YOLO detection models.
        
        Loads two models:
        - best.pt: Custom crash detection model
        - yolov8n.pt: General object detection (person, car, etc.)
        
        Returns:
            Tuple of (crash_model_loaded, object_model_loaded)
        """
        crash_loaded = False
        object_loaded = False
        face_loaded = False
        
        try:
            from ultralytics import YOLO
            
            # Load crash detection model (best.pt)
            model_path = self.settings.find_model_path()
            if model_path:
                self.crash_model = YOLO(model_path)
                self.model = self.crash_model  # For backwards compatibility
                logger.info(f"Crash detection model loaded from {model_path}")
                crash_loaded = True
            else:
                logger.warning("Crash detection model (best.pt) not found")
        except Exception as e:
            logger.error(f"Failed to load crash detection model: {e}")
        
        try:
            from ultralytics import YOLO
            
            # Load general object detection model (yolov8n.pt)
            self.object_model = YOLO("yolov8n.pt")
            logger.info("Object detection model (yolov8n.pt) loaded")
            object_loaded = True
        except Exception as e:
            logger.error(f"Failed to load object detection model: {e}")
        
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
        
        return crash_loaded or object_loaded, face_loaded
    
    def _calculate_iou(
        self, 
        box1: Tuple[int, int, int, int], 
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
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
                if self.crash_model is None and self.object_model is None:
                    frame = self._draw_error_message(
                        frame, "No models loaded! Please check model files."
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
        Process a single frame through dual-model detection and severity analysis.
        
        Uses yolov8n.pt for general objects (person, car) and best.pt for crashes.
        
        Args:
            frame: Input BGR frame
            conf_threshold: Confidence threshold
            
        Returns:
            Annotated frame
        """
        crash_detections = []  # For severity triage
        person_boxes = []  # Track person detections to filter false positives
        
        # === Run general object detection (yolov8n.pt) ===
        if self.object_model is not None:
            results = self.object_model(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_name = self.object_model.names[cls]
                    
                    if conf > conf_threshold:
                        # Collect person boxes for filtering false crash detections
                        if cls_name.lower() == 'person':
                            person_boxes.append(tuple(xyxy))
                        
                        # Draw general object detection box (GREEN)
                        label = f"{cls_name} {conf:.2f}"
                        cv2.rectangle(
                            frame, 
                            (xyxy[0], xyxy[1]), 
                            (xyxy[2], xyxy[3]), 
                            (0, 255, 0),  # Green for general objects
                            2
                        )
                        cv2.putText(
                            frame, 
                            label, 
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 255, 0),  # Green text
                            2
                        )
        
        # === Run crash detection (best.pt) ===
        # Only process actual crash/accident classes, not vehicles
        CRASH_CLASSES = {'accident', 'mild', 'moderate', 'severe', 'crash', 'collision', 'wreck', 'impact'}
        SKIP_CLASSES = {'no accident', 'car', 'motor cycle', 'motorcycle', 'truck', 'bus', 'person'}
        
        if self.crash_model is not None:
            results = self.crash_model(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_name = self.crash_model.names[cls]
                    cls_lower = cls_name.lower()
                    
                    # Skip non-crash classes (vehicles without crashes, "No Accident")
                    if cls_lower in SKIP_CLASSES:
                        continue
                    
                    # Only process if it's an actual crash class
                    is_crash = any(crash_word in cls_lower for crash_word in CRASH_CLASSES)
                    
                    if conf > conf_threshold and is_crash:
                        crash_box = tuple(xyxy)
                        
                        # === FILTER: Check if crash overlaps with person boxes ===
                        # If crash detection significantly overlaps with person(s), it's likely a false positive
                        overlaps_with_person = False
                        for person_box in person_boxes:
                            iou = self._calculate_iou(crash_box, person_box)
                            if iou > 0.4:  # 40% overlap threshold
                                overlaps_with_person = True
                                logger.debug(f"Filtered false crash detection overlapping with person (IoU: {iou:.2f})")
                                break
                        
                        # Skip this crash detection if it overlaps with people
                        if overlaps_with_person:
                            continue
                        
                        crash_detections.append((crash_box, conf, cls_name))
                        
                        # Draw crash detection box (RED)
                        label = f"CRASH: {cls_name} {conf:.2f}"
                        cv2.rectangle(
                            frame, 
                            (xyxy[0], xyxy[1]), 
                            (xyxy[2], xyxy[3]), 
                            (0, 0, 255),  # Red for crashes
                            3  # Thicker line for crashes
                        )
                        cv2.putText(
                            frame, 
                            label, 
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 0, 255),  # Red text
                            2
                        )
        
        # Perform severity triage analysis ONLY on crash detections
        if crash_detections:
            severity_results = self.triage_system.analyze_accident(
                crash_detections, self.frame_counter
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
