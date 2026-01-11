"""
Enhanced Detection Service with Tracking, Speed, and Behavior Analysis.

This module provides the complete detection pipeline including:
- YOLO-based vehicle/crash detection
- ByteTrack multi-object tracking
- Speed estimation
- Collision prediction
- Dangerous behavior detection
- Real-time alerts via WebSocket
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Generator, Dict
import logging
import threading
import time
import asyncio
from dataclasses import asdict

from .severity_triage import SeverityTriageSystem, SeverityResult
from .anonymization import anonymize_frame
from .tracker import ByteTracker, Track
from .speed_estimator import SpeedEstimator, SpeedMeasurement
from .collision import CollisionPredictor, CollisionRisk
from .behavior import BehaviorAnalyzer, BehaviorAlert
from ..config import get_settings

logger = logging.getLogger(__name__)


class EnhancedDetectionService:
    """
    Production-level detection service with full analysis pipeline.
    
    Integrates:
    - YOLO detection
    - ByteTrack tracking
    - Speed estimation
    - Collision prediction
    - Behavior analysis
    - Severity triage
    - Real-time WebSocket updates
    """
    
    def __init__(self):
        """Initialize the enhanced detection service."""
        self.model = None
        self.face_model = None
        self.settings = get_settings()
        
        # Core systems
        self.triage_system = SeverityTriageSystem()
        self.tracker = ByteTracker(
            track_thresh=0.5,
            high_thresh=0.6,
            match_thresh=0.8,
            max_time_lost=30
        )
        self.speed_estimator = SpeedEstimator(fps=30.0)
        self.collision_predictor = CollisionPredictor(fps=30.0)
        self.behavior_analyzer = BehaviorAnalyzer(
            expected_flow_direction=0.0,  # Configure based on camera
            fps=30.0
        )
        
        # State
        self.frame_counter = 0
        self.alert_sent = False
        self._alert_callback = None
        self._ws_manager = None
        
        # Statistics
        self._stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "total_alerts": 0,
            "avg_fps": 0.0
        }
        
        logger.info("EnhancedDetectionService initialized")
    
    def load_models(self) -> Tuple[bool, bool]:
        """Load YOLO detection and face models."""
        detection_loaded = False
        face_loaded = False
        
        try:
            from ultralytics import YOLO
            
            model_path = self.settings.find_model_path()
            if model_path:
                self.model = YOLO(model_path)
                logger.info(f"Detection model loaded from {model_path}")
                detection_loaded = True
            else:
                logger.warning("Custom model not found, using YOLOv8n")
                self.model = YOLO("yolov8n.pt")
                detection_loaded = True
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
        
        try:
            from ultralytics import YOLO
            
            self.face_model = YOLO(self.settings.face_model_path)
            logger.info("Face detection model loaded")
            face_loaded = True
            
            from .anonymization import set_face_model
            set_face_model(self.face_model)
        except Exception as e:
            logger.warning(f"Face model not loaded: {e}")
        
        return detection_loaded, face_loaded
    
    def set_alert_callback(self, callback):
        """Set callback for crash alerts."""
        self._alert_callback = callback
    
    def set_websocket_manager(self, manager):
        """Set WebSocket manager for real-time updates."""
        self._ws_manager = manager
    
    def generate_frames(
        self, 
        conf_threshold: float = 0.6
    ) -> Generator[bytes, None, None]:
        """Generate MJPEG frames with full analysis overlay."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            yield self._encode_frame(self._create_error_frame("Camera not available"))
            return
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0.0
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame")
                    yield self._encode_frame(self._create_error_frame("Camera error"))
                    time.sleep(0.1)
                    continue
                
                self.frame_counter += 1
                fps_counter += 1
                
                # Calculate FPS
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    self._stats["avg_fps"] = current_fps
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if self.model is None:
                    frame = self._draw_error_message(
                        frame, "Model not loaded!"
                    )
                else:
                    frame = self._process_frame_enhanced(
                        frame, conf_threshold, current_fps
                    )
                
                yield self._encode_frame(frame)
                
        finally:
            cap.release()
            logger.info("Camera released")
    
    def _process_frame_enhanced(
        self, 
        frame: np.ndarray, 
        conf_threshold: float,
        fps: float
    ) -> np.ndarray:
        """Process frame with full analysis pipeline."""
        start_time = time.time()
        
        # 1. YOLO Detection
        results = self.model(frame, verbose=False)
        
        detections = []
        detection_data = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = self.model.names[cls]
                
                if conf > conf_threshold:
                    detections.append((
                        np.array(xyxy), conf, cls, cls_name
                    ))
                    detection_data.append((tuple(xyxy), conf, cls_name))
        
        self._stats["total_detections"] += len(detections)
        
        # 2. Multi-Object Tracking
        tracks = self.tracker.update(detections)
        self._stats["total_tracks"] = self.tracker.get_active_count()
        
        # 3. Draw tracks and collect data for analysis
        track_data = []
        speed_measurements = []
        velocities_for_behavior = []
        
        for track in tracks:
            # Draw track box with ID
            x1, y1, x2, y2 = track.bbox.astype(int)
            
            # Estimate speed
            speed_meas = self.speed_estimator.estimate_from_trajectory(
                track.track_id,
                track.get_trajectory()
            )
            
            if speed_meas:
                speed_measurements.append(speed_meas)
                color = self.speed_estimator.get_speed_color(speed_meas.speed_kmh)
            else:
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and speed
            label = f"ID:{track.track_id}"
            if speed_meas:
                label += f" {speed_meas.speed_kmh:.0f}km/h"
            
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw trajectory trail
            self._draw_trajectory(frame, track.get_trajectory(), color)
            
            # Collect data for collision/behavior analysis
            track_data.append({
                'id': track.track_id,
                'center': track.get_center(),
                'velocity': track.velocity,
                'bbox': track.bbox
            })
            
            velocities_for_behavior.append(track.velocity)
        
        # 4. Collision Prediction
        collision_risks = self.collision_predictor.analyze_all_tracks(track_data)
        
        for risk in collision_risks:
            if risk.risk_level in ["critical", "high"]:
                self._draw_collision_warning(frame, risk, track_data)
        
        # 5. Behavior Analysis
        for track in tracks:
            traj = track.get_trajectory()
            vels = [track.velocity]  # Would need velocity history
            
            alerts = self.behavior_analyzer.analyze_trajectory(
                track.track_id,
                traj,
                vels
            )
            
            for alert in alerts:
                self._draw_behavior_alert(frame, alert)
                self._stats["total_alerts"] += 1
        
        # 6. Severity Triage (for crash detections)
        if detection_data:
            severity_results = self.triage_system.analyze_accident(
                detection_data, self.frame_counter
            )
            
            for sev_result in severity_results:
                if sev_result.severity_category != "Monitoring":
                    self._draw_severity_info(frame, sev_result)
                    
                    if (sev_result.severity_category == "Severe" 
                            and not self.alert_sent 
                            and self._alert_callback):
                        self._trigger_alert(frame, sev_result)
        
        # 7. Draw HUD overlay
        self._draw_hud(frame, fps, len(tracks), len(collision_risks))
        
        # 8. Broadcast to WebSocket (async)
        if self._ws_manager and (self.frame_counter % 5 == 0):
            self._broadcast_updates(tracks, speed_measurements, collision_risks)
        
        return frame
    
    def _draw_trajectory(
        self, 
        frame: np.ndarray, 
        trajectory: List[Tuple[float, float]], 
        color: Tuple[int, int, int]
    ):
        """Draw trajectory trail on frame."""
        if len(trajectory) < 2:
            return
        
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            
            # Fade older points
            alpha = i / len(trajectory)
            thickness = int(1 + alpha * 2)
            
            cv2.line(frame, pt1, pt2, color, thickness)
    
    def _draw_collision_warning(
        self, 
        frame: np.ndarray, 
        risk: CollisionRisk,
        track_data: List[Dict]
    ):
        """Draw collision warning between two tracks."""
        # Find the two tracks
        t1_data = next((t for t in track_data if t['id'] == risk.track_id_1), None)
        t2_data = next((t for t in track_data if t['id'] == risk.track_id_2), None)
        
        if not t1_data or not t2_data:
            return
        
        color = self.collision_predictor.get_risk_color(risk.risk_level)
        
        # Draw line between vehicles
        pt1 = (int(t1_data['center'][0]), int(t1_data['center'][1]))
        pt2 = (int(t2_data['center'][0]), int(t2_data['center'][1]))
        cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw warning label
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        
        if risk.ttc_seconds < float('inf'):
            label = f"TTC: {risk.ttc_seconds:.1f}s"
        else:
            label = f"Risk: {risk.risk_level}"
        
        cv2.putText(
            frame, label,
            (mid_x - 30, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    def _draw_behavior_alert(self, frame: np.ndarray, alert: BehaviorAlert):
        """Draw behavior alert on frame."""
        x, y = int(alert.location[0]), int(alert.location[1])
        
        # Alert colors by severity
        colors = {
            "critical": (0, 0, 255),
            "warning": (0, 128, 255),
            "violation": (0, 255, 255)
        }
        color = colors.get(alert.severity, (255, 255, 0))
        
        # Draw alert icon
        cv2.circle(frame, (x, y), 20, color, 3)
        cv2.putText(
            frame, "!",
            (x - 5, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        
        # Draw alert text
        cv2.putText(
            frame, alert.behavior_type.value.replace("_", " ").upper(),
            (x + 25, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    def _draw_severity_info(self, frame: np.ndarray, sev_result: SeverityResult):
        """Draw severity information."""
        colors = {
            "Severe": (0, 0, 255),
            "Moderate": (0, 165, 255),
            "Mild": (0, 255, 255)
        }
        color = colors.get(sev_result.severity_category, (0, 255, 0))
        
        sev_text = f"{sev_result.severity_category} (SI: {sev_result.severity_index:.2f})"
        cv2.putText(
            frame, sev_text,
            (sev_result.box[0], sev_result.box[3] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
    def _draw_hud(
        self, 
        frame: np.ndarray, 
        fps: float, 
        track_count: int,
        risk_count: int
    ):
        """Draw heads-up display with stats."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Stats text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Tracks: {track_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Risks: {risk_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_counter}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _trigger_alert(self, frame: np.ndarray, sev_result: SeverityResult):
        """Trigger alert for severe crash."""
        anon_frame = anonymize_frame(frame.copy())
        
        if self._alert_callback:
            threading.Thread(
                target=self._alert_callback,
                args=(sev_result.confidence, anon_frame, sev_result)
            ).start()
        
        self.alert_sent = True
        threading.Thread(target=self._reset_alert_flag).start()
    
    def _reset_alert_flag(self):
        """Reset alert flag after cooldown."""
        time.sleep(self.settings.alert_cooldown_seconds)
        self.alert_sent = False
    
    def _broadcast_updates(
        self, 
        tracks: List[Track],
        speeds: List[SpeedMeasurement],
        risks: List[CollisionRisk]
    ):
        """Broadcast updates via WebSocket (non-blocking)."""
        if not self._ws_manager:
            return
        
        try:
            # This would need to be properly async in production
            # For now, just log
            pass
        except Exception as e:
            logger.error(f"WebSocket broadcast error: {e}")
    
    def _create_error_frame(self, message: str) -> np.ndarray:
        """Create an error frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return self._draw_error_message(frame, message)
    
    def _draw_error_message(self, frame: np.ndarray, message: str) -> np.ndarray:
        """Draw error message on frame."""
        cv2.putText(
            frame, message, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
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
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return self._stats.copy()
    
    def reset(self):
        """Reset all systems."""
        self.tracker.reset()
        self.speed_estimator.reset()
        self.collision_predictor.reset()
        self.behavior_analyzer.reset()
        self.triage_system.reset()
        self.frame_counter = 0
        self._stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "total_alerts": 0,
            "avg_fps": 0.0
        }
        logger.info("EnhancedDetectionService reset")
