"""
Frame anonymization service for privacy compliance.

This module provides face and license plate anonymization using
Gaussian blur for GDPR/privacy compliance.

Supports two face detection methods:
1. YOLO face model (preferred, more accurate)
2. OpenCV Haar Cascade (fallback, no extra model needed)
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Global face model reference
_face_model = None

# OpenCV Haar Cascade for face detection (fallback)
_haar_cascade = None


def _get_haar_cascade():
    """Get or initialize the Haar Cascade face detector."""
    global _haar_cascade
    if _haar_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _haar_cascade = cv2.CascadeClassifier(cascade_path)
        if _haar_cascade.empty():
            logger.warning("Failed to load Haar Cascade")
            _haar_cascade = None
        else:
            logger.info("OpenCV Haar Cascade loaded for face detection")
    return _haar_cascade


def set_face_model(model):
    """Set the face detection model."""
    global _face_model
    _face_model = model


def anonymize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Anonymize faces and license plates in the frame using Gaussian blur.
    
    This function ensures GDPR/privacy compliance by:
    1. Detecting and blurring faces using YOLO or OpenCV Haar Cascade
    2. Detecting and blurring license plates using HSV color filtering
    
    Args:
        frame: Input BGR image as numpy array
        
    Returns:
        Anonymized frame with blurred faces and license plates
    """
    anonymized_frame = frame.copy()
    
    try:
        # Stage 1: Face Detection (using YOLO or Haar Cascade fallback)
        anonymized_frame = _blur_faces(anonymized_frame)
        
        # Stage 2: License Plate Detection
        anonymized_frame = _blur_license_plates(anonymized_frame, frame)
        
        logger.debug("Frame anonymized successfully")
        
    except Exception as e:
        logger.warning(f"Anonymization error: {e}")
        return frame
    
    return anonymized_frame


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    """
    Detect and blur faces in the frame.
    
    Uses YOLO face model if available, otherwise falls back to
    OpenCV Haar Cascade for face detection.
    
    Args:
        frame: Input BGR image
        
    Returns:
        Frame with blurred faces
    """
    # Try YOLO face model first (more accurate)
    if _face_model is not None:
        try:
            face_results = _face_model(frame, verbose=False)
            
            for result in face_results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    # Apply Gaussian blur to face region
                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                        frame[y1:y2, x1:x2] = blurred_roi
            
            return frame
        except Exception as e:
            logger.warning(f"YOLO face detection error: {e}")
    
    # Fallback to Haar Cascade
    haar = _get_haar_cascade()
    if haar is not None:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Apply Gaussian blur to face region
                roi = frame[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                frame[y:y+h, x:x+w] = blurred_roi
                
        except Exception as e:
            logger.warning(f"Haar Cascade face detection error: {e}")
    
    return frame


def _blur_license_plates(
    anonymized_frame: np.ndarray, 
    original_frame: np.ndarray
) -> np.ndarray:
    """
    Detect and blur license plates using color-based heuristics.
    
    Uses HSV color filtering to detect white/yellow rectangular regions
    with typical license plate aspect ratios.
    
    Args:
        anonymized_frame: Frame to apply blur to
        original_frame: Original frame for detection
        
    Returns:
        Frame with blurred license plates
    """
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
        
        # White/yellow plate detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (typical license plate size)
            if 500 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # License plates typically have aspect ratio between 2:1 and 5:1
                if 2.0 < aspect_ratio < 5.0:
                    # Apply Gaussian blur
                    roi = anonymized_frame[y:y+h, x:x+w]
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    anonymized_frame[y:y+h, x:x+w] = blurred_roi
    
    except Exception as e:
        logger.warning(f"License plate detection error: {e}")
    
    return anonymized_frame


def get_privacy_metrics(frame: np.ndarray) -> dict:
    """
    Get privacy anonymization metrics for a frame.
    
    Args:
        frame: Input BGR image
        
    Returns:
        Dictionary with detection counts and processing info
    """
    # Anonymization is always enabled (using YOLO or Haar Cascade)
    haar = _get_haar_cascade()
    metrics = {
        "faces_detected": 0,
        "plates_detected": 0,
        "anonymization_enabled": _face_model is not None or haar is not None,
        "method": "yolo" if _face_model is not None else ("haar" if haar is not None else "none")
    }
    
    try:
        # Count faces using YOLO or Haar Cascade
        if _face_model is not None:
            face_results = _face_model(frame, verbose=False)
            for result in face_results:
                metrics["faces_detected"] += len(result.boxes)
        elif haar is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            metrics["faces_detected"] = len(faces)
        
        # Count potential plates (rough estimate)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 2.0 < aspect_ratio < 5.0:
                    metrics["plates_detected"] += 1
    
    except Exception as e:
        logger.warning(f"Error getting privacy metrics: {e}")
    
    return metrics

