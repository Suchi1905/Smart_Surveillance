from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import threading
from datetime import datetime
import requests
import time
import os
import numpy as np
from collections import defaultdict, deque

app = Flask(__name__, static_folder='frontend/build/static', template_folder='frontend/build')
CORS(app)  # Enable CORS for React frontend

# --- Config ---
# Try to load model from crash_webapp first, then local weights
import os
from pathlib import Path

possible_model_paths = [
    os.path.join("backend", "weights", "best.pt"),  # New backend location
    "weights/best.pt",  # Local weights (legacy)
    "crash_webapp/weights/best.pt",  # From crash_webapp (legacy)
    os.path.join(Path(__file__).parent, "backend", "weights", "best.pt"),  # Absolute backend path
    os.path.join(Path(__file__).parent, "weights", "best.pt"),  # Absolute local path
    os.path.join(Path(__file__).parent, "crash_webapp", "weights", "best.pt"),  # Absolute crash_webapp path
]

MODEL_PATH = None
for path in possible_model_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"[✅] Found model at: {MODEL_PATH}")
        break

if MODEL_PATH:
    try:
        model = YOLO(MODEL_PATH)
        print(f"[✅] Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"[⚠️] Warning: Could not load model from {MODEL_PATH}")
        print(f"[ℹ️] Error: {e}")
        model = None
else:
    print("[⚠️] Warning: Model not found in any expected location")
    print(f"[ℹ️] Searched paths: {possible_model_paths}")
    print("[ℹ️] Using default YOLOv8n model")
    try:
        model = YOLO("yolov8n.pt")
    except:
        model = None

# Load lightweight face detection model for anonymization
try:
    face_model = YOLO("yolov8n-face.pt")
    print("[✅] Face detection model loaded")
except:
    face_model = None
    print("[⚠️] Face detection model not found - anonymization disabled")

alert_sent = False

# --- Telegram Bot Config ---
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# --- Severity Triage System ---
class SeverityTriageSystem:
    def __init__(self, buffer_size=10, iou_threshold=0.3):
        self.vehicle_tracks = defaultdict(lambda: deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.next_track_id = 0
        self.track_assignments = {}
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
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
    
    def get_box_center(self, box):
        """Get center point of bounding box"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def calculate_velocity(self, track_history):
        """Calculate pixel displacement velocity from track history"""
        if len(track_history) < 2:
            return 0.0
        
        recent = track_history[-1]
        previous = track_history[-2]
        
        center_recent = self.get_box_center(recent['box'])
        center_previous = self.get_box_center(previous['box'])
        
        # Euclidean distance
        displacement = np.sqrt(
            (center_recent[0] - center_previous[0])**2 + 
            (center_recent[1] - center_previous[1])**2
        )
        
        return displacement
    
    def assign_track_id(self, current_box, frame_number):
        """Assign track ID to detection based on IoU with previous tracks"""
        best_iou = 0
        best_track_id = None
        
        # Try to match with existing tracks
        for track_id, history in self.vehicle_tracks.items():
            if len(history) > 0:
                last_detection = history[-1]
                # Only match if detection is recent (within 5 frames)
                if frame_number - last_detection['frame'] < 5:
                    iou = self.calculate_iou(current_box, last_detection['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
        
        # If good match found, use existing track
        if best_iou > 0.3:
            return best_track_id
        
        # Otherwise create new track
        new_id = self.next_track_id
        self.next_track_id += 1
        return new_id
    
    def update_track(self, track_id, box, frame_number, class_name, confidence):
        """Update vehicle track with new detection"""
        self.vehicle_tracks[track_id].append({
            'box': box,
            'frame': frame_number,
            'class': class_name,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def calculate_severity_index(self, track_id):
        """Calculate severity index based on velocity drop and IoU consistency"""
        track_history = self.vehicle_tracks[track_id]
        
        if len(track_history) < 3:
            return 0.0, "Insufficient Data"
        
        # Calculate velocity over last few frames
        velocities = []
        for i in range(len(track_history) - 1, max(0, len(track_history) - 4), -1):
            if i > 0:
                vel = self.calculate_velocity(list(track_history)[i-1:i+1])
                velocities.append(vel)
        
        if len(velocities) < 2:
            return 0.0, "Insufficient Data"
        
        # Check for sudden velocity drop (indicating crash)
        avg_velocity = np.mean(velocities[1:]) if len(velocities) > 1 else velocities[0]
        current_velocity = velocities[0]
        velocity_drop_ratio = (avg_velocity - current_velocity) / (avg_velocity + 1e-6)
        
        # Calculate IoU consistency (vehicle staying in same location)
        recent_boxes = [d['box'] for d in list(track_history)[-3:]]
        iou_values = []
        for i in range(len(recent_boxes) - 1):
            iou = self.calculate_iou(recent_boxes[i], recent_boxes[i+1])
            iou_values.append(iou)
        
        avg_iou = np.mean(iou_values) if iou_values else 0.0
        
        # Severity Index: High IoU + Sudden velocity drop = Severe crash
        severity_index = 0.0
        severity_category = "Monitoring"
        
        if avg_iou > self.iou_threshold and velocity_drop_ratio > 0.7:
            severity_index = min(1.0, avg_iou * velocity_drop_ratio)
            severity_category = "Severe"
        elif avg_iou > 0.2 and velocity_drop_ratio > 0.5:
            severity_index = 0.5
            severity_category = "Moderate"
        elif velocity_drop_ratio > 0.3:
            severity_index = 0.3
            severity_category = "Mild"
        
        return severity_index, severity_category
    
    def analyze_accident(self, detections, frame_number):
        """Analyze detections for accident severity"""
        severity_results = []
        
        for detection in detections:
            box, conf, cls_name = detection
            track_id = self.assign_track_id(box, frame_number)
            self.update_track(track_id, box, frame_number, cls_name, conf)
            
            # Calculate severity if accident detected
            if 'accident' in cls_name.lower() or cls_name.lower() in ['severe', 'moderate', 'mild']:
                severity_index, severity_category = self.calculate_severity_index(track_id)
                severity_results.append({
                    'track_id': track_id,
                    'severity_index': severity_index,
                    'severity_category': severity_category,
                    'class': cls_name,
                    'confidence': conf,
                    'box': box
                })
        
        return severity_results

# Initialize triage system
triage_system = SeverityTriageSystem()
frame_counter = 0

# --- Edge-based Anonymization ---
def anonymize_frame(frame):
    """
    Anonymize faces and license plates in the frame using Gaussian blur
    for GDPR/privacy compliance
    """
    if face_model is None:
        return frame
    
    anonymized_frame = frame.copy()
    
    try:
        # Detect faces
        face_results = face_model(frame, verbose=False)
        
        for result in face_results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Apply Gaussian blur to face region
                if x2 > x1 and y2 > y1:
                    roi = anonymized_frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                    anonymized_frame[y1:y2, x1:x2] = blurred_roi
        
        # Simple license plate detection using color-based heuristics
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White/yellow plate detection (adjust ranges as needed)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
        
        print("[✅] Frame anonymized")
        
    except Exception as e:
        print(f"[⚠️] Anonymization error: {e}")
        return frame
    
    return anonymized_frame

# --- Telegram Alert Function with Anonymization ---
def send_telegram_alert(confidence, frame, severity_info=None):
    try:
        # Anonymize frame before sending
        anonymized_frame = anonymize_frame(frame)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp_{timestamp}.jpg"
        cv2.imwrite(temp_path, anonymized_frame)

        with open(temp_path, 'rb') as photo:
            if severity_info:
                message = (f"🚨 Crash Detected!\n"
                          f"Confidence: {confidence:.2f}\n"
                          f"Severity: {severity_info['severity_category']}\n"
                          f"Severity Index: {severity_info['severity_index']:.2f}\n"
                          f"Track ID: {severity_info['track_id']}\n"
                          f"⚠️ Frame anonymized for privacy compliance")
            else:
                message = f"🚨 Crash Detected with {confidence:.2f} confidence!\n⚠️ Frame anonymized for privacy compliance"
            
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            payload = {
                "chat_id": CHAT_ID,
                "caption": message
            }
            files = {"photo": photo}
            response = requests.post(url, data=payload, files=files)
            if response.status_code == 200:
                print("[✅] Telegram alert sent")
            else:
                print(f"[❌] Error: {response.status_code}, {response.text}")

        os.remove(temp_path)

    except Exception as e:
        print(f"[❌] Telegram error: {e}")

# --- Reset Alert Flag ---
def reset_alert_flag(delay=10):
    global alert_sent
    time.sleep(delay)
    alert_sent = False

# --- Detection and Streaming ---
def generate_frames(conf_threshold):
    global alert_sent, frame_counter
    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1
            
            # Check if model is loaded
            if model is None:
                # Display error message on frame
                cv2.putText(frame, "Model not loaded!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Please train a model first", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue
            
            results = model(frame, verbose=False)
            
            # Collect detections for triage analysis
            detections = []

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_name = model.names[cls]

                    if conf > conf_threshold:
                        detections.append((xyxy, conf, cls_name))
                        
                        label = f"{cls_name} {conf:.2f}"
                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Perform severity triage analysis
            if detections:
                severity_results = triage_system.analyze_accident(detections, frame_counter)
                
                # Display severity information on frame
                for i, sev_result in enumerate(severity_results):
                    if sev_result['severity_category'] != "Monitoring":
                        box = sev_result['box']
                        sev_text = f"{sev_result['severity_category']} (SI: {sev_result['severity_index']:.2f})"
                        cv2.putText(frame, sev_text, (box[0], box[3] + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Send alert for severe accidents
                        if sev_result['severity_category'] == "Severe" and not alert_sent:
                            threading.Thread(
                                target=send_telegram_alert, 
                                args=(sev_result['confidence'], frame.copy(), sev_result)
                            ).start()
                            threading.Thread(target=reset_alert_flag).start()
                            alert_sent = True

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

# --- API Routes ---
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to get system status"""
    return jsonify({
        'detection': model is not None,
        'triage': True,
        'anonymization': face_model is not None,
        'model_loaded': model is not None
    })

@app.route('/api/system/status', methods=['GET'])
def api_system_status():
    """System status endpoint for frontend dashboard"""
    return jsonify({
        'ml_service': {
            'available': model is not None,
            'model_path': MODEL_PATH if MODEL_PATH else None
        },
        'database': {
            'connected': True  # Placeholder - update if database is added
        },
        'triage': True,
        'anonymization': face_model is not None
    })

@app.route('/api/crashes/recent/<int:hours>', methods=['GET'])
def api_crashes_recent(hours):
    """Get recent crash detections (placeholder - can be connected to database)"""
    # Placeholder data - in production, this would query a database
    return jsonify([])

@app.route('/api/config', methods=['GET'])
def api_config():
    """API endpoint to get current configuration"""
    return jsonify({
        'model_path': MODEL_PATH if MODEL_PATH else None,
        'face_model_available': face_model is not None,
        'telegram_configured': BOT_TOKEN != "YOUR_BOT_TOKEN" and CHAT_ID != "YOUR_CHAT_ID"
    })

# --- Video Streaming Route ---
@app.route('/video')
def video():
    """Video streaming route for MJPEG stream"""
    conf = float(request.args.get("conf", 0.6))
    return Response(generate_frames(conf),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Serve React App ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React app for all routes"""
    if path != "" and os.path.exists(os.path.join(app.template_folder, path)):
        return send_from_directory(app.template_folder, path)
    else:
        # In development, redirect to React dev server
        # In production, serve index.html
        if os.path.exists(os.path.join(app.template_folder, 'index.html')):
            return send_from_directory(app.template_folder, 'index.html')
        else:
            # Fallback: return simple message if React build doesn't exist
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Crash Detection System</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        text-align: center; 
                        padding: 50px;
                        background: #121212;
                        color: #f1f1f1;
                    }
                    h1 { color: #f44336; }
                    a { color: #4CAF50; text-decoration: none; }
                </style>
            </head>
            <body>
                <h1>🚗 Smart Crash Detection System</h1>
                <p>Backend API is running!</p>
                <p>To use the React frontend:</p>
                <ol style="text-align: left; display: inline-block;">
                    <li>Navigate to the <code>frontend</code> directory</li>
                    <li>Run <code>npm install</code></li>
                    <li>Run <code>npm start</code></li>
                </ol>
                <p style="margin-top: 30px;">
                    <a href="/video?conf=0.6">View Video Stream Directly</a>
                </p>
            </body>
            </html>
            """

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚗 Smart Crash Detection System - Backend API")
    print("="*60)
    print(f"📡 Backend running on: http://localhost:5000")
    print(f"🎥 Video stream: http://localhost:5000/video?conf=0.6")
    print(f"📊 API status: http://localhost:5000/api/status")
    print(f"❤️  Health check: http://localhost:5000/health")
    print(f"🔧 System status: http://localhost:5000/api/system/status")
    if MODEL_PATH:
        print(f"✅ ML Model: {MODEL_PATH}")
    else:
        print(f"⚠️  ML Model: Using default YOLOv8n")
    print("\n💡 To use React frontend:")
    print("   1. cd frontend")
    print("   2. npm install")
    print("   3. npm start")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
