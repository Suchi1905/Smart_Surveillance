from flask import Flask, Response, jsonify, request
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

app = Flask(__name__)
CORS(app)

# --- Config ---
MODEL_PATH = "../weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"[✅] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[⚠️] Warning: Could not load model from {MODEL_PATH}")
    print(f"[ℹ️] Error: {e}")
    model = None

# Load face detection model
try:
    face_model = YOLO("yolov8n-face.pt")
    print("[✅] Face detection model loaded")
except:
    face_model = None
    print("[⚠️] Face detection model not found - anonymization disabled")

alert_sent = False

# Node.js backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001")

# --- Severity Triage System (same as before) ---
class SeverityTriageSystem:
    def __init__(self, buffer_size=10, iou_threshold=0.3):
        self.vehicle_tracks = defaultdict(lambda: deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.next_track_id = 0
        
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_box_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def calculate_velocity(self, track_history):
        if len(track_history) < 2:
            return 0.0
        
        recent = track_history[-1]
        previous = track_history[-2]
        
        center_recent = self.get_box_center(recent['box'])
        center_previous = self.get_box_center(previous['box'])
        
        displacement = np.sqrt(
            (center_recent[0] - center_previous[0])**2 + 
            (center_recent[1] - center_previous[1])**2
        )
        
        return displacement
    
    def assign_track_id(self, current_box, frame_number):
        best_iou = 0
        best_track_id = None
        
        for track_id, history in self.vehicle_tracks.items():
            if len(history) > 0:
                last_detection = history[-1]
                if frame_number - last_detection['frame'] < 5:
                    iou = self.calculate_iou(current_box, last_detection['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
        
        if best_iou > 0.3:
            return best_track_id
        
        new_id = self.next_track_id
        self.next_track_id += 1
        return new_id
    
    def update_track(self, track_id, box, frame_number, class_name, confidence):
        self.vehicle_tracks[track_id].append({
            'box': box,
            'frame': frame_number,
            'class': class_name,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def calculate_severity_index(self, track_id):
        track_history = self.vehicle_tracks[track_id]
        
        if len(track_history) < 3:
            return 0.0, "Insufficient Data"
        
        velocities = []
        for i in range(len(track_history) - 1, max(0, len(track_history) - 4), -1):
            if i > 0:
                vel = self.calculate_velocity(list(track_history)[i-1:i+1])
                velocities.append(vel)
        
        if len(velocities) < 2:
            return 0.0, "Insufficient Data"
        
        avg_velocity = np.mean(velocities[1:]) if len(velocities) > 1 else velocities[0]
        current_velocity = velocities[0]
        velocity_drop_ratio = (avg_velocity - current_velocity) / (avg_velocity + 1e-6)
        
        recent_boxes = [d['box'] for d in list(track_history)[-3:]]
        iou_values = []
        for i in range(len(recent_boxes) - 1):
            iou = self.calculate_iou(recent_boxes[i], recent_boxes[i+1])
            iou_values.append(iou)
        
        avg_iou = np.mean(iou_values) if iou_values else 0.0
        
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
        severity_results = []
        
        for detection in detections:
            box, conf, cls_name = detection
            track_id = self.assign_track_id(box, frame_number)
            self.update_track(track_id, box, frame_number, cls_name, conf)
            
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

triage_system = SeverityTriageSystem()
frame_counter = 0

# --- Anonymization (same as before) ---
def anonymize_frame(frame):
    if face_model is None:
        return frame
    
    anonymized_frame = frame.copy()
    
    try:
        face_results = face_model(frame, verbose=False)
        
        for result in face_results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    roi = anonymized_frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                    anonymized_frame[y1:y2, x1:x2] = blurred_roi
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 2.0 < aspect_ratio < 5.0:
                    roi = anonymized_frame[y:y+h, x:x+w]
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    anonymized_frame[y:y+h, x:x+w] = blurred_roi
        
    except Exception as e:
        print(f"[⚠️] Anonymization error: {e}")
        return frame
    
    return anonymized_frame

# --- Save crash to database via Node.js API ---
def save_crash_to_database(severity_info, frame_number, confidence):
    try:
        crash_data = {
            'severity': severity_info['severity_category'],
            'severity_index': float(severity_info['severity_index']),
            'confidence': float(confidence),
            'track_id': severity_info['track_id'],
            'frame_number': frame_number,
            'location': {
                'x': int(severity_info['box'][0]),
                'y': int(severity_info['box'][1]),
                'width': int(severity_info['box'][2] - severity_info['box'][0]),
                'height': int(severity_info['box'][3] - severity_info['box'][1])
            },
            'anonymized': True
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/crashes",
            json=crash_data,
            timeout=2
        )
        
        if response.status_code == 201:
            print(f"[✅] Crash event saved to database: {response.json().get('id')}")
        else:
            print(f"[⚠️] Failed to save crash event: {response.status_code}")
            
    except Exception as e:
        print(f"[❌] Error saving to database: {e}")

# --- Video Streaming ---
def generate_frames(conf_threshold):
    global alert_sent, frame_counter
    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1
            
            if model is None:
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
            
            if detections:
                severity_results = triage_system.analyze_accident(detections, frame_counter)
                
                for sev_result in severity_results:
                    if sev_result['severity_category'] != "Monitoring":
                        box = sev_result['box']
                        sev_text = f"{sev_result['severity_category']} (SI: {sev_result['severity_index']:.2f})"
                        cv2.putText(frame, sev_text, (box[0], box[3] + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Save to database and send alert for severe accidents
                        if sev_result['severity_category'] == "Severe" and not alert_sent:
                            # Save to database in background
                            threading.Thread(
                                target=save_crash_to_database,
                                args=(sev_result, frame_counter, sev_result['confidence'])
                            ).start()
                            
                            threading.Thread(target=reset_alert_flag).start()
                            alert_sent = True

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

def reset_alert_flag(delay=10):
    global alert_sent
    time.sleep(delay)
    alert_sent = False

# --- API Routes ---
@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'detection': model is not None,
        'triage': True,
        'anonymization': face_model is not None,
        'model_loaded': model is not None
    })

@app.route('/api/config', methods=['GET'])
def api_config():
    return jsonify({
        'model_path': MODEL_PATH,
        'face_model_available': face_model is not None
    })

@app.route('/video')
def video():
    conf = float(request.args.get("conf", 0.6))
    return Response(generate_frames(conf),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 ML Service - Python Video Processing")
    print("="*60)
    print(f"📡 ML Service running on: http://localhost:5000")
    print(f"🎥 Video stream: http://localhost:5000/video?conf=0.6")
    print(f"🔗 Backend URL: {BACKEND_URL}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)


