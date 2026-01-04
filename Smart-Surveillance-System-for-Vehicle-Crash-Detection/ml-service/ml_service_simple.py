"""
Simplified ML Service - Works without full dependencies
This version allows the dashboard to load even if cv2/ultralytics aren't installed
"""
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Try to import ML dependencies
try:
    import cv2
    from ultralytics import YOLO
    ML_AVAILABLE = True
except ImportError as e:
    print(f"[⚠️] ML dependencies not available: {e}")
    print("[ℹ️] Dashboard will work but video processing is disabled")
    ML_AVAILABLE = False

MODEL_PATH = "../weights/best.pt"
model = None
face_model = None

if ML_AVAILABLE:
    try:
        model = YOLO(MODEL_PATH)
        print(f"[✅] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[⚠️] Could not load model: {e}")
        model = None
    
    try:
        face_model = YOLO("yolov8n-face.pt")
        print("[✅] Face detection model loaded")
    except:
        face_model = None
        print("[⚠️] Face detection model not found")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001")

def generate_placeholder_frames():
    """Generate placeholder frames when ML is not available"""
    import numpy as np
    while True:
        # Create a simple colored frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 41, 59)  # Dark blue-gray
        
        # Add text
        import cv2
        cv2.putText(frame, "ML Service Starting...", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Install dependencies to enable", (50, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames(conf_threshold):
    """Generate video frames"""
    if not ML_AVAILABLE:
        yield from generate_placeholder_frames()
        return
    
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if model is None:
                cv2.putText(frame, "Model not loaded!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Please train a model first", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                results = model(frame, verbose=False)
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > conf_threshold:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            label = f"{model.names[int(box.cls[0])]} {conf:.2f}"
                            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if ML_AVAILABLE:
            cap.release()

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'detection': ML_AVAILABLE and model is not None,
        'triage': ML_AVAILABLE,
        'anonymization': ML_AVAILABLE and face_model is not None,
        'model_loaded': model is not None,
        'ml_available': ML_AVAILABLE
    })

@app.route('/api/config', methods=['GET'])
def api_config():
    return jsonify({
        'model_path': MODEL_PATH,
        'face_model_available': face_model is not None,
        'ml_available': ML_AVAILABLE
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
    if not ML_AVAILABLE:
        print("⚠️  ML dependencies not installed")
        print("ℹ️  Service will run in placeholder mode")
        print("ℹ️  See INSTALL_PYTHON.md for installation instructions")
    print(f"📡 ML Service running on: http://localhost:5000")
    print(f"🎥 Video stream: http://localhost:5000/video?conf=0.6")
    print(f"🔗 Backend URL: {BACKEND_URL}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

