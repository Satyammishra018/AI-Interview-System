import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import base64
import os
from ultralytics import YOLO
from queue import Queue
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
SMOOTHING = 0.8
EYE_DEADZONE = 2
EYE_THRESHOLD = 4

MAX_FACE_VIOLATIONS = 10
MAX_EYE_VIOLATIONS = 15

FACE_COOLDOWN = 1.0
EYE_COOLDOWN = 0.6

# YOLO Config
YOLO_MODEL_PATH = "yolov8n.pt"
TARGET_CLASSES = [0, 67] # 0: Person, 67: Cell Phone
YOLO_CONFIDENCE = 0.5
YOLO_SKIP_FRAMES = 5

# Mediapipe Tasks Config
MODEL_PATH = "face_landmarker.task"

# Indices (same as legacy)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1

# ---------------- ALERT SYSTEM ----------------
class AlertManager:
    def __init__(self, backend_url=None):
        self.backend_url = backend_url
        self.alert_queue = Queue()
        self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.worker_thread.start()

    def send_alert(self, violation_type, frame, data=None):
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        alert_payload = {
            "type": violation_type,
            "timestamp": time.time(),
            "data": data or {},
            "image": img_base64[:50] + "..."
        }
        self.alert_queue.put(alert_payload)

    def _process_alerts(self):
        while True:
            alert = self.alert_queue.get()
            print(f"[ALERT] {alert['type']} at {time.ctime(alert['timestamp'])}")
            self.alert_queue.task_done()

# ---------------- PROCTORING LOGIC ----------------
class ProctorSystem:
    def __init__(self):
        # 1. Initialize YOLO
        self.model = YOLO(YOLO_MODEL_PATH)
        
        # 2. Initialize Mediapipe Tasks
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.alert_manager = AlertManager()
        
        # State
        self.face_violations = 0
        self.eye_violations = 0
        self.session_risk_score = 0
        self.last_face_time = 0
        self.last_eye_time = 0
        self.prev_left_iris = None
        self.prev_right_iris = None
        self.prev_face_dir = "Face: CENTER"
        self.prev_eye_dir = "Eyes: CENTER"
        self.person_count = 0
        self.phone_detected = False
        self.frame_count = 0

    def get_head_direction(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        nose = landmarks[NOSE_TIP]
        left_face = landmarks[234]
        right_face = landmarks[454]

        offset = (nose.x - (left_face.x + right_face.x) / 2) * w
        if offset > w * 0.04: return "Face: RIGHT"
        elif offset < -w * 0.04: return "Face: LEFT"

        nose_y = nose.y * h
        center_y = (landmarks[152].y + landmarks[10].y) * h / 2
        v_offset = nose_y - center_y
        if v_offset > h * 0.03: return "Face: DOWN"
        elif v_offset < -h * 0.03: return "Face: UP"

        return "Face: CENTER"

    def get_eye_direction(self, iris_center, eye_points):
        eye_center = np.mean(eye_points, axis=0)
        dx, dy = iris_center - eye_center
        if abs(dx) < EYE_DEADZONE and abs(dy) < EYE_DEADZONE: return "Eyes: CENTER"
        if abs(dx) > abs(dy):
            if dx > EYE_THRESHOLD: return "Eyes: RIGHT"
            elif dx < -EYE_THRESHOLD: return "Eyes: LEFT"
        else:
            if dy > EYE_THRESHOLD: return "Eyes: DOWN"
            elif dy < -EYE_THRESHOLD: return "Eyes: UP"
        return "Eyes: CENTER"

    def process_frame(self, frame):
        self.frame_count += 1
        h, w = frame.shape[:2]
        now = time.time()

        # 1. YOLO DETECTION
        if self.frame_count % YOLO_SKIP_FRAMES == 0:
            yolo_results = self.model(frame, imgsz=320, conf=YOLO_CONFIDENCE, classes=TARGET_CLASSES, verbose=False)
            self.person_count = 0
            self.phone_detected = False
            for r in yolo_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0: self.person_count += 1
                    elif cls == 67: self.phone_detected = True
            
            if self.person_count > 1:
                self.alert_manager.send_alert("MULTIPLE_PERSONS", frame, {"count": self.person_count})
                self.session_risk_score += 2
            if self.phone_detected:
                self.alert_manager.send_alert("PHONE_DETECTED", frame)
                self.session_risk_score += 5
            if self.person_count == 0:
                self.session_risk_score += 1

        # 2. FACE LANDMARK DETECTION (Tasks API)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = self.landmarker.detect(mp_image)

        current_face_dir = "Face: CENTER"
        current_eye_dir = "Eyes: CENTER"

        if not detection_result.face_landmarks:
            if now - self.last_face_time > FACE_COOLDOWN:
                self.face_violations += 1
                self.last_face_time = now
                self.alert_manager.send_alert("NO_FACE_DETECTED", frame)
        else:
            face_landmarks = detection_result.face_landmarks[0]
            
            def pts(idxs):
                return np.array([[int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)] for i in idxs])

            landmarks_px = {
                "left_eye": pts(LEFT_EYE), "right_eye": pts(RIGHT_EYE),
                "left_iris": pts(LEFT_IRIS), "right_iris": pts(RIGHT_IRIS)
            }

            current_face_dir = self.get_head_direction(face_landmarks, frame.shape)
            
            raw_left = np.mean(landmarks_px["left_iris"], axis=0)
            raw_right = np.mean(landmarks_px["right_iris"], axis=0)

            if self.prev_left_iris is None:
                self.prev_left_iris, self.prev_right_iris = raw_left, raw_right

            left_iris = SMOOTHING * self.prev_left_iris + (1 - SMOOTHING) * raw_left
            right_iris = SMOOTHING * self.prev_right_iris + (1 - SMOOTHING) * raw_right
            self.prev_left_iris, self.prev_right_iris = left_iris, right_iris

            l_eye_dir = self.get_eye_direction(left_iris, landmarks_px["left_eye"])
            r_eye_dir = self.get_eye_direction(right_iris, landmarks_px["right_eye"])
            current_eye_dir = l_eye_dir if l_eye_dir == r_eye_dir else "Eyes: CENTER"

            # Violation Logic
            if current_face_dir != "Face: CENTER" and current_face_dir != self.prev_face_dir:
                if now - self.last_face_time > FACE_COOLDOWN:
                    self.face_violations += 1
                    self.last_face_time = now
                    self.alert_manager.send_alert("HEAD_DIRECTION_VIOLATION", frame, {"dir": current_face_dir})

            if current_eye_dir != "Eyes: CENTER" and current_eye_dir != self.prev_eye_dir:
                if now - self.last_eye_time > EYE_COOLDOWN:
                    self.eye_violations += 1
                    self.last_eye_time = now
                    self.alert_manager.send_alert("EYE_GAZE_VIOLATION", frame, {"dir": current_eye_dir})

            self.prev_face_dir = current_face_dir
            self.prev_eye_dir = current_eye_dir

        return self.draw_hud(frame, current_face_dir, current_eye_dir)

    def draw_hud(self, frame, face_dir, eye_dir):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        y = 30
        info = [
            (face_dir, (0, 255, 255)), (eye_dir, (0, 255, 0)),
            (f"Persons: {self.person_count}", (255, 255, 255)),
            (f"Phone: {'YES' if self.phone_detected else 'NO'}", (0, 0, 255) if self.phone_detected else (0, 255, 0)),
            (f"Risk Score: {int(self.session_risk_score)}", (0, 165, 255)),
            (f"Violations (H/E): {self.face_violations}/{self.eye_violations}", (200, 200, 200))
        ]
        for text, color in info:
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 28
        return frame

def run_proctor():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please download it first.")
        return

    print("[INFO] Initializing AI Models (YOLOv8 & MediaPipe)... This may take a few seconds.")
    start_time = time.time()
    
    # 1. Faster Initialization of AI models
    proctor = ProctorSystem()
    
    # 2. Open camera AFTER models are ready, so the feed is instant
    print("[INFO] Models loaded. Starting Camera...")
    cap = cv2.VideoCapture(0)
    
    # 3. Quick camera warm-up
    if cap.isOpened():
        for _ in range(3): 
            cap.read() 
    
    elapsed = time.time() - start_time
    print(f"--- Integrated Proctoring System Ready (Started in {elapsed:.1f}s) ---")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame = proctor.process_frame(frame)
        cv2.imshow("AI Proctoring System - Integrated", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("--- System Stopped Successfully ---")

if __name__ == "__main__":
    run_proctor()
