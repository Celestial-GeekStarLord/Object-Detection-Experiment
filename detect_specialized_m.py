import cv2
import time
import threading
import pyttsx3
import numpy as np
from ultralytics import YOLO

# --- 1. SPECIALIZED CONFIGURATION ---
# These MUST match the order in your data.yaml exactly
NEURALENS_CLASSES = [
    'bag', 'bin', 'bottle', 'cctv_camera', 'chair', 'copy', 'curtain', 'desk', 'door', 'glass', 
    'jug', 'light', 'pen', 'person', 'plants', 'poster', 'smartboard', 'stairs', 'switch', 'watch'
]

CONFIDENCE_THRESHOLD = 0.6  # Adjusted for specialized indoor environments
REANNOUNCE_TIMEOUT = 8      # Slightly faster for navigational objects
ABSENCE_FRAMES = 15         # Sensitivity for objects leaving the frame

# --- 2. IMPROVED VOICE ENGINE ---
class VoiceAnnouncer:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.current_announcement = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while self.running:
            announcement = None
            with self.lock:
                if self.current_announcement:
                    announcement = self.current_announcement
                    self.current_announcement = None
            
            if announcement:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 170)  # Slightly faster for real-time navigation
                    engine.setProperty('volume', 1.0)
                    engine.say(announcement)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"❌ TTS Error: {e}")
            time.sleep(0.1)
    
    def announce(self, text):
        with self.lock:
            self.current_announcement = text

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)

# --- 3. SPATIAL & PROXIMITY UTILS ---
def get_spatial_info(center_x, box_area, frame_width, frame_height):
    """Returns (position_string, proximity_string)"""
    # Horizontal Position
    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.67
    
    if center_x < left_boundary:
        pos = "on your left"
    elif center_x > right_boundary:
        pos = "on your right"
    else:
        pos = "ahead"
        
    # Proximity Estimation (based on area percentage)
    frame_area = frame_width * frame_height
    occupancy = (box_area / frame_area) * 100
    
    if occupancy > 25: # Object takes up 1/4 of the screen
        prox = "very close"
    elif occupancy > 10:
        prox = "near"
    else:
        prox = "" # Normal distance, no extra alert needed
        
    return pos, prox

# --- 4. INITIALIZATION ---
print(f"🧠 Loading Neuralens V2 Specialized Model (20 Classes)...")
model = YOLO('best.pt') 
announcer = VoiceAnnouncer()

# Camera Setup
IP_CAMERA_URL = "http://192.168.18.4:8080/video"
cap = cv2.VideoCapture(IP_CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Tracking State
last_announced_object = None
last_announcement_time = {}
object_present = {name: False for name in NEURALENS_CLASSES}
absence_counter = {name: 0 for name in NEURALENS_CLASSES}

print("✅ System Ready. Press 'Q' to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        
        # Inference
        results = model(frame, verbose=False)
        detected_this_frame = {}

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < CONFIDENCE_THRESHOLD: continue
                
                class_name = NEURALENS_CLASSES[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate Metadata
                center_x = (x1 + x2) // 2
                box_area = (x2 - x1) * (y2 - y1)
                pos_str, prox_str = get_spatial_info(center_x, box_area, frame_width, frame_height)
                
                if class_name not in detected_this_frame:
                    detected_this_frame[class_name] = (pos_str, prox_str)

                # Visual Feedback
                color = (0, 255, 0) if "close" not in prox_str else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {prox_str}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Announcement Logic
        for obj_name, (pos, prox) in detected_this_frame.items():
            object_present[obj_name] = True
            absence_counter[obj_name] = 0
            
            elapsed = current_time - last_announcement_time.get(obj_name, 0)
            
            if last_announced_object != obj_name or elapsed > REANNOUNCE_TIMEOUT:
                # Construct sentence: "Person, near, ahead"
                alert = f"{obj_name} {prox} {pos}".strip()
                announcer.announce(alert)
                print(f"🔊 {alert}")
                last_announced_object = obj_name
                last_announcement_time[obj_name] = current_time

        # Handle Absence
        for name in NEURALENS_CLASSES:
            if name not in detected_this_frame and object_present[name]:
                absence_counter[name] += 1
                if absence_counter[name] > ABSENCE_FRAMES:
                    object_present[name] = False
                    if last_announced_object == name:
                        last_announced_object = None

        cv2.imshow("Neuralens V2 - Specialized", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    announcer.stop()