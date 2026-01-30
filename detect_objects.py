import cv2
import time
import threading
import pyttsx3
import numpy as np
from ultralytics import YOLO

# =========================
# IMPROVED VOICE ENGINE
# =========================
class VoiceAnnouncer:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.current_announcement = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Worker thread that handles TTS announcements"""
        while self.running:
            announcement = None
            with self.lock:
                if self.current_announcement:
                    announcement = self.current_announcement
                    self.current_announcement = None
            
            if announcement:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.setProperty('volume', 1.0)
                    print(f"🔊 Speaking: {announcement}")
                    engine.say(announcement)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                    print(f"✅ Finished speaking: {announcement}")
                except Exception as e:
                    print(f"❌ TTS Error: {e}")
            
            time.sleep(0.1)
    
    def announce(self, object_name):
        """Queue an object to be announced"""
        with self.lock:
            self.current_announcement = f"{object_name} detected"
            print(f"📢 Queued: {object_name}")
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=3)

# =========================
# CONFIG
# =========================
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to detect object
COOLDOWN_SECONDS = 3  # Time between announcements for same object
ABSENCE_FRAMES = 20  # Frames to wait before marking as absent (~0.7 seconds)

# Objects to track (customize this list!)
# Leave empty [] to track ALL 80 COCO objects
# Or specify only what you want, e.g. ['person', 'laptop', 'cell phone', 'cup']
OBJECTS_TO_TRACK = []  # Empty = track everything

# Full COCO dataset (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# =========================
# SETUP TRACKING
# =========================
if not OBJECTS_TO_TRACK:
    # Track all COCO objects
    OBJECTS_TO_TRACK = COCO_CLASSES
    print("📦 Tracking ALL 80 COCO objects")
else:
    print(f"📦 Tracking {len(OBJECTS_TO_TRACK)} specific objects:")
    for obj in OBJECTS_TO_TRACK:
        print(f"   - {obj}")

# =========================
# LOAD YOLO MODEL
# =========================
print(f"\n🧠 Loading YOLOv11 model...")
model = YOLO('yolo11n.pt')  # Nano model (fast)
# Other options: yolo11s.pt, yolo11m.pt, yolo11l.pt (larger = more accurate)
print("✅ Model loaded successfully")

# =========================
# INITIALIZE VOICE
# =========================
announcer = VoiceAnnouncer()
time.sleep(0.5)

# =========================
# TRACKING STATE
# =========================
last_announcement_time = {}  # When object was last announced
object_present = {}  # Is object currently detected
absence_counter = {}  # Count frames of absence

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("\n🔍 Object detection started | Press Q to quit")
print(f"⚙️ Confidence: {CONFIDENCE_THRESHOLD} | Cooldown: {COOLDOWN_SECONDS}s")
print(f"📊 Tracking: {len(OBJECTS_TO_TRACK)} object types\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            time.sleep(0.1)
            continue
        
        # Ensure frame is numpy array
        frame = np.array(frame, dtype=np.uint8)
        frame_count += 1
        current_time = time.time()
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        detected_objects_this_frame = set()
        
        # =========================
        # OBJECT DETECTION
        # =========================
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = COCO_CLASSES[cls_id]
                confidence = float(box.conf[0])
                
                # Only process objects we're tracking
                if class_name not in OBJECTS_TO_TRACK:
                    continue
                
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Track detection
                detected_objects_this_frame.add(class_name)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        # =========================
        # PRESENCE TRACKING & ANNOUNCEMENTS
        # =========================
        for obj_name in OBJECTS_TO_TRACK:
            if obj_name in detected_objects_this_frame:
                # Object detected
                was_absent = not object_present.get(obj_name, False)
                object_present[obj_name] = True
                absence_counter[obj_name] = 0
                
                # Check if should announce
                last_announced = last_announcement_time.get(obj_name, 0)
                time_since_last = current_time - last_announced
                
                should_announce = (
                    last_announced == 0 or  # Never announced
                    (was_absent and time_since_last >= COOLDOWN_SECONDS)
                )
                
                if should_announce:
                    announcer.announce(obj_name)
                    last_announcement_time[obj_name] = current_time
                    status = "FIRST TIME" if last_announced == 0 else "RETURNED"
                    print(f"🎯 {obj_name} detected ({status}) - Announcing...")
            
            else:
                # Object NOT detected
                if object_present.get(obj_name, False):
                    absence_counter[obj_name] = absence_counter.get(obj_name, 0) + 1
                    
                    if absence_counter[obj_name] >= ABSENCE_FRAMES:
                        object_present[obj_name] = False
                        print(f"👋 {obj_name} left the frame")
        
        # =========================
        # DISPLAY INFO
        # =========================
        status_text = f"Frame: {frame_count}"
        if detected_objects_this_frame:
            status_text += f" | Present: {', '.join(detected_objects_this_frame)}"
        
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Show FPS
        if frame_count > 1:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        cv2.imshow("Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Quitting...")
            break

except KeyboardInterrupt:
    print("\n👋 Interrupted by user")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("🧹 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    announcer.stop()
    print("✅ Cleanup complete")