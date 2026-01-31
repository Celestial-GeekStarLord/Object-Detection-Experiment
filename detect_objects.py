import cv2
import time
import threading
import pyttsx3
import numpy as np
from ultralytics import YOLO


# IMPROVED VOICE ENGINE

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
    
    def announce(self, text):
        """Queue text to be announced"""
        with self.lock:
            self.current_announcement = text
            print(f"📢 Queued: {text}")
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


# SPATIAL POSITION HELPER

def get_position(center_x, frame_width):
    """
    Determine if object is on left, center, or right of frame
    Returns: 'left', 'center', or 'right'
    """
    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.67
    
    if center_x < left_boundary:
        return "left"
    elif center_x > right_boundary:
        return "right"
    else:
        return "center"


# CONFIG

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to detect object
REANNOUNCE_TIMEOUT = 10  # Seconds before re-announcing same object
ABSENCE_FRAMES = 20  # Frames to wait before marking as absent (~0.7 seconds)

# Objects to track (customize this list!)
OBJECTS_TO_TRACK = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                'traffic light', 'bench', 'cat', 'dog', 'backpack', 'sports ball',
                'bottle', 'cup', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'orange', 'carrot', 'chair', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'clock']

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
print("✅ Model loaded successfully")

# =========================
# INITIALIZE VOICE
# =========================
announcer = VoiceAnnouncer()
time.sleep(0.5)

# =========================
# TRACKING STATE
# =========================
last_announced_object = None  # The last object that was announced
last_announcement_time = {}  # When each object was last announced
object_present = {}  # Is object currently detected
absence_counter = {}  # Count frames of absence
object_positions = {}  # Store positions of objects (for spatial info)

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
print(f"⚙️ Confidence: {CONFIDENCE_THRESHOLD} | Re-announce timeout: {REANNOUNCE_TIMEOUT}s")
print(f"📊 Tracking: {len(OBJECTS_TO_TRACK)} object types\n")

frame_count = 0
frame_width = 640
frame_height = 480

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
        
        # Get actual frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        detected_objects_this_frame = {}  # {object_name: (center_x, center_y, position)}
        
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
                
                # Calculate center and position
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                position = get_position(center_x, frame_width)
                
                # Store the first (highest confidence) detection of each object type
                if class_name not in detected_objects_this_frame:
                    detected_objects_this_frame[class_name] = (center_x, center_y, position)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw position label
                label = f"{class_name} ({confidence:.2f}) - {position.upper()}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # =========================
        # SMART ANNOUNCEMENT LOGIC
        # =========================
        # Rule 1: Announce if it's a NEW object (different from last announced)
        # Rule 2: Announce if same object but 10+ seconds have passed
        # Rule 3: When object switches, reset and announce the new one
        
        for obj_name, (center_x, center_y, position) in detected_objects_this_frame.items():
            # Mark as present
            was_absent = not object_present.get(obj_name, False)
            object_present[obj_name] = True
            absence_counter[obj_name] = 0
            object_positions[obj_name] = position
            
            # Check if we should announce
            should_announce = False
            reason = ""
            
            # Case 1: This is a different object from what we last announced
            if last_announced_object != obj_name:
                should_announce = True
                reason = "NEW OBJECT"
            
            # Case 2: Same object, but 10+ seconds have passed
            elif last_announced_object == obj_name:
                last_time = last_announcement_time.get(obj_name, 0)
                time_elapsed = current_time - last_time
                if time_elapsed >= REANNOUNCE_TIMEOUT:
                    should_announce = True
                    reason = "TIMEOUT REACHED"
            
            if should_announce:
                # Build announcement with spatial info
                announcement = f"{obj_name}"
                
                # Add position info if there are multiple objects
                if len(detected_objects_this_frame) > 1:
                    announcement = f"{obj_name} on {position}"
                
                announcer.announce(announcement)
                last_announced_object = obj_name
                last_announcement_time[obj_name] = current_time
                print(f"🎯 Announcing: {announcement} ({reason})")
        
        # =========================
        # HANDLE ABSENT OBJECTS
        # =========================
        for obj_name in OBJECTS_TO_TRACK:
            if obj_name not in detected_objects_this_frame:
                # Object NOT detected
                if object_present.get(obj_name, False):
                    absence_counter[obj_name] = absence_counter.get(obj_name, 0) + 1
                    
                    if absence_counter[obj_name] >= ABSENCE_FRAMES:
                        object_present[obj_name] = False
                        print(f"👋 {obj_name} left the frame")
                        
                        # If this was the last announced object and it left, reset
                        if last_announced_object == obj_name:
                            print(f"   ↳ Resetting last announced (was {obj_name})")
        
        # =========================
        # DISPLAY INFO
        # =========================
        # Draw position zones on frame
        left_line = int(frame_width * 0.33)
        right_line = int(frame_width * 0.67)
        cv2.line(frame, (left_line, 0), (left_line, frame_height), (100, 100, 100), 1)
        cv2.line(frame, (right_line, 0), (right_line, frame_height), (100, 100, 100), 1)
        cv2.putText(frame, "LEFT", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "CENTER", (left_line + 10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "RIGHT", (right_line + 10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Status text
        status_text = f"Frame: {frame_count}"
        if detected_objects_this_frame:
            obj_list = [f"{obj}({pos[:1].upper()})" for obj, (_, _, pos) in detected_objects_this_frame.items()]
            status_text += f" | Present: {', '.join(obj_list)}"
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Last announced info
        if last_announced_object:
            time_since = current_time - last_announcement_time.get(last_announced_object, current_time)
            announced_text = f"Last: {last_announced_object} ({time_since:.1f}s ago)"
            cv2.putText(frame, announced_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 2)
        
        # Show FPS
        if frame_count > 1:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (frame_width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        cv2.imshow("Object Detection with Spatial Awareness", frame)
        
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