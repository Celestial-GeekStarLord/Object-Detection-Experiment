import cv2
import pickle
import os
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
OBJECT_NAME = "laptop"  # Change this to register different objects
DB_FILE = "object_db.pkl"
CAMERA_INDEX = 0

# COCO dataset class names (80 classes)
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

print("📦 Available objects to register:")
print("=" * 60)
for i, obj in enumerate(COCO_CLASSES, 1):
    print(f"{i:2d}. {obj:20s}", end="")
    if i % 3 == 0:
        print()
print("\n" + "=" * 60)

# Validate object name
if OBJECT_NAME not in COCO_CLASSES:
    print(f"\n⚠️  '{OBJECT_NAME}' is not in COCO dataset!")
    print(f"Please choose from the list above and update OBJECT_NAME")
    exit(1)

# =========================
# LOAD YOLO MODEL
# =========================
print(f"\n🧠 Loading YOLOv11 model...")
model = YOLO('yolo11n.pt')  # Nano model (fastest, smallest)
# Other options: yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large)
print("✅ Model loaded successfully")

# =========================
# LOAD OR CREATE DATABASE
# =========================
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        object_db = pickle.load(f)
    print(f"📦 Loaded existing database with {len(object_db)} registered objects")
else:
    object_db = {}
    print("📦 Created new database")

if OBJECT_NAME not in object_db:
    object_db[OBJECT_NAME] = {
        'count': 0,
        'bboxes': []  # Store bounding box samples
    }

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print(f"\n📸 Point camera at your {OBJECT_NAME}")
print("👉 Press S to save current detection")
print("👉 Press Q to quit")
print(f"🎯 Currently registered: {object_db[OBJECT_NAME]['count']} samples\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Get detections
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = COCO_CLASSES[cls_id]
                confidence = float(box.conf[0])
                
                if class_name == OBJECT_NAME and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_objects.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{OBJECT_NAME} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        
        # Display info
        status_text = f"Frame: {frame_count} | Saved: {object_db[OBJECT_NAME]['count']}"
        if detected_objects:
            status_text += f" | Detected: {len(detected_objects)} {OBJECT_NAME}(s)"
        
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            "Press S to Save | Q to Quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        cv2.imshow("Register Object", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # SAVE DETECTION
        if key == ord('s') and detected_objects:
            for obj in detected_objects:
                object_db[OBJECT_NAME]['bboxes'].append(obj['bbox'])
                object_db[OBJECT_NAME]['count'] += 1
            
            with open(DB_FILE, "wb") as f:
                pickle.dump(object_db, f)
            
            print(f"✅ Saved {len(detected_objects)} {OBJECT_NAME}(s)")
            print(f"📦 Total registered: {object_db[OBJECT_NAME]['count']} samples")
        
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n👋 Interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Registration complete")
    print(f"📦 Database saved to {DB_FILE}")