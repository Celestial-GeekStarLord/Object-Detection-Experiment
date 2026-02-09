import cv2
import time
import pyttsx3
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = 'yolo11n.onnx'
CONFIDENCE_THRESHOLD = 0.5
REPEAT_DELAY = 10  # seconds before repeating same object

# =========================
# TTS INITIALIZATION
# =========================
tts = pyttsx3.init(driverName="espeak")
tts.setProperty("rate", 155)
tts.setProperty("volume", 1.0)

# choose accent if you want (example: Indian English)
for v in tts.getProperty("voices"):
    if "en-in" in v.id.lower():
        tts.setProperty("voice", v.id)
        break

def speak(text):
    tts.say(text)
    tts.runAndWait()

# =========================
# MODEL INITIALIZATION
# =========================
print("🧠 Loading Headless Model...")
model = YOLO(MODEL_PATH, task="detect")
print("✅ Model loaded. Starting terminal stream...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit(1)

print("🚀 Tracking active. Press Ctrl+C to stop.\n")

# =========================
# MEMORY FOR SPOKEN OBJECTS
# =========================
last_spoken = {}  
# format:
# {
#   "person_LEFT": timestamp,
#   "bottle_CENTER": timestamp
# }

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False, imgsz=320, stream=True)

        found_objects = []
        now = time.time()

        for result in results:
            names = result.names

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls_id = int(box.cls[0])
                label = names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                frame_w = frame.shape[1]

                pos = "CENTER"
                if center_x < frame_w * 0.33:
                    pos = "LEFT"
                elif center_x > frame_w * 0.67:
                    pos = "RIGHT"

                key = f"{label}_{pos}"
                found_objects.append(f"{label} ({pos})")

                # =========================
                # SPEAK LOGIC
                # =========================
                last_time = last_spoken.get(key, 0)

                if key not in last_spoken or (now - last_time) >= REPEAT_DELAY:
                    speak_text = f"{label} on your {pos.lower()}"
                    speak(speak_text)
                    last_spoken[key] = now

        # =========================
        # TERMINAL OUTPUT
        # =========================
        if found_objects:
            print(f"🔎 Detected: {', '.join(found_objects)}", end="\r", flush=True)
        else:
            print("🌑 Scanning...             ", end="\r", flush=True)

except KeyboardInterrupt:
    print("\n\n👋 Stopping detection...")

finally:
    cap.release()
