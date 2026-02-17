#!/usr/bin/env python3
import torch
import cv2
import pickle
import os
import subprocess
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import RPi.GPIO as GPIO

# ========================
# CONFIG & GPIO
# ========================
DB_FILE = "face_db.pkl" 
THRESHOLD = 0.6  # Adjust based on testing
BTN_BACK = 27
ABSENCE_FRAMES = 15
COOLDOWN = 10 

GPIO.setmode(GPIO.BCM)
GPIO.setup(BTN_BACK, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

device = torch.device('cpu') # Pi usually runs on CPU

# ========================
# MODELS & SPEECH
# ========================
def speak(text):
    print(f"🔊 {text}")
    try:
        # Using espeak to match your master controller style
        subprocess.Popen(['espeak', '-a', '200', '-s', '165', text], stderr=subprocess.DEVNULL)
    except:
        pass

mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========================
# LOAD DATABASE
# ========================
if not os.path.exists(DB_FILE):
    speak("Database not found")
    exit(1)

with open(DB_FILE, "rb") as f:
    face_db = pickle.load(f)

# ========================
# MAIN LOOP
# ========================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # Lower resolution for Pi speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    person_present = {}
    absence_counter = {}
    last_speak_time = {}

    speak("Face recognition active")

    try:
        while True:
            if GPIO.input(BTN_BACK) == GPIO.HIGH:
                break

            ret, frame = cap.read()
            if not ret: continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            boxes, probs = mtcnn.detect(pil_img)
            detected_this_frame = set()

            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob < 0.9: continue
                    
                    # Process face
                    face = pil_img.crop(box)
                    face_tensor = preprocess(face).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = resnet(face_tensor)

                    # Match
                    best_name = "Unknown"
                    max_sim = 0
                    for name, known_embs in face_db.items():
                        for k_emb in known_embs:
                            sim = F.cosine_similarity(emb, k_emb.to(device)).item()
                            if sim > max_sim:
                                max_sim = sim
                                best_name = name
                    
                    if max_sim > THRESHOLD:
                        detected_this_frame.add(best_name)

            # Logic for Announcements
            for name in face_db.keys():
                if name in detected_this_frame:
                    absence_counter[name] = 0
                    now = time.time()
                    
                    # Speak if new or cooldown passed
                    if not person_present.get(name, False):
                        if now - last_speak_time.get(name, 0) > COOLDOWN:
                            speak(f"Hello {name}")
                            last_speak_time[name] = now
                        person_present[name] = True
                else:
                    # Increment absence if not seen
                    if person_present.get(name, False):
                        absence_counter[name] = absence_counter.get(name, 0) + 1
                        if absence_counter[name] > ABSENCE_FRAMES:
                            person_present[name] = False
                            print(f"Left: {name}")

            time.sleep(0.01)

    finally:
        cap.release()
        speak("Stopping face recognition")

if __name__ == "__main__":
    main()