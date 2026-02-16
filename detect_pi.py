#!/usr/bin/env python3
import cv2
import time
import signal
import sys
import subprocess
import RPi.GPIO as GPIO
from ultralytics import YOLO


# GPIO SETUP

BTN_BACK = 27
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BTN_BACK, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


# CONFIGURATION

MODEL_PATH = 'yolo11n.onnx'
CONFIDENCE_THRESHOLD = 0.5
REPEAT_DELAY = 10
CAMERA_RESOLUTION = 320

# Global variables
speech_process = None
running = True


# SIGNAL HANDLERS

def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# SPEECH FUNCTIONS

def speak(text):
    global speech_process
    
    print(f"[SPEAK] {text}")
    
    # Stop previous speech
    if speech_process is not None:
        try:
            speech_process.kill()
        except:
            pass
    
    try:
        speech_process = subprocess.Popen(
            ['espeak', '-a', '200', '-s', '165', text, '--stdout'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        
        subprocess.Popen(
            ['aplay', '-D', 'plughw:1,0'],
            stdin=speech_process.stdout,
            stderr=subprocess.DEVNULL
        )
        
        speech_process.stdout.close()
        
    except Exception as e:
        print(f"Speech error: {e}")

def check_back_button():
    global running
    if GPIO.input(BTN_BACK) == GPIO.HIGH:
        running = False
        return True
    return False


# POSITION HELPERS

def get_position(x_center, frame_width):
    if x_center < frame_width * 0.33:
        return "LEFT"
    elif x_center > frame_width * 0.67:
        return "RIGHT"
    else:
        return "CENTER"

def get_distance_estimate(box_area, frame_area):
    ratio = box_area / frame_area
    
    if ratio > 0.25:
        return "CLOSE"
    elif ratio > 0.10:
        return "NEAR"
    else:
        return "FAR"


# MAIN DETECTION LOOP

def main():
    global running
    
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH, task="detect")
        print("Model loaded")
    except Exception as e:
        print(f"Model error: {e}")
        speak("Failed to load model")
        sys.exit(1)
    
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Camera error")
        speak("Camera not available")
        sys.exit(1)
    
    # Warm up camera
    for _ in range(5):
        cap.read()
    
    speak("Detection started")
    print("Detection active. Press BACK button to stop.\n")
    
    last_spoken = {}
    
    try:
        while running:
            # Check for BACK button
            if check_back_button():
                break
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Get frame dimensions
            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w
            
            # Run detection
            results = model(frame, verbose=False, imgsz=CAMERA_RESOLUTION, stream=True)
            
            found_objects = []
            now = time.time()
            
            for result in results:
                names = result.names
                
                for box in result.boxes:
                    # Filter by confidence
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Get object info
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    
                    # Calculate position
                    position = get_position(center_x, frame_w)
                    
                    # Calculate distance
                    box_area = (x2 - x1) * (y2 - y1)
                    distance = get_distance_estimate(box_area, frame_area)
                    
                    # Create unique key
                    key = f"{label}_{position}"
                    
                    # Add to found objects
                    found_objects.append(f"{label} ({position})")
                    
                    # Speech logic
                    last_time = last_spoken.get(key, 0)
                    
                    if key not in last_spoken or (now - last_time) >= REPEAT_DELAY:
                        # Construct announcement
                        announcement = f"{label} on your {position.lower()}"
                        
                        # Add distance for important objects
                        if label in ["person", "car", "truck", "bicycle", "motorcycle"]:
                            announcement += f", {distance.lower()}"
                        
                        speak(announcement)
                        last_spoken[key] = now
            
            # Clean up old entries
            cutoff_time = now - (REPEAT_DELAY * 2)
            last_spoken = {k: v for k, v in last_spoken.items() if v > cutoff_time}
            
            # Terminal output
            if found_objects:
                print(f"Detected: {', '.join(found_objects)}", end="\r", flush=True)
            else:
                print("Scanning...             ", end="\r", flush=True)
            
            time.sleep(0.1)
    
    except Exception as e:
        print(f"\nError: {e}")
    
    finally:
        print("\nCleaning up...")
        cap.release()
        GPIO.cleanup()
        print("Detection stopped")

if __name__ == "__main__":
    main()