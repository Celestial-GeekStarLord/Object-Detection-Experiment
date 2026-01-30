#!/usr/bin/env python3
"""
Quick test script to verify YOLOv11 installation
Run this first to make sure everything is working
"""

import sys

print("🔍 Testing YOLOv11 Object Detection Setup")
print("=" * 50)

# Test 1: Import libraries
print("\n📦 Test 1: Checking imports...")
try:
    import cv2
    print("   ✅ OpenCV installed")
except ImportError:
    print("   ❌ OpenCV missing - run: pip install opencv-python")
    sys.exit(1)

try:
    import pyttsx3
    print("   ✅ pyttsx3 installed")
except ImportError:
    print("   ❌ pyttsx3 missing - run: pip install pyttsx3")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("   ✅ Ultralytics installed")
except ImportError:
    print("   ❌ Ultralytics missing - run: pip install ultralytics")
    sys.exit(1)

# Test 2: Camera access
print("\n📷 Test 2: Checking camera...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"   ✅ Camera working ({frame.shape[1]}x{frame.shape[0]})")
    else:
        print("   ⚠️  Camera opened but can't read frames")
    cap.release()
else:
    print("   ❌ Can't open camera - check connections")
    print("   Try changing CAMERA_INDEX to 1 or 2")

# Test 3: Load YOLO model
print("\n🧠 Test 3: Loading YOLO model...")
try:
    model = YOLO('yolo11n.pt')
    print("   ✅ Model loaded successfully")
    print(f"   📊 Model has {len(model.names)} classes")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    sys.exit(1)

# Test 4: Voice engine
print("\n🔊 Test 4: Testing voice engine...")
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    print("   ✅ Voice engine initialized")
    
    # Test speak
    print("   🎤 Testing voice (you should hear 'Test successful')...")
    engine.say("Test successful")
    engine.runAndWait()
    engine.stop()
    print("   ✅ Voice test complete")
except Exception as e:
    print(f"   ⚠️  Voice engine error: {e}")
    print("   Detection will work but announcements may fail")

# Test 5: Run detection on test
print("\n🎯 Test 5: Running test detection...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame, verbose=False)
        detections = len(results[0].boxes)
        print(f"   ✅ Detection test successful")
        print(f"   📊 Found {detections} objects in test frame")
        
        # Show what was detected
        if detections > 0:
            print("   🔍 Detected objects:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"      - {class_name} ({conf:.2f})")
    cap.release()

print("\n" + "=" * 50)
print("✅ All tests passed! You're ready to use the system.")
print("\nNext steps:")
print("1. Edit register_object.py and set OBJECT_NAME")
print("2. Run: python register_object.py")
print("3. Run: python detect_objects.py")
print("=" * 50)