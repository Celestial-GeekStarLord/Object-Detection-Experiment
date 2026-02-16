import cv2
import time
import threading
import pyttsx3
import numpy as np
import easyocr
import sys

# --- VOICE ENGINE ---
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
                    engine.setProperty('rate', 150) # Slightly slower for better clarity
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

# --- OCR ENGINE ---
class OCRProcessor:
    def __init__(self):
        print("⏳ Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False) 
        print("✅ OCR Engine Ready.")

    def process_frame(self, frame):
        # Resize to 50% for faster processing
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return self.reader.readtext(gray, detail=0)

# --- CONFIGURATION ---
IP_CAMERA_URL = "http://192.168.18.4:8080/video"
SCAN_INTERVAL = 5.0  # <--- Set to 5 seconds
MIN_CHAR_LENGTH = 3 

def main():
    print("\n--- 📖 STARTING OCR SCANNER (5s INTERVAL) ---")
    
    announcer = VoiceAnnouncer()
    ocr = OCRProcessor()
    
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ ERROR: Camera not found.")
        return

    print("✅ Connected. Ready to read.")
    
    last_scan_time = time.time()
    last_spoken_text = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            current_time = time.time()
            elapsed = current_time - last_scan_time

            # Console visual for the countdown
            remaining = max(0, SCAN_INTERVAL - elapsed)
            print(f"⏳ Next scan in: {remaining:.1f}s    ", end='\r')

            if elapsed >= SCAN_INTERVAL:
                print(f"\n🔍 Scanning now...")
                
                results = ocr.process_frame(frame)
                valid_words = [word for word in results if len(word) >= MIN_CHAR_LENGTH]
                full_text = " ".join(valid_words).strip()

                if full_text:
                    if full_text != last_spoken_text:
                        print(f"📖 Detected: {full_text}")
                        announcer.announce(full_text)
                        last_spoken_text = full_text
                    else:
                        print("🔁 Text same as last scan, skipping speech.")
                else:
                    print("📭 No text found.")

                last_scan_time = time.time()

    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        cap.release()
        announcer.stop()
        print("✅ System Offline.")

if __name__ == "__main__":
    main()