"""
phone_detection.py - Detects presence of phones (cell phones or mobiles) in video frames.

This module uses a pretrained YOLOv5 model (e.g., yolov5su.pt) to detect objects.
It processes the video frame-by-frame (skipping frames for performance) and
flags frames where a phone is detected with a confidence above a threshold.

This helps identify potential cheating via unauthorized devices.
"""

import cv2
import os
from ultralytics import YOLO  

# --- Configuration Constants ---
YOLO_MODEL_PATH = "yolov5su.pt"    # Path to the YOLO model weights file
FRAME_PROCESS_INTERVAL = 30         # Process every 30th frame to reduce computation
CONFIDENCE_THRESHOLD = 0.4          # Minimum confidence to consider a detection valid
TARGET_PHONE_CLASSES = ["cell phone", "mobile phone"]  # Class names to look for (case-insensitive)

# --- Load the YOLO model once globally ---
try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO model '{YOLO_MODEL_PATH}' loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model '{YOLO_MODEL_PATH}': {e}")

def detect_phone(video_path: str) -> list:
    """
    Detect frames containing phones.

    Args:
        video_path (str): Full path to the video file.

    Returns:
        List[dict]: List of events with timestamps where phones were detected.
                    Example: [{"timestamp": "12.3", "event_type": "PHONE_DETECTED"}]
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Could not determine FPS, defaulting to 30 FPS.")
        fps = 30

    frame_number = 0
    detected_events = []

    print(f"Starting phone detection on video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video reached
            break

        frame_number += 1

        # Skip frames to improve speed
        if frame_number % FRAME_PROCESS_INTERVAL != 0:
            continue

        # Run YOLO detection on the current frame
        results = model(frame, verbose=False)[0]  # Process the frame, get first result

        # Loop through all detected bounding boxes
        phone_found = False
        for box in results.boxes.data.tolist():
            # Unpack detection results: x1, y1, x2, y2, confidence, class_id
            _, _, _, _, confidence, class_id = box
            class_name = model.names[int(class_id)]

            # Check if detected object is a phone with confidence above threshold
            if confidence >= CONFIDENCE_THRESHOLD and class_name.lower() in TARGET_PHONE_CLASSES:
                timestamp_sec = frame_number / fps
                detected_events.append({
                    "timestamp": f"{timestamp_sec:.1f}",
                    "event_type": "PHONE_DETECTED"
                })
                print(f"[{timestamp_sec:.1f}s] Phone detected (confidence: {confidence:.2f}).")
                phone_found = True
                break  # Only one phone event per frame

    cap.release()
    cv2.destroyAllWindows()

    print(f"Phone detection complete. Found {len(detected_events)} events.")
    return detected_events

