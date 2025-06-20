"""
phone_detection.py - Detects presence of phones (cell phones or mobiles) in video frames.

This module uses a pretrained YOLOv5 model (e.g., yolov5su.pt) to detect objects.
It processes the video frame-by-frame (skipping frames for performance) and
flags frames where a phone is detected with a confidence above a threshold.

This helps identify potential cheating via unauthorized devices (e.g., smartphones).
"""

import cv2
import os
from ultralytics import YOLO  

# --- Configuration Constants ---
YOLO_MODEL_PATH = "yolov5su.pt"     # Path to the YOLO model weights
FRAME_PROCESS_INTERVAL = 30         # Analyze every 30th frame for speed
CONFIDENCE_THRESHOLD = 0.4          # Minimum required confidence for valid detections
TARGET_PHONE_CLASSES = ["cell phone", "mobile phone"]  # Class names to match

# --- Load YOLO model globally (loaded once) ---
try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO model '{YOLO_MODEL_PATH}' loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model '{YOLO_MODEL_PATH}': {e}")

def detect_phone(video_path: str) -> list:
    """
    Detects presence of phones in a video using YOLO object detection.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        List[dict]: A list of events where a phone was detected.
                    Each dict contains:
                        - "timestamp": Timestamp of detection (in seconds, str)
                        - "event_type": Always "PHONE_DETECTED"
        Example:
            [{"timestamp": "12.3", "event_type": "PHONE_DETECTED"}]
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    # Get frames-per-second (fps) for timestamp calculations
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Could not determine FPS, defaulting to 30 FPS.")
        fps = 30

    frame_number = 0           # Index of the current frame
    detected_events = []       # Stores detection results

    print(f"Starting phone detection on video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_number += 1

        # Skip frames for performance
        if frame_number % FRAME_PROCESS_INTERVAL != 0:
            continue

        # Run YOLO detection
        results = model(frame, verbose=False)[0]  # Process frame, take first result only

        phone_found = False  # Flag to mark if a phone was detected in this frame

        # Iterate over detected bounding boxes
        for box in results.boxes.data.tolist():
            # YOLO returns [x1, y1, x2, y2, confidence, class_id]
            _, _, _, _, confidence, class_id = box
            class_name = model.names[int(class_id)]  # Map class_id to readable label

            # Check if this is a phone and confidence is acceptable
            if confidence >= CONFIDENCE_THRESHOLD and class_name.lower() in TARGET_PHONE_CLASSES:
                timestamp_sec = frame_number / fps
                detected_events.append({
                    "timestamp": f"{timestamp_sec:.1f}",
                    "event_type": "PHONE_DETECTED"
                })
                print(f"[{timestamp_sec:.1f}s] Phone detected (confidence: {confidence:.2f}).")
                phone_found = True
                break  # Only record first valid detection per frame

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print(f"Phone detection complete. Found {len(detected_events)} events.")
    return detected_events
