"""
multiple_people.py - Detects frames containing multiple people in the video.

This module uses MediaPipe's Face Detection model to detect faces in each frame.
If two or more faces are detected in a frame, it flags that frame with a
'MULTIPLE_PEOPLE' event and records the timestamp.

This helps detect suspicious situations where extra people might be
present off-camera during a test.
"""

import cv2
import mediapipe as mp
import os

# --- MediaPipe Face Detection Initialization ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils  # Optional: for drawing bounding boxes
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,               # Use short-range detection suitable for webcams
    min_detection_confidence=0.5    # Confidence threshold to filter weak detections
)

FRAME_PROCESS_INTERVAL = 5  # Process every 5th frame for performance

def detect_multiple_people(video_path: str) -> list:
    """
    Detects frames where two or more people appear simultaneously.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        List[dict]: List of detected events with timestamps in seconds.
        Example: [{"timestamp": "25.0", "event_type": "MULTIPLE_PEOPLE"}]
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
        print(f"Warning: Could not determine FPS from video; defaulting to 30 FPS.")
        fps = 30  # Default FPS if not found

    frame_number = 0
    detected_events = []

    print(f"Starting multiple people detection on video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video reached
            break

        frame_number += 1

        # Skip frames to improve performance
        if frame_number % FRAME_PROCESS_INTERVAL != 0:
            continue

        # Convert frame to RGB as MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        results = face_detector.process(frame_rgb)

        # Check if there are 2 or more faces detected
        if results.detections and len(results.detections) >= 2:
            timestamp_sec = frame_number / fps
            detected_events.append({
                "timestamp": f"{timestamp_sec:.1f}",  # One decimal place (seconds)
                "event_type": "MULTIPLE_PEOPLE"
            })
            print(f"[{timestamp_sec:.1f}s] Multiple people detected ({len(results.detections)} faces).")

    cap.release()
    cv2.destroyAllWindows()

    print(f"Multiple people detection complete. Found {len(detected_events)} events.")
    return detected_events

# === Test the multiple people detection function ===
if __name__ == "__main__":
    test_video_path = r"\videos\videoplayback.mp4"
    detect_multiple_people(test_video_path)
       