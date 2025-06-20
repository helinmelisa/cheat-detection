"""
multiple_people.py - Detects frames containing multiple people in the video.

This module uses MediaPipe's Face Detection model to detect faces in each frame.
If two or more faces are detected in a frame, it flags that frame with a
'MULTIPLE_PEOPLE' event and records the timestamp.

This helps detect suspicious situations where extra people might be
present off-camera during a test (e.g., someone helping the candidate).
"""

import cv2
import mediapipe as mp
import os

# --- MediaPipe Face Detection Initialization ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils  # For drawing bounding boxes (debug view)

# Initialize MediaPipe face detector
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,               # 0 = short-range (for webcams), 1 = full-range
    min_detection_confidence=0.5     # Filter out weak detections
)

# Process every Nth frame to reduce computation
FRAME_PROCESS_INTERVAL = 5

def detect_multiple_people(video_path: str) -> list:
    """
    Detects frames where two or more faces appear at the same time.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        List[dict]: List of detected suspicious events.
            Example: [{"timestamp": "25.0", "event_type": "MULTIPLE_PEOPLE"}]
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    # Get video frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not determine FPS from video; defaulting to 30 FPS.")
        fps = 30

    frame_number = 0  # Counter to keep track of frame index
    detected_events = []  # List to store detected events

    print(f"Starting multiple people detection on video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video file
            break

        frame_number += 1

        # Skip frames for performance (process every Nth frame)
        if frame_number % FRAME_PROCESS_INTERVAL != 0:
            continue

        # Convert the image to RGB (MediaPipe requires RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detector.process(frame_rgb)

        # If two or more faces are detected, record this frame as suspicious
        if results.detections and len(results.detections) >= 2:
            timestamp_sec = frame_number / fps
            detected_events.append({
                "timestamp": f"{timestamp_sec:.1f}",  # Keep to 1 decimal place
                "event_type": "MULTIPLE_PEOPLE"
            })
            print(f"[{timestamp_sec:.1f}s] Multiple people detected ({len(results.detections)} faces).")

    # Release video capture resources
    cap.release()
    cv2.destroyAllWindows()

    print(f"Multiple people detection complete. Found {len(detected_events)} events.")
    return detected_events

# === Optional test run ===
if __name__ == "__main__":
    # Sample test video path 
    test_video_path = r"\videos\videoplayback.mp4"
    detect_multiple_people(test_video_path)
