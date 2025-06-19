import cv2
import os
from ultralytics import YOLO


YOLO_MODEL = "yolov5su.pt"
FRAME_PROCESS_INTERVAL = 30 # Process every Nth frame to save computation
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence to consider a detection valid
# Classes to specifically look for. Using lowercase for robust comparison.
TARGET_PHONE_CLASSES = ["cell phone", "mobile phone"]

# Load the YOLO model globally to avoid reloading for every function call.
# The model will be downloaded automatically if not found.
try:
    model = YOLO(YOLO_MODEL)
    print(f"YOLO model '{YOLO_MODEL}' loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model '{YOLO_MODEL}'. Ensure it's valid and accessible. Error: {e}")

def detect_phone(video_path: str) -> list:
    """
    Detects instances where a phone is present in the video.

    Args:
        video_path (str): The full path to the video file to analyze.

    Returns:
        list: A list of dictionaries, each representing a 'PHONE_DETECTED' event
              with a timestamp.
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
        print(f"Warning: Could not get FPS from video {video_path}. Defaulting to 30 FPS.")
        fps = 30  # Default to 30 FPS if unable to retrieve

    frame_number = 0
    events = []

    print(f"Starting phone detection for {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video or error reading frame
            break

        frame_number += 1

        # Process only every Nth frame for performance
        if frame_number % FRAME_PROCESS_INTERVAL != 0:
            continue

        # Perform object detection on the frame
        # The [0] is used to get the first (and usually only) result object
        # when processing a single image/frame.
        results = model(frame, verbose=False)[0] # verbose=False to suppress print output per inference

        phone_found_in_frame = False
        # Iterate through detected objects
        for result in results.boxes.data.tolist():
            # Unpack detection results: x1, y1, x2, y2 (bounding box coords),
            # confidence (detection confidence), cls_id (class index)
            _, _, _, _, confidence, cls_id = result # We only need confidence and class_id here

            class_name = model.names[int(cls_id)]

            # Check if the detected object is a target phone class and meets confidence threshold
            if confidence >= CONFIDENCE_THRESHOLD and class_name.lower() in TARGET_PHONE_CLASSES:
                timestamp = frame_number / fps
                events.append({
                    "timestamp": f"{timestamp:.1f}", # Format to one decimal place
                    "event_type": "PHONE_DETECTED"
                })
                phone_found_in_frame = True
                break  # Exit loop after first phone detection in this frame, as per original logic


    cap.release()
    cv2.destroyAllWindows() # Ensure all OpenCV windows are closed if any were opened
    print(f"Phone detection finished. Found {len(events)} 'PHONE_DETECTED' events.")
    return events

if __name__ == "__main__":
    test_video_path = r"\videos\Movie on 17.06.2025 at 14.42.mov"
    
    if not os.path.exists(test_video_path):
        print(f"Warning: Test video not found at '{test_video_path}'. Skipping direct run test.")
        print("Please replace 'test_video_path' with a valid video path for testing.")
    else:
        print(f"Running phone detection on: {test_video_path}")
        phone_events = detect_phone(test_video_path)
        print("\n--- Detected Phone Events ---")
        if phone_events:
            for event in phone_events:
                print(event)
        else:
            print("No 'PHONE_DETECTED' events detected.")