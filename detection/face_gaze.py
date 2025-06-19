import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- Configuration Constants ---
# Yaw: Rotation around the Y-axis (looking left/right)
# Pitch: Rotation around the X-axis (looking up/down)
# Roll: Rotation around the Z-axis (tilting head side-to-side)
HEAD_POSE_YAW_THRESHOLD_DEGREES = 20 # Degrees left or right to consider "looking away"
HEAD_POSE_PITCH_THRESHOLD_DEGREES = 20 # Degrees up or down to consider "looking away" (optional)
FRAME_PROCESS_INTERVAL = 5           # Process every Nth frame to save computation
SHOW_DEBUG_WINDOW = False            # Toggle to True to show visual output (for testing)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Face Mesh
# max_num_faces=1: Focus on the main subject to avoid complexity with multiple faces
face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Provides more detailed landmarks, especially for eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define 3D model points (object points) for head pose estimation
# These are relative coordinates in a generic 3D head model.
# These specific points are chosen to correspond to MediaPipe's facial landmarks
# and are common for PnP (Perspective-n-Point) algorithm.
# Source: https://github.com/ManuelTS/augmentedFaceMeshModule/blob/master/utils.py
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1, or sometimes 4)
    (0.0, -330.0, -65.0),        # Chin (landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye left corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye right corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype="double")


def _get_image_points_from_landmarks(landmarks, image_w, image_h):
    """
    Extracts 2D image coordinates for specific facial landmarks used for head pose.

    Args:
        landmarks: MediaPipe's detected facial landmarks object.
        image_w (int): Width of the image frame.
        image_h (int): Height of the image frame.

    Returns:
        numpy.ndarray: An array of 2D (x, y) coordinates for the selected landmarks.
    """
    # MediaPipe landmark indices for the 6 points corresponding to `model_points`:
    # 1: Nose tip
    # 152: Chin
    # 33: Left eye (inner corner)
    # 263: Right eye (inner corner)
    # 61: Left mouth corner
    # 291: Right mouth corner
    selected_landmark_indices = [1, 152, 33, 263, 61, 291]

    image_points = np.array([
        (landmarks.landmark[idx].x * image_w, landmarks.landmark[idx].y * image_h)
        for idx in selected_landmark_indices
    ], dtype="double")
    return image_points

def detect_looking_away(video_path: str) -> list:
    """
    Detects frames where the subject is likely looking away from the screen using MediaPipe for head pose.
    Returns a list of timestamped 'LOOKING_AWAY' events.

    Args:
        video_path (str): The path to the input video file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a 'looking away' event
              and contains the timestamp and event type.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not retrieve FPS for '{video_path}'. Defaulting to 30 FPS for timestamps.")
        fps = 30 # Default to 30 FPS to avoid division by zero

    frame_count = 0
    looking_away_events = []

    # Get camera matrix (intrinsic parameters) - crucial for PnP
    # fx, fy are focal lengths, cx, cy are optical centers.
    img_h, img_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    focal_length = img_w # A common approximation for wide-angle lenses or average webcams
    camera_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype="double")

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Create a blank image for drawing landmarks if debug window is enabled
    if SHOW_DEBUG_WINDOW:
        cv2.namedWindow('Debug Output', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video stream

        frame_count += 1
        # Skip frames to improve performance
        if frame_count % FRAME_PROCESS_INTERVAL != 0:
            continue

        # Convert the BGR image to RGB before processing with MediaPipe Face Mesh
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = face_mesh_detector.process(image_rgb)

        # Draw the face mesh annotations on the image.
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the 2D image points from the detected landmarks
                try:
                    image_points = _get_image_points_from_landmarks(face_landmarks, img_w, img_h)
                except IndexError:
                    # This can happen if a landmark index is out of bounds or not detected
                    print("Warning: Could not extract all required landmarks for head pose.")
                    continue

                # SolvePnP for pose estimation: Find rotation and translation vectors
                # `solvePnP` returns `(success, rotation_vector, translation_vector)`
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                    # Get Euler angles from rotation matrix (yaw, pitch, roll)
                    # Simplified Euler angles from rotation matrix (Y-X-Z convention is common for head pose):
                    # yaw (y-axis rotation): looking left/right
                    # pitch (x-axis rotation): looking up/down
                    # roll (z-axis rotation): tilting head left/right
                    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +
                                   rotation_matrix[1, 0] * rotation_matrix[1, 0])
                    singular = sy < 1e-6

                    if not singular:
                        x_angle = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        y_angle = math.atan2(-rotation_matrix[2, 0], sy)
                        z_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        x_angle = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                        y_angle = math.atan2(-rotation_matrix[2, 0], sy)
                        z_angle = 0

                    # Convert radians to degrees
                    pitch = math.degrees(x_angle)
                    yaw = math.degrees(y_angle)
                    roll = math.degrees(z_angle)

                    # Determine if looking away based on yaw and optionally pitch
                    is_looking_away = (abs(yaw) > HEAD_POSE_YAW_THRESHOLD_DEGREES) or \
                                      (abs(pitch) > HEAD_POSE_PITCH_THRESHOLD_DEGREES)

                    if is_looking_away:
                        timestamp_seconds = frame_count / fps
                        looking_away_events.append({
                            "timestamp": f"{timestamp_seconds:.2f}",
                            "event_type": "LOOKING_AWAY",
                            "head_pose_yaw_degrees": f"{yaw:.2f}",
                            "head_pose_pitch_degrees": f"{pitch:.2f}"
                        })

                    if SHOW_DEBUG_WINDOW:
                        # Draw axes on the face to visualize head pose
                        # Origin (0,0,0) in the 3D model (nose tip)
                        nose_tip_2d = tuple(np.array(image_points[0], dtype=int))
                        # Project 3D points (axes) to 2D image plane
                        axis_points, _ = cv2.projectPoints(
                            np.array([(300.0,0.0,0.0), (0.0,300.0,0.0), (0.0,0.0,300.0)]),
                            rotation_vector, translation_vector, camera_matrix, dist_coeffs
                        )
                        p1 = tuple(np.array(axis_points[0][0], dtype=int))
                        p2 = tuple(np.array(axis_points[1][0], dtype=int))
                        p3 = tuple(np.array(axis_points[2][0], dtype=int))

                        cv2.line(image_bgr, nose_tip_2d, p1, (0, 0, 255), 2) # X-axis (Red)
                        cv2.line(image_bgr, nose_tip_2d, p2, (0, 255, 0), 2) # Y-axis (Green)
                        cv2.line(image_bgr, nose_tip_2d, p3, (255, 0, 0), 2) # Z-axis (Blue)

                        # Display pose angles
                        cv2.putText(image_bgr, f"Yaw: {yaw:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image_bgr, f"Pitch: {pitch:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image_bgr, f"Roll: {roll:.2f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image_bgr, "Looking Away!" if is_looking_away else "Looking Fwd", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_looking_away else (255, 0, 0), 2)


        if SHOW_DEBUG_WINDOW:
            cv2.imshow('Debug Output', image_bgr)
            if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit debug window
                break

    cap.release()
    if SHOW_DEBUG_WINDOW:
        cv2.destroyAllWindows()
    return looking_away_events

if __name__ == "__main__":
    video_path = r"\videos\Movie on 17.06.2025 at 14.39.mov" # Example path, replace with yours!

    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        events = detect_looking_away(video_path)
        if events:
            print("\n--- Detected Looking Away Events ---")
            for event in events:
                print(event)
        else:
            print("No 'looking away' events detected.")
    else:
        print(f"Error: Video file not found at '{video_path}'. Please provide a valid video path for testing.")
