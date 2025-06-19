"""
face_gaze.py - Detects suspicious head movement, such as looking away from the screen.

This module uses MediaPipe Face Mesh to estimate head pose (pitch, yaw, roll).
It analyzes these angles over time to detect if the user frequently looks away
from the screen (e.g., turning head or looking up/down too much).

Suspicious movement is flagged if:
- Yaw exceeds ±20° (i.e., turning head left or right).
- Pitch changes significantly from the average over time (e.g., looking up/down).
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Define 3D model points of key facial features ---
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def get_camera_matrix(frame_width, frame_height):
    """
    Returns a basic camera matrix for pose estimation.
    """
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # No lens distortion
    return camera_matrix, dist_coeffs

def get_2d_image_points(landmarks, frame_w, frame_h):
    """
    Maps selected 3D model points to their corresponding 2D image locations.
    """
    image_points = np.array([
        (landmarks[1][0] * frame_w, landmarks[1][1] * frame_h),     # Nose tip
        (landmarks[152][0] * frame_w, landmarks[152][1] * frame_h), # Chin
        (landmarks[263][0] * frame_w, landmarks[263][1] * frame_h), # Left eye
        (landmarks[33][0] * frame_w, landmarks[33][1] * frame_h),   # Right eye
        (landmarks[287][0] * frame_w, landmarks[287][1] * frame_h), # Left mouth
        (landmarks[57][0] * frame_w, landmarks[57][1] * frame_h),   # Right mouth
    ], dtype=np.float64)
    return image_points

def estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Uses PnP to estimate head orientation (pitch, yaw, roll).
    """
    image_points = get_2d_image_points(landmarks, frame_w, frame_h)
    camera_matrix, dist_coeffs = get_camera_matrix(frame_w, frame_h)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll  # in degrees

# --- Pitch Smoothing Logic ---
pitch_history = deque(maxlen=10)
smoothed_pitch = None
alpha = 0.3  # Smoothing factor

def update_pitch(pitch):
    """
    Smooth pitch using exponential moving average and detect abrupt changes.
    """
    global smoothed_pitch
    if smoothed_pitch is None:
        smoothed_pitch = pitch
    else:
        smoothed_pitch = alpha * pitch + (1 - alpha) * smoothed_pitch

    pitch_history.append(smoothed_pitch)
    pitch_avg = sum(pitch_history) / len(pitch_history)
    diff = abs(smoothed_pitch - pitch_avg)
    suspicious = diff > 1  # Flag if pitch deviates too much from average
    return suspicious, smoothed_pitch

# --- Detection Logic ---
output_folder = "detected_screenshots"
os.makedirs(output_folder, exist_ok=True)

def detect_looking_away(video_path):
    """
    Detects if user frequently looks away from the screen.
    
    Returns:
        List[dict]: Timestamps and events for looking away.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    suspicious_frames = 0
    total_frames = 0
    consecutive_suspicious_frames = 0
    SUSPICIOUS_FRAME_THRESHOLD = 10  # Require 10 frames in a row

    detected_timestamps = []
    detected_info = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        cheat_detected = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            pose = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])

            if pose:
                pitch, yaw, roll = pose
                suspicious_pitch, smoothed_pitch_val = update_pitch(pitch)
                suspicious_yaw = abs(yaw) > 20
                suspicious = suspicious_pitch or suspicious_yaw

                if suspicious:
                    consecutive_suspicious_frames += 1
                else:
                    consecutive_suspicious_frames = 0

                if consecutive_suspicious_frames >= SUSPICIOUS_FRAME_THRESHOLD:
                    cheat_detected = True
                    timestamp_sec = total_frames / fps
                    detected_timestamps.append(timestamp_sec)
                    detected_info.append((timestamp_sec, suspicious_pitch, smoothed_pitch_val))
                    filename = f"cheat_{timestamp_sec:.2f}s.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, frame)

        else:
            consecutive_suspicious_frames = 0

        cv2.imshow("Cheat Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert results to JSON-friendly output
    detected_events = [
        {
            "timestamp": f"{timestamp:.2f}",
            "event_type": "LOOKING_AWAY"
        }
        for timestamp, _, _ in detected_info
    ]

    print(f"Total frames analyzed: {total_frames}")
    print(f"Suspicious frames detected: {suspicious_frames}")
    print(f"Detected timestamps: {[f'{t:.2f}' for t, _, _ in detected_info]}")
    
    return detected_events

if __name__ == "__main__":
    video_path = r"\videos\Movie on 17.06.2025 at 14.34.mov"
    detect_looking_away(video_path)
