"""
face_gaze.py - Detects 'LOOKING_AWAY' events using head pose estimation.

This module uses MediaPipe's Face Mesh to extract key facial landmarks,
and OpenCV's solvePnP function to estimate head pose angles: pitch, yaw, and roll.
A person is flagged as 'LOOKING_AWAY' when their head rotation appears suspicious
(e.g., looking far to the sides or excessively nodding).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# === Initialize MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,          # Set to False to process a video stream
    max_num_faces=1,                  # Track only one face (the test-taker)
    min_detection_confidence=0.5,     # Threshold for face detection confidence
    min_tracking_confidence=0.5       # Confidence threshold for tracking
)

# === 3D model points for head pose estimation ===
# These are approximate coordinates in a 3D space (in millimeters)
# for key facial features based on a generic human head.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye (outer corner)
    (225.0, 170.0, -135.0),      # Right eye (outer corner)
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# === Smoothing and filtering setup ===
pitch_history = deque(maxlen=10)  # Keep the last 10 pitch values
smoothed_pitch = None             # Exponential moving average value
ALPHA = 0.3                       # Smoothing factor
SUSPICIOUS_FRAME_THRESHOLD = 10   # Flag after N suspicious frames in a row

# === Camera matrix calculation (intrinsic parameters) ===
def get_camera_matrix(frame_w, frame_h):
    """
    Estimate camera intrinsic matrix assuming pinhole camera model.
    """
    focal_length = frame_w  # Assuming fx = fy = width
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No distortion assumed
    return camera_matrix, dist_coeffs

# === Map 2D landmark coordinates from MediaPipe to image space ===
def get_2d_image_points(landmarks, frame_w, frame_h):
    """
    Map selected landmark indices to 2D pixel coordinates.
    """
    return np.array([
        (landmarks[1][0] * frame_w, landmarks[1][1] * frame_h),     # Nose tip
        (landmarks[152][0] * frame_w, landmarks[152][1] * frame_h), # Chin
        (landmarks[263][0] * frame_w, landmarks[263][1] * frame_h), # Left eye outer corner
        (landmarks[33][0] * frame_w, landmarks[33][1] * frame_h),   # Right eye outer corner
        (landmarks[287][0] * frame_w, landmarks[287][1] * frame_h), # Left mouth corner
        (landmarks[57][0] * frame_w, landmarks[57][1] * frame_h)    # Right mouth corner
    ], dtype=np.float64)

# === Head Pose Estimation via SolvePnP ===
def estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Returns pitch, yaw, roll angles based on 2D-3D point correspondence.
    """
    image_points = get_2d_image_points(landmarks, frame_w, frame_h)
    camera_matrix, dist_coeffs = get_camera_matrix(frame_w, frame_h)

    success, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rotation_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # Extract pitch (x), yaw (y), and roll (z)
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll

# === Smooth pitch signal and detect outliers ===
def update_pitch(pitch):
    """
    Applies exponential smoothing to pitch and flags large deviations.
    """
    global smoothed_pitch
    if smoothed_pitch is None:
        smoothed_pitch = pitch
    else:
        smoothed_pitch = ALPHA * pitch + (1 - ALPHA) * smoothed_pitch

    pitch_history.append(smoothed_pitch)
    avg_pitch = sum(pitch_history) / len(pitch_history)
    deviation = abs(smoothed_pitch - avg_pitch)

    # Only consider pitch suspicious if it's outside typical forward-facing range
    is_suspicious = deviation > 1 and not (10.5 <= smoothed_pitch <= 12.5)
    return is_suspicious, smoothed_pitch

# === Main gaze detection function ===
def detect_looking_away(video_path):
    """
    Detects when the person is looking away (head turned or nodding suspiciously).
    Returns a list of timestamped events.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = 0
    consecutive_suspicious_frames = 0
    detected_events = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Track the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

            pose = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])
            if pose:
                pitch, yaw, roll = pose

                suspicious_pitch, smooth_pitch = update_pitch(pitch)
                suspicious_yaw = abs(yaw) > 20  # Looking far left/right

                suspicious = suspicious_pitch or suspicious_yaw

                if suspicious:
                    consecutive_suspicious_frames += 1
                else:
                    consecutive_suspicious_frames = 0

                # Trigger event only after enough consecutive suspicious frames
                if consecutive_suspicious_frames >= SUSPICIOUS_FRAME_THRESHOLD:
                    timestamp = total_frames / fps
                    detected_events.append({
                        "timestamp": f"{int(timestamp // 60):02d}:{timestamp % 60:04.1f}",
                        "event_type": "LOOKING_AWAY",
                        "details": {
                            "pitch": round(pitch, 2),
                            "yaw": round(yaw, 2),
                            "roll": round(roll, 2),
                            "smoothed_pitch": round(smooth_pitch, 2),
                            "reason": [
                                "Pitch angle suspicious" if suspicious_pitch else "",
                                "Yaw angle suspicious" if suspicious_yaw else ""
                            ]
                        }
                    })
                    consecutive_suspicious_frames = 0

        else:
            # No face detected in this frame
            consecutive_suspicious_frames = 0

    cap.release()
    return detected_events


# === Test the gaze detection function ===
if __name__ == "__main__":
    video_path = r"\videos\Movie on 17.06.2025 at 14.39.mov"
    detect_looking_away(video_path)
