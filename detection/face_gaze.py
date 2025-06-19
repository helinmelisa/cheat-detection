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
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

def get_camera_matrix(frame_width, frame_height):
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    return camera_matrix, dist_coeffs

def get_2d_image_points(landmarks, frame_w, frame_h):
    image_points = np.array([
        (landmarks[1][0] * frame_w, landmarks[1][1] * frame_h),
        (landmarks[152][0] * frame_w, landmarks[152][1] * frame_h),
        (landmarks[263][0] * frame_w, landmarks[263][1] * frame_h),
        (landmarks[33][0] * frame_w, landmarks[33][1] * frame_h),
        (landmarks[287][0] * frame_w, landmarks[287][1] * frame_h),
        (landmarks[57][0] * frame_w, landmarks[57][1] * frame_h),
    ], dtype=np.float64)
    return image_points

def estimate_head_pose(landmarks, frame_w, frame_h):
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
    return pitch, yaw, roll

pitch_history = deque(maxlen=10)
smoothed_pitch = None
alpha = 0.3

def update_pitch(pitch):
    global smoothed_pitch
    if smoothed_pitch is None:
        smoothed_pitch = pitch
    else:
        smoothed_pitch = alpha * pitch + (1 - alpha) * smoothed_pitch
    pitch_history.append(smoothed_pitch)
    pitch_avg = sum(pitch_history) / len(pitch_history)
    diff = abs(smoothed_pitch - pitch_avg)
    suspicious = diff > 3
    return suspicious, smoothed_pitch

def compute_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_vertical_gaze_ratio(eye_top_idx, eye_bottom_idx, iris_center_idx, landmarks, frame_w, frame_h):
    top = np.array([landmarks[eye_top_idx][0] * frame_w, landmarks[eye_top_idx][1] * frame_h])
    bottom = np.array([landmarks[eye_bottom_idx][0] * frame_w, landmarks[eye_bottom_idx][1] * frame_h])
    center = np.array([landmarks[iris_center_idx][0] * frame_w, landmarks[iris_center_idx][1] * frame_h])
    eye_height = np.linalg.norm(top - bottom)
    iris_to_top = np.linalg.norm(center - top)
    ratio = iris_to_top / eye_height if eye_height > 0 else 0
    return ratio

output_folder = "detected_screenshots"
os.makedirs(output_folder, exist_ok=True)

def detect_looking_away(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = 0
    consecutive_suspicious_frames = 0
    SUSPICIOUS_FRAME_THRESHOLD = 10

    detected_info = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            pose = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])

            def pt(i): return np.array([landmarks[i][0] * frame.shape[1], landmarks[i][1] * frame.shape[0]])
            left_eye = [pt(i) for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [pt(i) for i in [362, 385, 387, 263, 373, 380]]
            avg_ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
            eyes_closed = avg_ear < 0.2

            vertical_gaze_ratio = (
                get_vertical_gaze_ratio(159, 145, 468, landmarks, frame.shape[1], frame.shape[0]) +
                get_vertical_gaze_ratio(386, 374, 473, landmarks, frame.shape[1], frame.shape[0])
            ) / 2.0
            looking_up = vertical_gaze_ratio < 0.35
            looking_down = vertical_gaze_ratio > 0.65

            if pose:
                pitch, yaw, roll = pose
                suspicious_pitch, smoothed_pitch_val = update_pitch(pitch)
                suspicious_yaw = abs(yaw) > 20

                suspicious_reasons = []
                if suspicious_pitch:
                    suspicious_reasons.append("pitch")
                if suspicious_yaw:
                    suspicious_reasons.append("yaw")
                if looking_up:
                    suspicious_reasons.append("looking_up")
                if looking_down:
                    suspicious_reasons.append("looking_down")
                if eyes_closed:
                    suspicious_reasons.append("eyes_closed")

                suspicious = len(suspicious_reasons) > 0

                if suspicious:
                    consecutive_suspicious_frames += 1
                else:
                    consecutive_suspicious_frames = 0

                if consecutive_suspicious_frames >= SUSPICIOUS_FRAME_THRESHOLD:
                    timestamp_sec = total_frames / fps
                    detected_info.append({
                        "timestamp": timestamp_sec,
                        "pitch": pitch,
                        "yaw": yaw,
                        "ear": avg_ear,
                        "gaze_ratio": vertical_gaze_ratio,
                        "reasons": suspicious_reasons,
                        "event_type": "LOOKING_AWAY",
                    })
                    filename = f"cheat_{timestamp_sec:.2f}s.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, frame)

                text = f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f} | EAR: {avg_ear:.2f} | Gaze: {vertical_gaze_ratio:.2f}"
                status = "CHEATING" if suspicious else "Normal"
                reason_text = f"Reasons: {', '.join(suspicious_reasons)}" if suspicious else "Reasons: -"

                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if suspicious else (0, 255, 0), 2)
                cv2.putText(frame, reason_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 200), 1)

        else:
            consecutive_suspicious_frames = 0

        cv2.imshow("Cheat Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total frames analyzed: {total_frames}")
    print(f"Suspicious events detected: {len(detected_info)}")
    for event in detected_info:
        print(f"[{event['timestamp']:.2f}s] Reasons: {', '.join(event['reasons'])}")

    return detected_info

if __name__ == "__main__":
    video_path = r"\videos\Movie on 17.06.2025 at 14.34.mov"
    detect_looking_away(video_path)