# Video-Based Cheat Detection

This project provides a system for analyzing video footage of a person taking a test or an interview to detect potential signs of cheating. The goal is to offer a practical and explainable solution using computer vision and audio analysis to flag suspicious behaviors.

This solution prioritizes thoughtful design and the use of high-level tools and pre-trained models over building a perfect, flawless system.

## Objective

The primary objective is to identify and flag a set of suspicious behaviors that could indicate cheating, including:

- Looking off-screen (e.g., at notes)
- Speaking to someone off-camera
- Use of a phone or unauthorized devices
- Presence of multiple people


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/helinmelisa/cheat-detection.git
cd cheat-detection
```
### 2.  Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Detection
Put your video under the videos/ folder. Then edit the VIDEO_PATH in main.py to point to your video.

```bash
python main.py
```
Reports will be saved in the outputs/ folder.

## Approach & Module Overview

The system is modular and consists of four detection units:

### 1. Gaze Detection (Visual)

- Uses MediaPipe FaceMesh to track facial landmarks.

- Applies head pose estimation to detect excessive pitch/yaw, indicating “looking away.”

- Applies smoothing over frames and confirms only if the behavior is consistent across N frames.

### 2. Phone Detection (Visual)

- Uses a pretrained YOLOv5 model to detect mobile phones in the scene.

- Flags a detection if the confidence is high enough.

### 3. Multiple People Detection (Visual)

- Uses MediaPipe Face Detection to count the number of faces per frame.

- If ≥ 2 faces are present in a single frame, it's flagged as a suspicious event.

### 4. Speech Detection (Audio)

- Uses OpenAI Whisper to transcribe speech from the video.

- Merges segments with short gaps to reduce noise.

- Flags speech events longer than a threshold as potential communication with someone off-screen.


Each module outputs its own timestamped events. These are compiled into a single report.

## Tools, Libraries, and Models Used
| Tool / Library      | Purpose                                 |   
| ------------------- | --------------------------------------- | 
| OpenCV              | Frame-by-frame video analysis           |   
| MediaPipe           | Face landmark and face detection        |   
| YOLOv5              | Real-time object detection (phones)     |   
| OpenAI Whisper      | Audio transcription (speech detection)  |   
| ffmpeg-python       | Extracting audio from video             |   
| NumPy               | Numeric operations, smoothing, matrices |   
| Python Standard Lib | File I/O, JSON, datetime, os            |   

## Limitations, Assumptions, and Trade-offs
- False Positives/Negatives:
The system uses heuristic thresholds (e.g., pitch > 20°, yaw > 25°, EAR < 0.2), which may lead to occasional misclassification depending on lighting, head pose, or accents.

- Single-Person Assumption:
The gaze detection and Whisper models assume the subject is in focus and centered. Multiple people or noisy environments may confuse results.

- YOLO Phone Detection:
Only works well when the phone is clearly visible to the camera and not occluded or partially shown.

- Language Bias:
Whisper detects and transcribes any spoken language, but performance may vary across accents and background noise levels.

- Frame Skipping:
To optimize performance, not every frame is analyzed. This may miss very short events, especially for phone or people detection.

- Offline-Only:
The system currently processes pre-recorded videos and does not work in real time.

## Future Work & Possible Improvements
While this system offers a practical baseline for video-based cheat detection, several enhancements could significantly improve its robustness, scalability, and real-time performance:

1. Real-Time Monitoring
- Extend the system to support live webcam streams (e.g., via WebRTC) for real-time proctoring rather than post-event analysis.

2. Improved Gaze Calibration
- Introduce per-user calibration to account for different head positions, eye shapes, or webcam angles—especially to reduce false gaze detections.

3. Multi-Person Context Awareness
- Instead of just counting faces, use person re-identification and pose estimation to track which person is speaking or using a phone.

4. Environment Awareness
- Add scene classification (e.g., detecting mirrors, TVs, or multiple screens) to better understand the testing environment.

5. Speaker Diarization
- Combine Whisper with voice separation or diarization tools to detect if multiple people are speaking (vs. a single speaker talking to themselves).

6. Custom Training for Phones
- Detect contextual cues like screens, mirrors, or multiple monitors that could facilitate cheating but aren't currently flagged.

7. Temporal Correlation Across Modalities
- Fuse visual and audio modalities: for example, confirming speech only when the mouth is visibly moving, or correlating head turn with speech onset.

