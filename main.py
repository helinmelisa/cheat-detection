"""
main.py - Main entry point for video-based cheat detection system.

This script runs multiple detection modules on a given video file:
- Gaze detection (looking away)
- Phone detection
- Speech detection (speaking to someone)
- Multiple people detection

It aggregates all detected events and outputs a consolidated JSON report
with timestamps and event types.

Usage:
    python main.py
"""

import json
import os
import datetime
from collections import defaultdict

from detection.face_gaze import detect_looking_away
from detection.phone_detection import detect_phone
from detection.speech_detection import detect_speaking_to_someone
from detection.multiple_people import detect_multiple_people

# --- Configuration ---
VIDEO_PATH = r"\videos\Movie on 17.06.2025 at 14.42.mov"
OUTPUT_DIR = "output"


def generate_output_filename(video_path: str) -> str:
    video_basename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_basename)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name_without_ext}_{timestamp}_report.json"


def run_detection_module(detector_function, video_path: str, module_name: str, all_events: list) -> list:
    print(f"Running {module_name}...")
    try:
        new_events = detector_function(video_path)
        all_events.extend(new_events)
        print(f"{module_name} completed. Found {len(new_events)} events.")
        return new_events
    except Exception as e:
        print(f"Error during {module_name}: {e}")
        return []


def generate_summary(events: list) -> str:
    if not events:
        return "No suspicious activity detected in the video."

    summary_parts = []
    event_counts = defaultdict(int)

    for e in events:
        event_counts[e['event_type']] += 1

    if event_counts['LOOKING_AWAY']:
        summary_parts.append(f"{event_counts['LOOKING_AWAY']} instance(s) of looking away detected.")
    if event_counts['PHONE_DETECTED']:
        summary_parts.append(f"{event_counts['PHONE_DETECTED']} phone usage event(s) detected.")
    if event_counts['MULTIPLE_PEOPLE']:
        summary_parts.append(f"{event_counts['MULTIPLE_PEOPLE']} time(s) multiple people appeared on screen.")
    speaking_events = [e for e in events if e["event_type"] == "SPEAKING_TO_SOMEONE"]
    if speaking_events:
        summary_parts.append(f"{len(speaking_events)} segment(s) with speaking detected.")

    return "Summary of suspicious activity: " + " ".join(summary_parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_events = []
    output_filename = generate_output_filename(VIDEO_PATH)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Starting cheat detection for video: {VIDEO_PATH}")

    # Call each detection module separately to ensure they are working
    gaze_events = run_detection_module(detect_looking_away, VIDEO_PATH, "Gaze Detection", all_events)
    phone_events = run_detection_module(detect_phone, VIDEO_PATH, "Phone Detection", all_events)
    speaking_events = run_detection_module(detect_speaking_to_someone, VIDEO_PATH, "Audio Speech Detection", all_events)
    multiple_people_events = run_detection_module(detect_multiple_people, VIDEO_PATH, "Multiple People Detection", all_events)

    # Reconstruct the full list in case something failed earlier
    all_events = gaze_events + phone_events + speaking_events + multiple_people_events

    print(f"\nCompiling {len(all_events)} total events...")
    try:
        summary_text = generate_summary(all_events)
        output_data = {
            "events": all_events,
            "summary": summary_text
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        summary_txt_path = output_path.replace(".json", "_summary.txt")
        with open(summary_txt_path, "w") as f:
            f.write(summary_text)

        print(f"Detection complete. Report saved to: {output_path}")
        print(f"Summary saved to: {summary_txt_path}")

    except IOError as e:
        print(f"Error saving report to {output_path}: {e}")


if __name__ == "__main__":
    main()
