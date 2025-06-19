"""
speech_detection.py - Detects when the test-taker is speaking during the video.

This module extracts the audio track from the input video using ffmpeg,
then uses OpenAI's Whisper model to transcribe speech segments.

It groups short segments close in time to reduce fragmentation,
and returns timestamps with detected speech along with the transcribed text.
"""

import os
import ffmpeg
import whisper
import json

# --- Configuration Constants ---
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
TEMP_AUDIO_PATH = "temp_audio.wav"
MIN_TEXT_LENGTH = 5
MERGE_THRESHOLD_SEC = 1.5
SAVE_TRANSCRIPT_JSON = True  

def extract_audio_with_ffmpeg(video_path: str, output_wav_path: str, sample_rate: int = 16000) -> bool:
    """
    Extracts audio from video as a mono WAV file using ffmpeg.

    Args:
        video_path (str): Input video path.
        output_wav_path (str): Output WAV audio path.
        sample_rate (int): Target audio sample rate.

    Returns:
        bool: True if extraction succeeded, else False.
    """
    try:
        ffmpeg.input(video_path).output(
            output_wav_path,
            format='wav',
            acodec='pcm_s16le',
            ac=1,              # Mono audio channel
            ar=sample_rate     # Sample rate in Hz
        ).run(overwrite_output=True, quiet=True)
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode(errors='ignore') if e.stderr else str(e)}")
        return False

def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to a formatted string MM:SS.s

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted timestamp string.
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"

def detect_speaking_to_someone(video_path: str) -> list:
    """
    Detects speaking events in the video by transcribing audio using Whisper.

    Args:
        video_path (str): Path to input video file.

    Returns:
        list of dict: List of detected speaking events with start/end timestamps and transcribed text.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return []

    print(f"[INFO] Extracting audio from video: {video_path}")
    if not extract_audio_with_ffmpeg(video_path, TEMP_AUDIO_PATH):
        return []

    try:
        print(f"[INFO] Loading Whisper model: {WHISPER_MODEL_SIZE}")
        model = whisper.load_model(WHISPER_MODEL_SIZE)

        print("[INFO] Transcribing audio...")
        transcription = model.transcribe(TEMP_AUDIO_PATH, verbose=False, temperature=0.0)
        segments = transcription.get("segments", [])
        language = transcription.get("language", "unknown")

        print(f"[INFO] Detected language: {language}")
        print(f"[INFO] Total segments detected: {len(segments)}")

        merged_segments = []
        current = None

        for seg in segments:
            text = seg.get("text", "").strip()
            if len(text) < MIN_TEXT_LENGTH:
                continue

            start = seg.get("start")
            end = seg.get("end")

            if current is None:
                current = {"start": start, "end": end, "text": text}
            elif start - current["end"] <= MERGE_THRESHOLD_SEC:
                current["text"] += " " + text
                current["end"] = end
            else:
                merged_segments.append(current)
                current = {"start": start, "end": end, "text": text}

        if current:
            merged_segments.append(current)

        print(f"[INFO] Merged segments: {len(merged_segments)}")

        speaking_events = []
        for seg in merged_segments:
            speaking_events.append({
                "timestamp_start": format_timestamp(seg["start"]),
                "timestamp_end": format_timestamp(seg["end"]),
                "event_type": "SPEAKING_TO_SOMEONE",
                "text": seg["text"],
                "language": language
            })

            print(speaking_events)

        if SAVE_TRANSCRIPT_JSON:
            out_path = os.path.splitext(video_path)[0] + "_speech.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(speaking_events, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved speaking events to: {out_path}")

        print(f"[SUCCESS] Transcription complete. {len(speaking_events)} speaking events detected.")
        return speaking_events

    except Exception as e:
        print(f"[ERROR] Whisper transcription failed: {e}")
        return []

    finally:
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
            print(f"[INFO] Removed temporary audio file: {TEMP_AUDIO_PATH}")

# --- Test ---
if __name__ == "__main__":
    test_video = r"\videos\Movie on 17.06.2025 at 14.42.mov"
    detect_speaking_to_someone(test_video)
