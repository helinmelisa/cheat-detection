"""
speech_detection.py - Detects when the test-taker is speaking during the video.

This module extracts the audio track from the input video using ffmpeg,
then uses OpenAI's Whisper model to transcribe speech segments.

It groups short segments that are close in time to reduce fragmentation,
and returns timestamps with detected speech along with the transcribed text.

Use case: Helps identify unauthorized speech (e.g., asking someone for help during a test).
"""

import os
import ffmpeg
import whisper
import json

# --- Configuration Constants ---
WHISPER_MODEL_SIZE = "base"        # Options: tiny, base, small, medium, large
TEMP_AUDIO_PATH = "temp_audio.wav" # Temporary path for extracted audio
MIN_TEXT_LENGTH = 5                # Skip very short (likely noise) segments
MERGE_THRESHOLD_SEC = 1.5          # Max gap (in sec) to merge consecutive segments
SAVE_TRANSCRIPT_JSON = False        # Whether to save output to a JSON file

def extract_audio_with_ffmpeg(video_path: str, output_wav_path: str, sample_rate: int = 16000) -> bool:
    """
    Extracts mono WAV audio from a video using ffmpeg.

    Args:
        video_path (str): Path to the input video.
        output_wav_path (str): Path to store the extracted audio (WAV).
        sample_rate (int): Sampling rate in Hz (default 16000 for Whisper compatibility).

    Returns:
        bool: True if extraction succeeded, otherwise False.
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
    Converts seconds into MM:SS.s format (human-friendly).

    Args:
        seconds (float): Timestamp in seconds.

    Returns:
        str: Formatted string (e.g., "03:25.6").
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"

def detect_speaking_to_someone(video_path: str) -> list:
    """
    Detects spoken speech events in a video using Whisper.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list[dict]: List of detected speech segments.
            Each entry includes:
            - timestamp_start (str): When the speech starts (formatted MM:SS.s)
            - timestamp_end (str): When the speech ends (formatted MM:SS.s)
            - event_type (str): Always "SPEAKING_TO_SOMEONE"
            - text (str): Transcribed content
            - language (str): Detected language
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return []

    print(f"[INFO] Extracting audio from video: {video_path}")
    if not extract_audio_with_ffmpeg(video_path, TEMP_AUDIO_PATH):
        return []

    try:
        # Load Whisper model
        print(f"[INFO] Loading Whisper model: {WHISPER_MODEL_SIZE}")
        model = whisper.load_model(WHISPER_MODEL_SIZE)

        # Transcribe audio
        print("[INFO] Transcribing audio...")
        transcription = model.transcribe(TEMP_AUDIO_PATH, verbose=False, temperature=0.0)
        segments = transcription.get("segments", [])
        language = transcription.get("language", "unknown")

        print(f"[INFO] Detected language: {language}")
        print(f"[INFO] Total raw segments detected: {len(segments)}")

        # Merge adjacent short segments (less than 1.5s apart)
        merged_segments = []
        current = None

        for seg in segments:
            text = seg.get("text", "").strip()
            if len(text) < MIN_TEXT_LENGTH:
                continue  # Skip irrelevant/noisy text

            start = seg.get("start")
            end = seg.get("end")

            # Merge logic: combine with previous segment if within threshold
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

        # Format final output
        speaking_events = []
        for seg in merged_segments:
            speaking_events.append({
                "timestamp_start": format_timestamp(seg["start"]),
                "timestamp_end": format_timestamp(seg["end"]),
                "event_type": "SPEAKING_TO_SOMEONE",
                "text": seg["text"],
                "language": language
            })
            print(f"[{speaking_events[-1]['timestamp_start']} - {speaking_events[-1]['timestamp_end']}] "
                  f"{speaking_events[-1]['text']}")

        # Save JSON
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
        # Clean up temporary audio file
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
            print(f"[INFO] Removed temporary audio file: {TEMP_AUDIO_PATH}")

# --- Testing ---
if __name__ == "__main__":
    test_video = r"\videos\Movie on 17.06.2025 at 14.42.mov"
    detect_speaking_to_someone(test_video)
