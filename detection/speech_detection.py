"""
speech_detection.py - Detects when the test-taker is speaking during the video.

This module extracts the audio track from the input video using ffmpeg,
then uses OpenAI's Whisper model to transcribe speech segments.

It groups short segments close in time to reduce fragmentation,
and returns timestamps with detected speech along with the transcribed text.

Useful to detect if the candidate is speaking to someone off-camera,
which could indicate potential cheating.
"""

import os
import ffmpeg
import whisper

# --- Configuration Constants ---
WHISPER_MODEL_SIZE = "base"  # Model size: tiny, base, small, medium, large (adjust for speed vs accuracy)
TEMP_AUDIO_PATH = "temp_audio.wav"
MIN_TEXT_LENGTH = 5          # Minimum length of transcribed text to consider valid
MERGE_THRESHOLD_SEC = 1.5    # Merge speech segments if closer than this in seconds

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
        err_msg = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        print(f"Error extracting audio: {err_msg}")
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
        print(f"Error: Video file not found: {video_path}")
        return []

    # Step 1: Extract audio track from video
    success = extract_audio_with_ffmpeg(video_path, TEMP_AUDIO_PATH)
    if not success:
        return []

    try:
        # Step 2: Load Whisper model
        model = whisper.load_model(WHISPER_MODEL_SIZE)
        print(f"Loaded Whisper ASR model '{WHISPER_MODEL_SIZE}'.")

        # Step 3: Transcribe the extracted audio file
        transcription_result = model.transcribe(TEMP_AUDIO_PATH, verbose=False)
        segments = transcription_result.get("segments", [])
        detected_language = transcription_result.get("language", "unknown")

        # Step 4: Merge nearby speech segments to reduce fragmentation
        merged_segments = []
        current_segment = None

        for seg in segments:
            text = seg.get("text", "").strip()
            start = seg.get("start")
            end = seg.get("end")

            # Skip very short or empty segments
            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            if current_segment is None:
                current_segment = {"start": start, "end": end, "text": text}
            else:
                # Merge if gap between segments is small
                if start - current_segment["end"] <= MERGE_THRESHOLD_SEC:
                    current_segment["text"] += " " + text
                    current_segment["end"] = end
                else:
                    merged_segments.append(current_segment)
                    current_segment = {"start": start, "end": end, "text": text}

        # Add the last segment if any
        if current_segment:
            merged_segments.append(current_segment)

        # Step 5: Prepare output events
        speaking_events = []
        for seg in merged_segments:
            speaking_events.append({
                "timestamp_start": format_timestamp(seg["start"]),
                "timestamp_end": format_timestamp(seg["end"]),
                "event_type": "SPEAKING_TO_SOMEONE",
                "text": seg["text"],
                "language": detected_language
            })

        print(f"Transcription complete. Detected {len(speaking_events)} speaking events.")
        return speaking_events

    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return []

    finally:
        # Clean up temporary audio file
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
        print(f"Temporary audio file '{TEMP_AUDIO_PATH}' removed.")


# === Test the speech detection function ===
if __name__ == "__main__":
    test_video = r"\videos\Movie on 17.06.2025 at 14.39.mov"
    detect_speaking_to_someone(test_video)
