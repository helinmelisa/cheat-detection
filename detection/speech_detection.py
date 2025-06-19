import whisper
import ffmpeg
import os
import uuid

WHISPER_MODEL_SIZE = "base"
TEMP_AUDIO_PATH = "temp_full_audio.wav"
MIN_TRANSCRIBED_TEXT_LENGTH = 1  

def extract_audio_with_ffmpeg(video_path: str, output_wav_path: str, sample_rate: int = 16000):
    """
    Extract audio from the video file as mono WAV format suitable for Whisper.
    """
    try:
        ffmpeg.input(video_path).output(
            output_wav_path,
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar=sample_rate
        ).run(overwrite_output=True, quiet=True)
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode(errors='ignore') if e.stderr else str(e)}")
        return False


def detect_speaking_to_someone(video_path: str):
    """
    Uses Whisper to transcribe the whole video audio and detect speaking events.
    No VAD — relies entirely on Whisper's timestamped output.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []

    # Extract audio from the video
    audio_extracted = extract_audio_with_ffmpeg(video_path, TEMP_AUDIO_PATH)
    if not audio_extracted:
        return []

    try:
        # Load Whisper model
        model = whisper.load_model(WHISPER_MODEL_SIZE)
        print(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded.")

        # Transcribe with timestamps
        result = model.transcribe(TEMP_AUDIO_PATH, verbose=False)
        segments = result.get("segments", [])

        # Filter and format output
        events = []
        for seg in segments:
            text = seg.get("text", "").strip()
            start = seg.get("start")
            if text and len(text) >= MIN_TRANSCRIBED_TEXT_LENGTH:
                events.append({
                    "timestamp": f"{start:.1f}",
                    "event_type": "SPEAKING_TO_SOMEONE",
                    "text": text
                })

        print(f"Transcription complete. Found {len(events)} speaking events.")
        return events

    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return []

    finally:
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)

# Example usage
if __name__ == "__main__":
    test_video = r"\videos\Movie on 17.06.2025 at 14.39.mov"
    if os.path.exists(test_video):
        results = detect_speaking_to_someone(test_video)
        for event in results:
            print(event)
    else:
        print("Test video path does not exist.")
