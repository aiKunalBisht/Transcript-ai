# audio_processor.py
# Audio/Video Transcript Extraction
#
# Supports: MP4, MP3, WAV, M4A, WebM
# Transcription: Groq Whisper API (free, fast) or local openai-whisper
#
# Install for local: pip install openai-whisper ffmpeg-python
# Groq Whisper: same GROQ_API_KEY, no extra setup needed

import os
import io
import tempfile

def transcribe_audio(audio_bytes: bytes, filename: str, language: str = None) -> dict:
    """
    Transcribes audio/video file to text.
    
    Tries in order:
    1. Groq Whisper API (free, fast, no local install needed)
    2. Local openai-whisper (needs pip install)
    3. Returns error message if neither available
    
    Returns:
        {"text": "...", "language": "ja", "provider": "groq_whisper", "segments": [...]}
    """
    # Try Groq Whisper first (free tier, no install needed)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        try:
            import streamlit as st
            groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if groq_key:
        return _transcribe_groq(audio_bytes, filename, language, groq_key)
    
    # Fall back to local whisper
    return _transcribe_local(audio_bytes, filename, language)


def _transcribe_groq(audio_bytes: bytes, filename: str, 
                     language: str, api_key: str) -> dict:
    """Use Groq Whisper API — free tier, ~30s for 1hr audio."""
    import requests
    
    # Groq whisper endpoint
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    # Determine content type
    ext = filename.lower().split(".")[-1]
    mime_map = {
        "mp4": "video/mp4", "mp3": "audio/mpeg",
        "wav": "audio/wav", "m4a": "audio/mp4",
        "webm": "video/webm", "ogg": "audio/ogg"
    }
    mime = mime_map.get(ext, "audio/mpeg")
    
    try:
        files = {"file": (filename, io.BytesIO(audio_bytes), mime)}
        data  = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "temperature": 0,
        }
        if language:
            data["language"] = language[:2]  # "ja", "en"
        
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "text":      result.get("text", ""),
            "language":  result.get("language", "unknown"),
            "provider":  "groq_whisper",
            "duration":  result.get("duration", 0),
            "segments":  result.get("segments", []),
            "success":   True
        }
    except Exception as e:
        return {
            "text": "",
            "error": str(e),
            "provider": "groq_whisper_failed",
            "success": False
        }


def _transcribe_local(audio_bytes: bytes, filename: str, 
                      language: str) -> dict:
    """Use local openai-whisper — needs pip install openai-whisper."""
    try:
        import whisper
        import numpy as np
        
        # Save to temp file
        ext = filename.lower().split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            model = whisper.load_model("base")  # smallest model
            options = {"language": language[:2]} if language else {}
            result  = model.transcribe(tmp_path, **options)
            
            return {
                "text":     result["text"],
                "language": result.get("language", "unknown"),
                "provider": "local_whisper",
                "segments": result.get("segments", []),
                "success":  True
            }
        finally:
            os.unlink(tmp_path)
            
    except ImportError:
        return {
            "text": "",
            "error": (
                "No transcription provider available. "
                "Add GROQ_API_KEY (free at console.groq.com) for instant transcription, "
                "or run: pip install openai-whisper"
            ),
            "provider": "none",
            "success": False
        }
    except Exception as e:
        return {
            "text": "",
            "error": str(e),
            "provider": "local_whisper_failed",
            "success": False
        }


def format_transcript_with_timestamps(segments: list) -> str:
    """
    Converts Whisper segments to transcript with timestamps.
    
    DIARIZATION NOTE: True speaker diarization (who said what) requires
    pyannote.audio which needs a HuggingFace token. This implementation
    uses silence-gap heuristics to infer speaker turns — not 100% accurate
    but significantly better than a flat transcript for downstream analysis.
    
    Heuristic: gap > 1.5s between segments = likely speaker change
    """
    if not segments:
        return ""

    lines        = []
    speaker_num  = 1
    prev_end     = 0.0
    SPEAKER_GAP  = 1.5  # seconds — gap suggests speaker change

    for seg in segments:
        start = seg.get("start", 0)
        end   = seg.get("end",   start + 1)
        text  = seg.get("text", "").strip()
        if not text:
            continue

        # Infer speaker change from silence gap
        if start - prev_end > SPEAKER_GAP and prev_end > 0:
            speaker_num = (speaker_num % 5) + 1  # cycle through 5 speakers max

        minutes = int(start // 60)
        seconds = int(start % 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] Speaker {speaker_num}: {text}")
        prev_end = end

    return "\n".join(lines)


def format_transcript_simple(text: str) -> str:
    """
    Formats plain Whisper text without segment timestamps.
    Used when verbose_json segments not available.
    """
    return text.strip()


SUPPORTED_FORMATS = {
    "mp4":  "video/mp4",
    "mp3":  "audio/mpeg", 
    "wav":  "audio/wav",
    "m4a":  "audio/mp4",
    "webm": "video/webm",
    "ogg":  "audio/ogg",
    "flac": "audio/flac",
}

MAX_FILE_SIZE_MB = 25  # Groq limit is 25MB


if __name__ == "__main__":
    print("Audio processor ready")
    print(f"Supported formats: {list(SUPPORTED_FORMATS.keys())}")
    print(f"Groq key available: {bool(os.getenv('GROQ_API_KEY'))}")
    print("Local whisper:", end=" ")
    try:
        import whisper
        print("installed")
    except ImportError:
        print("not installed (pip install openai-whisper)")