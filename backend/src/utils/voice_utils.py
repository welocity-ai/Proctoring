"""Voice diarization utilities."""

import logging
from typing import Optional
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

# Initialize pretrained diarization model
# (Requires HuggingFace token if using pyannote pretrained)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", hf_token="")

# Buffer for collecting small chunks of audio
audio_buffer = []


async def add_audio_chunk(chunk_bytes: bytes) -> None:
    """Append audio chunk to global buffer for processing."""
    audio_buffer.append(chunk_bytes)
    if len(audio_buffer) > 5:  # keep last 5 chunks (~5s)
        audio_buffer.pop(0)


async def detect_multiple_speakers(session_id: str) -> Optional[bool]:
    """Analyze audio buffer every few seconds."""
    if not audio_buffer:
        return None

    # Combine chunks into one numpy array
    wav_data = b"".join(audio_buffer)
    try:
        # Write temporary file
        path = f"temp_audio_{session_id}.wav"
        with open(path, "wb") as f:
            f.write(wav_data)

        # Run diarization
        diarization = pipeline(path)
        speakers = set([seg.track for seg in diarization.itertracks()])
        speaker_count = len(speakers)

        if speaker_count > 1:
            logger.warning(f"[VOICE ALERT] {session_id}: multiple voices detected")
            return True
    except Exception as e:
        logger.error(f"Voice detection error: {e}")
    return False

