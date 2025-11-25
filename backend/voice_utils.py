"""
Backward compatibility shim for voice_utils.

This file re-exports functions from the new structured location.
For new code, import directly from src.utils.voice_utils instead.
"""

# Re-export all functions from the new location
from src.utils.voice_utils import (
    add_audio_chunk,
    detect_multiple_speakers,
    audio_buffer,
    pipeline,
)

__all__ = [
    "add_audio_chunk",
    "detect_multiple_speakers",
    "audio_buffer",
    "pipeline",
]
