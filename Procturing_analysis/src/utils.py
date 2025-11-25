import os
import cv2
from datetime import timedelta

# ----------------------------
# Directory & File Utilities
# ----------------------------

def ensure_dir(path: str):
    """Creates a directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_image(path, image):
    """Utility to save a proof image in PNG format."""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)
    return path

# ----------------------------
# Time Formatting
# ----------------------------

def secs_to_hhmmss(s: float) -> str:
    """
    Converts seconds (float) into H:MM:SS.mmm format.
    Example: 90.5 -> "0:01:30.500"
    """
    if s < 0:
        s = 0
    return str(timedelta(seconds=s))[:-3]

# Backwards compatibility
def _secs_to_hhmmss(s: float) -> str:
    return secs_to_hhmmss(s)

# ----------------------------
# Voice Log Merging
# ----------------------------

def merge_voice_segments(segments: list, max_gap_s: float = 2.0) -> list:
    if not segments:
        return []

    def parse_time(t_str):
        """Convert H:MM:SS.mmm string into seconds float."""
        parts = t_str.split(':')
        sec_ms = parts[2].split('.')
        return timedelta(
            hours=int(parts[0]),
            minutes=int(parts[1]),
            seconds=int(sec_ms[0]),
            milliseconds=int(sec_ms[1])
        ).total_seconds()

    segments.sort(key=lambda x: (x["start"], x["speaker"]))
    merged, current = [], segments[0].copy()

    for nxt in segments[1:]:
        try:
            gap = parse_time(nxt["start"]) - parse_time(current["end"])
        except Exception:
            gap = max_gap_s + 1

        if nxt["speaker"] == current["speaker"] and 0 <= gap <= max_gap_s:
            current["end"] = nxt["end"]
            current["duration"] = parse_time(current["end"]) - parse_time(current["start"])
        else:
            merged.append(current)
            current = nxt.copy()

    merged.append(current)
    return merged

# ----------------------------
# Gadget Log Merging
# ----------------------------

def merge_gadget_logs(logs: list, max_gap_s: float = 0.5) -> list:
    """Merge adjacent gadget detections into continuous events."""
    if not logs:
        return []

    def to_seconds(t_str):
        try:
            parts = t_str.split(':')
            sec_ms = parts[2].split('.')
            return timedelta(
                hours=int(parts[0]),
                minutes=int(parts[1]),
                seconds=int(sec_ms[0]),
                milliseconds=int(sec_ms[1])
            ).total_seconds()
        except Exception:
            return 0.0

    merged, current = [], logs[0].copy()

    for nxt in logs[1:]:
        gap = to_seconds(nxt['start']) - to_seconds(current['end'])

        if nxt['type'] == current['type'] and gap <= max_gap_s:
            current['end'] = nxt['end']
            current['duration'] = to_seconds(current['end']) - to_seconds(current['start'])
        else:
            merged.append(current)
            current = nxt.copy()

    merged.append(current)
    return merged
