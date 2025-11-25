"""Session management service."""

import logging
import json
import os
from datetime import datetime, timedelta
from ..models.session import get_or_create_session
from ..config import REPORTS_DIR

logger = logging.getLogger(__name__)

def get_formatted_time(dt_str: str) -> str:
    """Converts 'YYYY-MM-DD HH:MM:SS' to 'HH:MM:SS'"""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%H:%M:%S")
    except:
        return dt_str

def save_json_report(session_id: str) -> None:
    """Saves the current session data to a JSON file."""
    session = get_or_create_session(session_id)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    json_path = os.path.join(REPORTS_DIR, f"{session_id}.json")
    
    start_dt = datetime.strptime(session.start_time, "%Y-%m-%d %H:%M:%S")
    now = datetime.now()
    total_duration = int((now - start_dt).total_seconds())
    
    # Clean up logs for export (format timestamps)
    clean_logs = []
    for log in session.structured_logs:
        clean_logs.append({
            "activity": log["activity"],
            "start_time": get_formatted_time(log["start_time"]),
            "end_time": get_formatted_time(log["end_time"]),
            "duration_sec": log["duration_sec"]
        })

    # Create a clean export format
    export_data = {
        "session_id": session_id,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "session_duration_sec": total_duration,
        "activity_log": clean_logs
    }
    
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2)
    logger.info(f"[JSON SAVED] {json_path}")

def log_flag(session_id: str, event: str, duration_sec: int = 0) -> None:
    """Log a flag event for a session with structured logging and merging."""
    session = get_or_create_session(session_id)
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Add to raw flags list (legacy support)
    session.flags.append(event)
    
    # 2. Structured Merging Logic
    logs = session.structured_logs
    merged = False
    
    if logs:
        last_entry = logs[-1]
        last_activity = last_entry["activity"]
        last_end_time = datetime.strptime(last_entry["end_time"], "%Y-%m-%d %H:%M:%S")
        
        time_diff = (now - last_end_time).total_seconds()
        
        if last_activity == event:
            if duration_sec > 0:
                # Explicit duration update (e.g. returning from tab switch)
                last_entry["end_time"] = timestamp_str
                last_entry["duration_sec"] = duration_sec
                # Correct start time based on duration
                new_start = now - timedelta(seconds=duration_sec)
                last_entry["start_time"] = new_start.strftime("%Y-%m-%d %H:%M:%S")
                last_entry["count"] += 1
                merged = True
                logger.info(f"[UPDATED] {session_id}: {event} (Duration: {duration_sec}s)")
            elif time_diff <= 3.0:
                # Continuous heartbeat merge
                last_entry["end_time"] = timestamp_str
                # Recalculate duration
                start_dt = datetime.strptime(last_entry["start_time"], "%Y-%m-%d %H:%M:%S")
                last_entry["duration_sec"] = int((now - start_dt).total_seconds())
                last_entry["count"] += 1
                merged = True
                logger.info(f"[MERGED] {session_id}: {event} (Duration: {last_entry['duration_sec']}s)")

    if not merged:
        # Create new entry
        start_ts = timestamp_str
        if duration_sec > 0:
            start_ts = (now - timedelta(seconds=duration_sec)).strftime("%Y-%m-%d %H:%M:%S")

        new_entry = {
            "activity": event,
            "start_time": start_ts,
            "end_time": timestamp_str,
            "duration_sec": duration_sec,
            "count": 1
        }
        logs.append(new_entry)
        logger.info(f"[FLAG] {session_id}: {event}")
    
    # 3. Add to simple logs as well
    log_entry = f"[{timestamp_str}] {event}"
    if duration_sec > 0:
        log_entry += f" (Duration: {duration_sec}s)"
    session.logs.append(log_entry)
    
    # Save JSON immediately for persistence
    save_json_report(session_id)
