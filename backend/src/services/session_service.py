"""Session management service."""

import logging
import time
from datetime import datetime, timedelta
from ..models.session import get_or_create_session

logger = logging.getLogger(__name__)

def log_flag(session_id: str, event: str, duration_sec: int = 0) -> None:
    """Log a flag event for a session."""
    session = get_or_create_session(session_id)
    
    # Update end time on every activity to track "last active"
    # Note: Session class uses datetime, so we should be consistent if possible, 
    # but the original code mixed time.time() and datetime. 
    # Let's stick to the Session class structure which uses lists.
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Logic from original app.py to handle duration and merging
    # 1. Add to raw flags list
    session.flags.append(event)
    
    # 2. Structured Merging Logic (simulated for now by appending to logs with duration)
    # In a full refactor we might want structured_logs in the Session class, 
    # but for now we will format the log string to include duration if > 0
    
    log_entry = f"[{timestamp}] {event}"
    if duration_sec > 0:
        log_entry += f" (Duration: {duration_sec}s)"
        
    session.logs.append(log_entry)
    logger.info(f"[FLAG] {session_id}: {event} (Duration: {duration_sec}s)")
