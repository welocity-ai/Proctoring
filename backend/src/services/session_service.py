"""Session management service."""

import logging
from typing import Optional
from ..models.session import Session, SESSIONS

logger = logging.getLogger(__name__)


# In-memory store: { session_id: { 'flags': [], 'start_time': float, 'end_time': float } }
# This new 'sessions' dictionary will replace the usage of the 'SESSIONS' global and the 'Session' class
# for tracking session data directly within this service.
_sessions_data = {}


def get_or_create_session(session_id: str) -> dict:
    """Initialize a session if it doesn't exist, and return its data dictionary."""
    if session_id not in _sessions_data:
        _sessions_data[session_id] = {
            "flags": [],
            "start_time": time.time(),
            "end_time": None
        }
        logger.info(f"Created new session: {session_id}")
    return _sessions_data[session_id]


def get_session(session_id: str) -> Optional[dict]:
    """Get a session's data dictionary by ID."""
    return _sessions_data.get(session_id)


def log_flag(session_id: str, event: str) -> None:
    """Log a flag event for a session."""
    session_data = get_or_create_session(session_id)
    # Update end time on every activity to track "last active"
    session_data["end_time"] = time.time()
    
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    session_data["flags"].append(f"[{timestamp}] {event}")
    logger.info(f"[FLAG] {session_id}: {event}")

```
