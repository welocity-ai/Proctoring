"""Session data models."""

from typing import List, Dict, Any
from datetime import datetime


class Session:
    """Represents a proctoring session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.flags: List[str] = []
        self.logs: List[str] = []
        self.structured_logs: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_flag(self, event: str) -> None:
        """Add a flag event to the session."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.flags.append(event)
        self.logs.append(f"[{timestamp}] {event}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "flags": self.flags,
            "logs": self.logs,
            "structured_logs": self.structured_logs,
            "total_flags": len(self.flags),
            "created_at": self.created_at.isoformat(),
            "start_time": self.start_time
        }



# Global session storage (in-memory)
SESSIONS: Dict[str, Session] = {}

def get_or_create_session(session_id: str) -> Session:
    """Return the Session object for a given session_id, create one if not exists."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session(session_id)
    return SESSIONS[session_id]

def get_session(session_id: str) -> Any:
    """Return the Session object for a given session_id, or None."""
    return SESSIONS.get(session_id)


