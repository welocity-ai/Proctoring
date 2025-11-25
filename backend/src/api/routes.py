"""API routes for the proctoring application."""

import json
import logging
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from ..models.session import get_or_create_session
from ..services.session_service import log_flag
from ..services.report_service import generate_report

logger = logging.getLogger(__name__)


from .schemas import SessionEvent
from pydantic import ValidationError

async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Handle WebSocket connections for proctoring events."""
    await websocket.accept()
    logger.info(f"[CONNECTED] {session_id}")
    
    # Initialize session if not existing
    get_or_create_session(session_id)
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
                # Validate with Pydantic
                event_data = SessionEvent(**data)
                
                if event_data.type == "event":
                    event_name = event_data.event or "unknown_event"
                    duration = event_data.duration or 0
                    
                    # Normalize name if needed
                    if isinstance(event_name, str):
                        log_flag(session_id, event_name, duration)
                    else:
                        log_flag(session_id, json.dumps(event_name), duration)
                else:
                    # Ignore other types (we aren't receiving frames/audio right now)
                    logger.debug(f"[WS] {session_id} ignored message type: {event_data.type}")
            except ValidationError as e:
                logger.warning(f"[WS] {session_id} invalid message format: {e}")
                continue
                
    except WebSocketDisconnect:
        logger.info(f"[DISCONNECTED] {session_id}")
        # Do not call websocket.close() here (it's already closed)
    except Exception as e:
        logger.error(f"[WS ERROR] {session_id}: {e}")
    # DO NOT attempt to send or close here — the connection is already closed on disconnect


async def generate_report_endpoint(session_id: str) -> FileResponse:
    """
    Generate a PDF report containing the session logs (behavioral flags).
    Returns a FileResponse with the PDF.
    """
    report_path = generate_report(session_id)
    if not report_path:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return FileResponse(
        path=report_path,
        filename=f"{session_id}_report.pdf",
        media_type="application/pdf"
    )

