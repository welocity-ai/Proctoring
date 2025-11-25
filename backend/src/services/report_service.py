"""Report generation service."""

import os
import logging
from datetime import datetime, timedelta
from fpdf import FPDF
from typing import Optional
from ..models.session import get_session
from ..config import REPORTS_DIR
from .session_service import save_json_report, get_formatted_time

logger = logging.getLogger(__name__)


def generate_report(session_id: str) -> Optional[str]:
    """
    Generate a PDF report for a session with structured activity logs.
    
    Args:
        session_id: The session ID to generate a report for
        
    Returns:
        Path to the generated PDF file, or None if session not found
    """
    session = get_session(session_id)
    if not session:
        logger.warning(f"Session not found: {session_id}")
        return None
    
    logs = session.structured_logs
    total_flags = sum(log.get("count", 1) for log in logs)
    
    # Calculate total session duration
    start_dt = datetime.strptime(session.start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.now()
    duration_sec = int((end_time - start_dt).total_seconds())
    duration_str = str(timedelta(seconds=duration_sec))

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt="AI Proctoring Session Report", ln=True, align='C')
    pdf.ln(8)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, txt=f"Session ID: {session_id}", ln=True)
    pdf.cell(0, 8, txt=f"Duration: {duration_str}", ln=True)
    pdf.cell(0, 8, txt=f"Total Events Detected: {total_flags}", ln=True)
    pdf.ln(8)
    
    # Table Header
    pdf.set_font("Arial", "B", 11)
    pdf.cell(40, 8, "From", 1)
    pdf.cell(40, 8, "To", 1)
    pdf.cell(25, 8, "Duration", 1)
    pdf.cell(85, 8, "Activity", 1)
    pdf.ln()

    # Table Rows
    pdf.set_font("Arial", size=10)
    for log in logs:
        start = get_formatted_time(log["start_time"])
        end = get_formatted_time(log["end_time"])
        duration = f"{log['duration_sec']}s"
        activity = log["activity"]
        
        # Encode to latin-1 to handle PDF encoding
        safe_activity = activity.encode("latin-1", "replace").decode("latin-1")
        
        pdf.cell(40, 8, start, 1)
        pdf.cell(40, 8, end, 1)
        pdf.cell(25, 8, duration, 1)
        pdf.cell(85, 8, safe_activity, 1)
        pdf.ln()
    
    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"{session_id}_report.pdf")
    pdf.output(report_path)
    
    logger.info(f"[REPORT GENERATED] {report_path}")
    
    # Also ensure JSON is saved one last time
    save_json_report(session_id)
    
    return report_path
