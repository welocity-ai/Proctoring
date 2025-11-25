"""Report generation service."""

import os
import logging
from fpdf import FPDF
from typing import Optional
from ..models.session import get_session
from ..config import REPORTS_DIR

logger = logging.getLogger(__name__)


def generate_report(session_id: str) -> Optional[str]:
    """
    Generate a PDF report for a session.
    
    Args:
        session_id: The session ID to generate a report for
        
    Returns:
        Path to the generated PDF file, or None if session not found
    """
    session = get_session(session_id)
    if not session:
        logger.warning(f"Session not found: {session_id}")
        return None
    
    flags = session.flags
    logs = session.logs
    start_time = session.start_time
    end_time = session.end_time or time.time()
    
    duration_sec = int(end_time - start_time) if start_time else 0
    duration_str = time.strftime('%H:%M:%S', time.gmtime(duration_sec))

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Proctoring Report: {session_id}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Duration: {duration_str}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    if not flags:
        pdf.cell(200, 10, txt="No suspicious activity detected.", ln=True)
    else:
        for flag in flags:
            pdf.cell(200, 10, txt=flag, ln=True)
    pdf.ln(8)
    pdf.cell(0, 8, "Session Logs:", ln=True)
    pdf.ln(6)
    
    pdf.set_font("Arial", size=11)
    for log in logs:
        # Multi_cell to avoid encoding issues for PDF; FPDF uses latin-1 internally,
        # so avoid characters that can't be encoded; replace unsupported characters.
        safe_log = log.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 7, safe_log)
    
    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"{session_id}_report.pdf")
    pdf.output(report_path)
    
    logger.info(f"[REPORT GENERATED] {report_path}")
    return report_path

    #soc2 gdpr compliant. 
    #can we use yolov8, aws system
    

