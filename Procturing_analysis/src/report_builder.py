import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fpdf import FPDF

from .utils import ensure_dir

# ---------------- PDF Builder ---------------- #
class ReportPDF(FPDF):

    def header(self) -> None:
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "AI Proctoring Report",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

    def section_title(self, title: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def log_row(self, cols: List[str], widths: List[int], bold: bool = False) -> None:
        self.set_font("Helvetica", "B" if bold else "", 10)
        for text, width in zip(cols, widths):
            self.cell(width, 8, str(text), border=1, align="C")
        self.ln(8)

    def add_summary_line(self, key: str, value: Any) -> None:
        self.set_font("Helvetica", "B", 11)
        self.cell(60, 8, str(key))
        self.set_font("Helvetica", "", 11)
        self.cell(0, 8, str(value),
                  new_x="LMARGIN", new_y="NEXT")


# ---------------- PDF Report ---------------- #
def build_pdf_report(outdir: str, session_id: str, duration: str,
                     face_flags: List[Dict[str, Any]], 
                     voice_segments: List[Dict[str, Any]],
                     gaze_summary: Dict[str, Any], 
                     gadget_flags: List[Dict[str, Any]]) -> None:

    logging.info("Building PDF and JSON reports...")

    pdf = ReportPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Session ID: {session_id}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Video Duration: {duration}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {datetime.now()}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Gaze Summary
    pdf.section_title("Gaze Analysis Summary")
    for k in [
        ("Gaze Accuracy (on-screen):", 'gaze_accuracy'),
        ("Total Frames:", 'total_frames'),
        ("Gaze Sample Rate:", 'gaze_frame_step'),
        ("Frames Processed for Gaze:", 'processed_samples'),
        ("Looking Away Frames:", 'looking_away_frames'),
        ("No Face Frames:", 'no_face_frames'),
        ("Multiple Face Frames:", 'multiple_face_frames')
    ]:
        pdf.add_summary_line(k[0], gaze_summary.get(k[1], 0))
    pdf.ln(8)

    # Face Flags
    pdf.section_title("Face Detection Flags (Proofs)")
    pdf.log_row(["Timestamp", "Type", "Proof"],
                [50, 70, 70], bold=True)
    if not face_flags:
        pdf.log_row(["-", "No issues detected", "-"],
                    [50, 70, 70])
    else:
        for f in face_flags[:10]:
            pdf.log_row([
                f.get("timestamp", "-"),
                f.get("reason", "-"),
                os.path.basename(f.get("proof_image", "-"))
            ], [50, 70, 70])
    pdf.ln(8)

    # Voice Diarization
    pdf.section_title("Suspicious Voice Detection")
    pdf.log_row(["Start", "End", "Duration(s)", "Speaker",
                 "Proof"], [35, 35, 35, 45, 50], bold=True)
    if not voice_segments:
        pdf.log_row(["-", "-", "-", "-", "No suspicious voices"],
                    [35, 35, 35, 45, 50])
    else:
        for seg in voice_segments:
            pdf.log_row([
                seg.get("start", "N/A"),
                seg.get("end", "N/A"),
                round(seg.get("duration", 0), 2),
                seg.get("speaker", "Unknown"),
                os.path.basename(seg.get("proof_audio", "-"))
            ], [35, 35, 35, 45, 50])
    pdf.ln(8)

    # Gadget detection
    pdf.section_title("Electronic Gadget Detection")
    pdf.log_row(["Start", "End", "Duration(s)", "Type"],
                [40, 40, 40, 70], bold=True)
    if not gadget_flags:
        pdf.log_row(["-", "-", "-", "No gadgets detected"],
                    [40, 40, 40, 70])
    else:
        for g in gadget_flags:
            pdf.log_row([
                g.get("start", "-"),
                g.get("end", "-"),
                round(g.get("duration", 0), 2),
                g.get("type", "-")
            ], [40, 40, 40, 70])

    # Save
    ensure_dir(outdir)
    pdf_path = os.path.join(outdir, "report.pdf")
    pdf.output(pdf_path)
    logging.info(f"PDF saved: {pdf_path}")

    json_path = os.path.join(outdir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "session_id": session_id,
            "duration": duration,
            "gaze_summary": gaze_summary,
            "face_flags": face_flags,
            "voice_segments": voice_segments,
            "gadget_flags": gadget_flags
        }, f, indent=2)

    logging.info(f"JSON summary saved: {json_path}")