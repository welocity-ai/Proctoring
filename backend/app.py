# backend/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fpdf import FPDF
from datetime import datetime, timedelta
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session logs
# Structure:
# {
#   "session_id": {
#       "start_time": "YYYY-MM-DD HH:MM:SS",
#       "flags": ["raw_string", ...],
#       "structured_logs": [
#           {
#               "activity": "keyboard",
#               "start_time": "YYYY-MM-DD HH:MM:SS",
#               "end_time": "YYYY-MM-DD HH:MM:SS",
#               "duration_sec": 0,
#               "count": 1
#           },
#           ...
#       ]
#   }
# }
SESSIONS: dict = {}

def get_formatted_time(dt_str):
    """Converts 'YYYY-MM-DD HH:MM:SS' to 'HH:MM:SS'"""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%H:%M:%S")
    except:
        return dt_str

def save_json_report(session_id: str):
    """Saves the current session data to a JSON file."""
    if session_id in SESSIONS:
        os.makedirs("reports", exist_ok=True)
        json_path = f"reports/{session_id}.json"
        
        session = SESSIONS[session_id]
        start_dt = datetime.strptime(session["start_time"], "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        total_duration = int((now - start_dt).total_seconds())
        
        # Clean up logs for export (remove 'count', format timestamps)
        clean_logs = []
        for log in session["structured_logs"]:
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
        print(f"[JSON SAVED] {json_path}")

def log_flag(session_id: str, event: str, duration_sec: int = 0):
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "start_time": timestamp_str,
            "flags": [], 
            "structured_logs": []
        }
    
    # 1. Add to raw flags list (legacy support)
    SESSIONS[session_id]["flags"].append(event)
    
    # 2. Structured Merging Logic
    logs = SESSIONS[session_id]["structured_logs"]
    merged = False
    
    if logs:
        last_entry = logs[-1]
        last_activity = last_entry["activity"]
        last_end_time = datetime.strptime(last_entry["end_time"], "%Y-%m-%d %H:%M:%S")
        
        # Check if same activity
        # If duration_sec is provided (explicit update from frontend), we force merge/update
        # If not, we check the 3-second window
        
        time_diff = (now - last_end_time).total_seconds()
        
        if last_activity == event:
            if duration_sec > 0:
                # Explicit duration update (e.g. returning from tab switch)
                # We assume this event corresponds to the last one started
                last_entry["end_time"] = timestamp_str
                last_entry["duration_sec"] = duration_sec
                # Correct start time based on duration
                new_start = now - timedelta(seconds=duration_sec)
                last_entry["start_time"] = new_start.strftime("%Y-%m-%d %H:%M:%S")
                last_entry["count"] += 1
                merged = True
                print(f"[UPDATED] {session_id}: {event} (Duration: {duration_sec}s)")
            elif time_diff <= 3.0:
                # Continuous heartbeat merge
                last_entry["end_time"] = timestamp_str
                # Recalculate duration
                start_dt = datetime.strptime(last_entry["start_time"], "%Y-%m-%d %H:%M:%S")
                last_entry["duration_sec"] = int((now - start_dt).total_seconds())
                last_entry["count"] += 1
                merged = True
                print(f"[MERGED] {session_id}: {event} (Duration: {last_entry['duration_sec']}s)")

    if not merged:
        # Create new entry
        # If duration_sec > 0 but we couldn't merge (e.g. first event?), create one with backdated start
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
        print(f"[FLAG] {session_id}: {event}")
    
    # Save JSON immediately for persistence
    save_json_report(session_id)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[CONNECTED] {session_id}")

    # initialize session if not existing
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "flags": [], 
            "structured_logs": []
        }

    try:
        while True:
            data = await websocket.receive_json()
            # Expecting { type: 'event', event: 'keyboard', duration: 5 } from frontend
            msg_type = data.get("type")
            if msg_type == "event":
                event_name = data.get("event", "unknown_event")
                duration = data.get("duration", 0)
                
                # normalize name if needed
                if isinstance(event_name, str):
                    log_flag(session_id, event_name, duration)
                else:
                    log_flag(session_id, json.dumps(event_name), duration)
            else:
                # ignore other types (we aren't receiving frames/audio right now)
                print(f"[WS] {session_id} ignored message type: {msg_type}")
    except WebSocketDisconnect:
        print(f"[DISCONNECTED] {session_id}")
        # do not call websocket.close() here (it's already closed)
    except Exception as e:
        print(f"[WS ERROR] {session_id}: {e}")
    # DO NOT attempt to send or close here — the connection is already closed on disconnect

@app.post("/generate_report/{session_id}")
async def generate_report(session_id: str):
    """
    Generate a PDF report containing the session logs (behavioral flags).
    Returns a FileResponse with the PDF.
    """
    session = SESSIONS.get(session_id, {"flags": [], "structured_logs": []})
    logs = session["structured_logs"]
    total_flags = sum(l["count"] for l in logs)
    
    # Calculate total session duration
    start_time_str = session.get("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    duration_sec = int((datetime.now() - start_dt).total_seconds())
    duration_str = str(timedelta(seconds=duration_sec))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Proctoring Session Report", ln=True)
    pdf.ln(8)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Session ID: {session_id}", ln=True)
    pdf.cell(0, 8, f"Duration: {duration_str}", ln=True)
    pdf.cell(0, 8, f"Total Events Detected: {total_flags}", ln=True)
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
        
        pdf.cell(40, 8, start, 1)
        pdf.cell(40, 8, end, 1)
        pdf.cell(25, 8, duration, 1)
        pdf.cell(85, 8, activity, 1)
        pdf.ln()

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{session_id}_report.pdf"
    pdf.output(report_path)
    print(f"[REPORT GENERATED] {report_path}")
    
    # Also ensure JSON is saved one last time
    save_json_report(session_id)

    # Return PDF as a file response
    return FileResponse(path=report_path, filename=f"{session_id}_report.pdf", media_type="application/pdf")
