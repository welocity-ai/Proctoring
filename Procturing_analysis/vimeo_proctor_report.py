#!/usr/bin/env python3
"""
AI Proctoring Report (Simplified + Proofs) - Parallel Orchestration (Option A)
-------------------------------------------------------------------------------
- Runs Face+Gaze and Object Detection in parallel (processes).
- Extracts audio while those run, then runs Voice Diarization (needs audio).
- Keeps PDF layout, JSON structure, function signatures and returned dicts unchanged.
"""

import os
import uuid
import shutil
import argparse
import subprocess
import time
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# ------------------------------------------------------
# 🌟 GLOBAL CLEAN CONSOLE SETTINGS (Option A)
# ------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

# Silence TensorFlow, Mediapipe, Ultralytics, Torch spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
# ------------------------------------------------------

from src import config
from src.face_and_gaze_analysis import analyze_face_and_gaze
from src.voice_diarization import run_voice_analysis_with_audio_extraction
from src.object_detection import run_object_detection
#from src.object_detection_rf import run_object_detection
from src.report_builder import build_pdf_report
from src.utils import merge_gadget_logs


# ---------------- TIMER HELPERS ---------------- #
def start_timer() -> float:
    return time.perf_counter()

def end_timer(t0: float, label: str = "") -> float:
    dt = round(time.perf_counter() - t0, 4)
    logging.info(f"[TIMER] {label} took: {dt} seconds")
    return dt


# ---------------- Utility ---------------- #
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)





# ---------------- MAIN ---------------- #
def main() -> None:
    # Configure Logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format=config.LOG_FORMAT
    )

    t0_total = start_timer()
    parser = argparse.ArgumentParser(description="AI Proctoring Analyzer")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--outdir", default=config.DEFAULT_OUT_DIR)
    parser.add_argument("--hf-token", default=config.HF_TOKEN)
    parser.add_argument("--object-sample-ms", type=int, default=config.OBJECT_SAMPLE_MS)
    parser.add_argument("--gaze-frame-step", type=int, default=config.GAZE_FRAME_STEP)
    args = parser.parse_args()

    if not args.hf_token:
         logging.error("HuggingFace Token is required. Set HF_TOKEN env var or pass --hf-token.")
         exit(1)

    ensure_dir(args.outdir)
    tmp_dir = os.path.join(args.outdir, "tmp")
    ensure_dir(tmp_dir)

    session_id = str(uuid.uuid4())[:8]
    logging.info(f"Starting analysis for {args.video}")
    logging.info(f"Session ID: {session_id}")
    logging.info(f"Output folder: {args.outdir}")

    face_flags: List[Dict[str, Any]] = []
    gaze_summary: Dict[str, Any] = {}
    voice_segments: List[Dict[str, Any]] = []
    gadget_flags: List[Dict[str, Any]] = []

    executor = None

    try:
        # Fetch video stats
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(args.video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            if cap:
                cap.release()

        duration = f"{round(total_frames / max(fps, 1), 2)}s"

        # -----------------------------------------------------
        # START PARALLEL WORKERS
        # -----------------------------------------------------
        t0_parallel = start_timer()
        # INCREASED WORKERS TO 3 to include Diarization
        executor = ProcessPoolExecutor(max_workers=3)

        logging.info("Submitting Face + Gaze Analysis (worker)...")
        face_future = executor.submit(
            analyze_face_and_gaze,
            args.video,
            args.gaze_frame_step,
            args.outdir
        )

        logging.info("Submitting Gadget Detection (worker)...")
        obj_future = executor.submit(
            run_object_detection,
            args.video,
            args.outdir,
            config.YOLO_MODEL_PATH,
            0.25,
            args.object_sample_ms
        )

        # SUBMIT DIARIZATION TO WORKER (Parallel) - audio extraction inside worker
        logging.info("Submitting Voice Diarization (worker)...")
        t0_voice = start_timer()
        voice_future = executor.submit(
            run_voice_analysis_with_audio_extraction,
            args.video,          # video path (audio will be extracted inside worker)
            tmp_dir,
            args.outdir,
            hf_token=args.hf_token
        )

        # -----------------------------------------------------
        # WAIT FOR PARALLEL PROCESSES
        # -----------------------------------------------------
        logging.info("Waiting for workers to finish...")

        future_to_task = {
            face_future: "face",
            obj_future: "object",
            voice_future: "voice"
        }

        for fut in as_completed(future_to_task):
            task_type = future_to_task[fut]
            res = fut.result()

            if task_type == "face":
                fg_result = res
                face_flags = fg_result.get("face_flag_logs", [])
                gaze_summary = {
                    "total_frames": fg_result.get("total_frames", 0),
                    "processed_samples": fg_result.get("processed_samples", 0),
                    "gaze_frame_step": fg_result.get("gaze_frame_step",
                                                     args.gaze_frame_step),
                    "no_face_frames": fg_result.get("no_face_frames", 0),
                    "multiple_face_frames": fg_result.get("multiple_face_frames", 0),
                    "looking_away_frames": fg_result.get("looking_away_frames", 0),
                    "gaze_accuracy": fg_result.get("gaze_accuracy", "N/A")
                }

            elif task_type == "object":
                unmerged_gadget_logs, _ = res
                logging.info(f"Merging {len(unmerged_gadget_logs)} gadget detections...")
                gadget_flags = merge_gadget_logs(unmerged_gadget_logs)
            
            elif task_type == "voice":
                 voice_raw, _ = res
                 voice_segments = [v for v in voice_raw if v.get("flagged")]
                 end_timer(t0_voice, "Voice Diarization (Worker)")

        end_timer(t0_parallel, "Parallel Face+Gaze + Gadget Detection")

        if not gaze_summary:
            gaze_summary = {
                "total_frames": total_frames,
                "processed_samples": 0,
                "gaze_frame_step": args.gaze_frame_step,
                "no_face_frames": 0,
                "multiple_face_frames": 0,
                "looking_away_frames": 0,
                "gaze_accuracy": "N/A"
            }

        # -----------------------------------------------------
        # BUILD REPORT
        # -----------------------------------------------------
        build_pdf_report(
            args.outdir,
            session_id,
            duration,
            face_flags,
            voice_segments,
            gaze_summary,
            gadget_flags
        )

    except Exception as e:
        logging.critical("Pipeline Failed:")
        logging.critical(traceback.format_exc())

    finally:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logging.info(f"Cleaned up {tmp_dir}")

    end_timer(t0_total, "TOTAL Pipeline Runtime")
    logging.info(f"Proctoring Analysis Complete! Report saved in: {args.outdir}")


if __name__ == "__main__":
    main()

