# import os
# import cv2
# import uuid
# from datetime import timedelta
# from tqdm import tqdm

# from src.roboflow_client import rf_predict_from_frame

# import warnings, logging
# warnings.filterwarnings("ignore")
# logging.getLogger("absl").setLevel(logging.ERROR)
# logging.getLogger("tensorflow").setLevel(logging.ERROR)


# def _secs_to_hhmmss(seconds: float) -> str:
#     td = timedelta(seconds=seconds)
#     total_seconds = int(td.total_seconds())
#     h, m = divmod(total_seconds, 3600)
#     m, s = divmod(m, 60)
#     ms = int((seconds - int(seconds)) * 1000)
#     return f"{h:02}:{m:02}:{s:02}.{ms:03}"


# def run_object_detection(
#     video_path: str,
#     outdir: str,
#     model_name: str = None,       # UNUSED (for compatibility)
#     conf_thresh: float = 0.25,    # RF model already includes confidence
#     frame_step_ms: int = 1000
# ):
#     """
#     Drop-in replacement for your YOLO object_detection.py
#     Uses Roboflow Hosted Inference to detect Phones / Laptops.
#     Returns (logs, total_flags)
#     """

#     print("[INFO] Using Roboflow Hosted Model for gadget detection")

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames == 0:
#         print(f"[WARN] Video {video_path} has 0 frames")
#         cap.release()
#         return [], 0

#     frame_step = max(1, int((frame_step_ms / 1000) * fps))

#     base_dir = os.path.join(outdir, "gadget_detection")
#     fullshot_dir = os.path.join(base_dir, "screenshots")
#     crop_dir = os.path.join(base_dir, "crops")
#     os.makedirs(fullshot_dir, exist_ok=True)
#     os.makedirs(crop_dir, exist_ok=True)

#     logs = []
#     flag_count = 0
#     frame_idx = 0

#     print("[STEP] Scanning frames using Roboflow model...")

#     # --- PROGRESS BAR ---
#     total_steps = total_frames // frame_step
#     pbar = tqdm(
#         total=total_steps,
#         desc="[OBJECT-RF]",
#         unit="frames",
#         position=1,
#         leave=True,
#         dynamic_ncols=True
#     )

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Skip frames
#         if frame_idx % frame_step != 0:
#             frame_idx += 1
#             continue

#         timestamp_s = frame_idx / fps
#         timestamp_h = _secs_to_hhmmss(timestamp_s)

#         # --- RF API CALL ---
#         try:
#             rf_json = rf_predict_from_frame(frame)
#         except Exception as e:
#             print(f"[ERROR] Roboflow inference failed at frame {frame_idx}: {e}")
#             frame_idx += 1
#             pbar.update(1)
#             continue

#         preds = rf_json.get("predictions", [])

#         for p in preds:
#             label = p.get("class", "").lower()
#             conf = float(p.get("confidence", 0))

#             # Convert RF center-based bbox → xyxy
#             x, y, w, h = p["x"], p["y"], p["width"], p["height"]
#             x1 = int(x - w / 2)
#             y1 = int(y - h / 2)
#             x2 = int(x + w / 2)
#             y2 = int(y + h / 2)

#             # Duration end
#             timestamp_s_end = timestamp_s + (frame_step_ms / 1000)
#             duration_s_calc = max(round(timestamp_s_end - timestamp_s, 2), 0.1)

#             flag_count += 1
#             timestamp_safe = timestamp_h.replace(":", "-").replace(".", "_")

#             # Save full screenshot
#             shot_name = f"flag_{flag_count}_{label}_{timestamp_safe}.jpg"
#             shot_path = os.path.join(fullshot_dir, shot_name)
#             cv2.imwrite(shot_path, frame)

#             # Crop
#             crop_img = frame[y1:y2, x1:x2]
#             if crop_img.size == 0:
#                 crop_img = frame
#             crop_name = f"crop_{flag_count}_{label}_{timestamp_safe}.jpg"
#             crop_path = os.path.join(crop_dir, crop_name)
#             cv2.imwrite(crop_path, crop_img)

#             logs.append({
#                 "start": timestamp_h,
#                 "end": _secs_to_hhmmss(timestamp_s_end),
#                 "duration": duration_s_calc,
#                 "type": label,
#                 "confidence": round(conf, 2),
#                 "screenshot": shot_path,
#                 "crop": crop_path
#             })

#         frame_idx += 1
#         pbar.update(1)

#     pbar.close()
#     cap.release()

#     print(f"[INFO] Roboflow detection complete: {flag_count} frames flagged.")
#     print(f"[INFO] Proofs saved at: {base_dir}")

#     return logs, flag_count


# if __name__ == "__main__":
#     import argparse, json
#     parser = argparse.ArgumentParser()
#     parser.add_argument("video")
#     parser.add_argument("--outdir", default="reports/test_rf")
#     parser.add_argument("--frame-step-ms", type=int, default=1000)
#     args = parser.parse_args()

#     logs, count = run_object_detection(args.video, args.outdir)

#     json_path = os.path.join(args.outdir, "gadget_detection", "rf_detection_log.json")
#     os.makedirs(os.path.dirname(json_path), exist_ok=True)
#     with open(json_path, "w") as f:
#         json.dump(logs, f, indent=2)

#     print("Wrote:", json_path)
