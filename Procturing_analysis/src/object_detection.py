import os
import cv2
import logging
import torch
from datetime import timedelta
from tqdm import tqdm
from ultralytics import YOLO

# Try relative import first (for package usage), fallback to direct import (for script usage)
try:
    from . import config
except ImportError:
    import config

# Configure logger
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # FIX 1: MPS does not support half precision with YOLOv8, only CUDA does safely.
        self.half = (self.device == "cuda")

    def load_model(self):
        if self.model is None:
            logger.info(f"[YOLO-PHONE] Loading model from {self.model_path} on {self.device}...")
            
            # Resolve path if it's relative
            if not os.path.isabs(self.model_path):
                 base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
                 self.model_path = os.path.join(base_dir, self.model_path)

            self.model = YOLO(self.model_path)
            # FIX 2: YOLOv8 handles device movement internally, explicit .to() is redundant/harmful
            # self.model.to(self.device) 

    def _secs_to_hhmmss(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total = int(td.total_seconds())
        h, m = divmod(total, 3600)
        m, s = divmod(m, 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"

    def detect(self, video_path: str, outdir: str, conf_thresh: float = 0.60, frame_step_ms: int = None):
        self.load_model()
        
        frame_step_ms = frame_step_ms or config.OBJECT_SAMPLE_MS
        
        logger.info(f"[INFO] Using YOLO Phone Detector for gadget detection on {video_path}")
        logger.info(f"[CONFIG] Resolution: 640px | Conf Thresh: {conf_thresh} | Device: {self.device} | Half: {self.half}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get video dimensions for area filtering
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_area = width * height
        min_area = total_area * 0.02 # 1% of screen area

        if total_frames == 0:
            logger.warning(f"[WARN] Video {video_path} has 0 frames")
            cap.release()
            return [], 0

        frame_step = max(1, int((frame_step_ms / 1000) * fps))

        base_dir = os.path.join(outdir, "gadget_detection")
        fullshot_dir = os.path.join(base_dir, "screenshots")
        crop_dir = os.path.join(base_dir, "crops")
        os.makedirs(fullshot_dir, exist_ok=True)
        os.makedirs(crop_dir, exist_ok=True)

        logs = []
        flag_count = 0
        frame_idx = 0

        logger.info(f"[STEP] Scanning frames using YOLO model... (Min Area: {int(min_area)} px)")

        total_steps = total_frames // frame_step
        pbar = tqdm(
            total=total_steps,
            desc="[OBJECT-YOLO]",
            unit="frames",
            position=1,
            leave=True,
            dynamic_ncols=True
        )

        while True:
            # Optimization: Skip frames using grab()
            if frame_idx % frame_step != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            # Read target frame
            ret, frame = cap.read()
            if not ret:
                break

            # FIX 3: Use actual timestamp from decoder to avoid drift
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms > 0:
                timestamp_s = timestamp_ms / 1000.0
            else:
                # Fallback if POS_MSEC is not supported/reliable
                timestamp_s = frame_idx / fps
            
            timestamp_h = self._secs_to_hhmmss(timestamp_s)

            # YOLO INFERENCE
            try:
                results = self.model(frame, imgsz=640, half=self.half, verbose=False)
            except Exception as e:
                logger.error(f"[ERROR] YOLO inference failed at frame {frame_idx}: {e}")
                frame_idx += 1
                pbar.update(1)
                continue

            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_thresh:
                        continue

                    # YOLO xyxy -> ints
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # HEURISTIC: Area Filter
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h
                    
                    if box_area < min_area:
                        # Skip small detections (likely noise or mouth)
                        continue

                    # Get label safely
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id] if self.model.names else "phone"

                    timestamp_s_end = timestamp_s + (frame_step_ms / 1000)
                    duration_s_calc = max(round(timestamp_s_end - timestamp_s, 2), 0.1)

                    flag_count += 1
                    timestamp_safe = timestamp_h.replace(":", "-").replace(".", "_")

                    # Save full screenshot
                    shot_name = f"flag_{flag_count}_{label}_{timestamp_safe}.jpg"
                    shot_path = os.path.join(fullshot_dir, shot_name)
                    cv2.imwrite(shot_path, frame)

                    # Save crop
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        crop_img = frame
                    crop_name = f"crop_{flag_count}_{label}_{timestamp_safe}.jpg"
                    crop_path = os.path.join(crop_dir, crop_name)
                    cv2.imwrite(crop_path, crop_img)

                    logs.append({
                        "start": timestamp_h,
                        "end": self._secs_to_hhmmss(timestamp_s_end),
                        "duration": duration_s_calc,
                        "type": label,
                        "confidence": round(conf, 2),
                        "screenshot": shot_path,
                        "crop": crop_path
                    })

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        logger.info(f"[INFO] YOLO detection complete: {flag_count} frames flagged.")
        logger.info(f"[INFO] Proofs saved at: {base_dir}")

        return logs, flag_count

# Global instance for easy usage
_detector = ObjectDetector()

def run_object_detection(
    video_path: str,
    outdir: str,
    model_name: str = None,         # kept for compatibility
    conf_thresh: float = 0.60,      # Increased default threshold
    frame_step_ms: int = 1000
):
    """
    Wrapper for backward compatibility.
    """
    return _detector.detect(video_path, outdir, conf_thresh, frame_step_ms)

if __name__ == "__main__":
    import argparse, json
    
    # Setup basic logging for standalone run
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--outdir", default="reports/test_yolo")
    parser.add_argument("--frame-step-ms", type=int, default=1000)
    args = parser.parse_args()

    logs, count = run_object_detection(args.video, args.outdir, frame_step_ms=args.frame_step_ms)

    json_path = os.path.join(args.outdir, "gadget_detection", "yolo_detection_log.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(logs, f, indent=2)

    print("Wrote:", json_path)
