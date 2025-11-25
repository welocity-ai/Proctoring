import cv2
import mediapipe as mp
import os
import logging
from tqdm import tqdm
from datetime import timedelta
from collections import deque
from typing import List, Tuple, Dict, Any, Union
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)


# Landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_VERT = [159, 145]
RIGHT_EYE_VERT = [386, 374]


def _avg(landmarks: Any, idxs: List[int], axis: str = "x") -> float:
    if axis == "x":
        return sum(landmarks[i].x for i in idxs) / len(idxs)
    else:
        return sum(landmarks[i].y for i in idxs) / len(idxs)


def _secs_to_hhmmss_ms(sec: float) -> str:
    td = timedelta(seconds=sec)
    total = td.total_seconds()
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _is_gaze_centered_article(landmarks: Any,
                              horiz_thresh: float = 0.25,
                              vert_thresh: float = 0.25) -> bool:

    def iris_center(ids: List[int]) -> Tuple[float, float]:
        x = sum(landmarks[i].x for i in ids) / len(ids)
        y = sum(landmarks[i].y for i in ids) / len(ids)
        return x, y

    left_ix, left_iy = iris_center(LEFT_IRIS)
    right_ix, right_iy = iris_center(RIGHT_IRIS)

    def eye_bounds(ids: List[int]) -> Tuple[float, float, float, float]:
        xs = [landmarks[i].x for i in ids]
        ys = [landmarks[i].y for i in ids]
        return min(xs), max(xs), min(ys), max(ys)

    lx_min, lx_max, ly_min, ly_max = eye_bounds(LEFT_EYE_CORNERS + LEFT_EYE_VERT)
    rx_min, rx_max, ry_min, ry_max = eye_bounds(RIGHT_EYE_CORNERS + RIGHT_EYE_VERT)

    left_h = (left_ix - lx_min) / max(1e-6, (lx_max - lx_min))
    right_h = (right_ix - rx_min) / max(1e-6, (rx_max - rx_min))

    left_v = (left_iy - ly_min) / max(1e-6, (ly_max - ly_min))
    right_v = (right_iy - ry_min) / max(1e-6, (ry_max - ry_min))

    avg_h = (left_h + right_h) / 2.0
    avg_v = (left_v + right_v) / 2.0

    center_h = (0.5 - horiz_thresh <= avg_h <= 0.5 + horiz_thresh)
    center_v = (0.5 - vert_thresh <= avg_v <= 0.5 + vert_thresh)

    return center_h and center_v


def analyze_face_and_gaze(video_path: str,
                          gaze_frame_step: int = 1,
                          outdir: str = "reports/tmp") -> Dict[str, Any]:

    os.makedirs(outdir, exist_ok=True)
    proof_dir = os.path.join(outdir, "face_proofs")
    os.makedirs(proof_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    screenshot_interval = 1.0
    last_shot_time = -1.0

    mp_face_mesh = mp.solutions.face_mesh
    mp_face_det = mp.solutions.face_detection

    no_face_frames = 0
    multiple_face_frames = 0
    single_face_frames = 0
    looking_away_frames = 0

    face_flag_logs: List[Dict[str, Any]] = []

    gaze_buffer: deque = deque(maxlen=5)

    # Vimeo-optimized multi-face buffer
    multi_face_buffer: deque = deque(maxlen=7)    # <── BEST SETTING

    processed_samples = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        static_image_mode=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as mesh, mp_face_det.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.4
    ) as detector:

        frame_idx = 0
       # Detect if running inside ProcessPoolExecutor worker
        IS_WORKER = multiprocessing.current_process().name != "MainProcess"

        pbar = tqdm(
            total=total_frames,
            desc="[FACE+GAZE]",
            unit="frames",
            position=0,        # fixed row
            leave=True,
            dynamic_ncols=True,
            disable=IS_WORKER   # <── KEY FIX: disable tqdm inside workers
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update(1)

            if frame_idx % gaze_frame_step != 0:
                frame_idx += 1
                continue

            processed_samples += 1
            timestamp_s = frame_idx / fps
            timestamp = _secs_to_hhmmss_ms(timestamp_s)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = detector.process(rgb)
            detections = det.detections or []

            threshold = (gaze_buffer.maxlen // 2) + 1

            # ------------------- NO FACE --------------------
            if len(detections) == 0:
                no_face_frames += 1
                gaze_buffer.append(True)

                centered = (sum(gaze_buffer) >= threshold)

                if not centered:
                    looking_away_frames += 1

                if timestamp_s - last_shot_time >= screenshot_interval:
                    last_shot_time = timestamp_s
                    path = os.path.join(
                        proof_dir, f"noface_{timestamp.replace(':','-')}.jpg"
                    )
                    cv2.imwrite(path, frame)
                    face_flag_logs.append({
                        "timestamp": timestamp,
                        "reason": "No Face",
                        "proof_image": path
                    })

                frame_idx += 1
                continue

            # ---------------- MULTIPLE FACES (VIMEO-OPTIMIZED) ----------------
            multi_face_buffer.append(len(detections) > 1)
            stable_multiface = sum(multi_face_buffer) >= 3    # <── BEST SETTING

            if stable_multiface:
                multiple_face_frames += 1
                gaze_buffer.append(False)

                centered = (sum(gaze_buffer) >= threshold)
                if not centered:
                    looking_away_frames += 1

                if timestamp_s - last_shot_time >= screenshot_interval:
                    last_shot_time = timestamp_s
                    path = os.path.join(
                        proof_dir,
                        f"multiface_{timestamp.replace(':','-')}.jpg"
                    )
                    cv2.imwrite(path, frame)
                    face_flag_logs.append({
                        "timestamp": timestamp,
                        "reason": f"Multiple Faces ({len(detections)})",
                        "proof_image": path
                    })

                frame_idx += 1
                continue

            # ---------------- SINGLE FACE ----------------
            single_face_frames += 1

            fm = mesh.process(rgb)
            if not fm.multi_face_landmarks:
                gaze_buffer.append(True)
                centered = (sum(gaze_buffer) >= threshold)
                if not centered:
                    looking_away_frames += 1
                frame_idx += 1
                continue

            landmarks = fm.multi_face_landmarks[0].landmark

            centered = _is_gaze_centered_article(landmarks)
            gaze_buffer.append(centered)

            if sum(gaze_buffer) < threshold:
                looking_away_frames += 1

            frame_idx += 1

        pbar.close()
        cap.release()

    if processed_samples == 0:
        accuracy = 0.0
    else:
        accuracy = 100 * (1 - looking_away_frames / processed_samples)
        accuracy = round(accuracy, 2)

    return {
        "total_frames": total_frames,
        "fps": fps,
        "gaze_frame_step": gaze_frame_step,
        "processed_samples": processed_samples,
        "looking_away_frames": looking_away_frames,
        "no_face_frames": no_face_frames,
        "multiple_face_frames": multiple_face_frames,
        "single_face_frames": single_face_frames,
        "face_flag_logs": face_flag_logs,
        "gaze_accuracy": f"{accuracy:.2f}%"
    }

