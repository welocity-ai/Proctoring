import cv2
import numpy as np
import base64
import logging
from typing import Tuple
import mediapipe as mp

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load MediaPipe Face Detector
# -----------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Model selection:
# 0: short-range (0–2 meters), 1: full-range (2+ meters)
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# -----------------------------
# Detect Faces from JPEG Bytes
# -----------------------------
def detect_faces_bboxes(jpeg_bytes: bytes) -> Tuple[np.ndarray, list]:
    """Return BGR image and list of bounding boxes (x, y, w, h) using MediaPipe."""
    if not jpeg_bytes:
        raise ValueError("Empty JPEG bytes provided")

    try:
        # Decode bytes → OpenCV BGR image
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPEG image")

        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run MediaPipe detector
        results = face_detector.process(rgb_img)
        faces = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)

                # Clamp bbox to image bounds
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                if bw > 0 and bh > 0:
                    faces.append((x, y, bw, bh))

        return img, faces

    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        raise

# -----------------------------
# Crop and Perceptual Hash
# -----------------------------
def crop_and_phash(img: np.ndarray, bbox) -> str:
    """Crop bbox from BGR image and compute a simple perceptual hash string."""
    try:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox dimensions: {bbox}")

        img_h, img_w = img.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            raise ValueError(f"Empty crop from bbox: {bbox}")

        # Convert to grayscale and resize
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16,16), interpolation=cv2.INTER_AREA)
        avg = small.mean()
        bits = (small > avg).astype(int).flatten()

        # Convert bits to hex string
        hexstr = "".join([
            format(int("".join(map(str, bits[i:i+8])), 2), "02x")
            for i in range(0, len(bits), 8)
        ])
        return hexstr
    except Exception as e:
        logger.error(f"Error computing phash: {e}")
        raise

# -----------------------------
# Convert Base64 → JPEG Bytes
# -----------------------------
def jpeg_bytes_from_base64(b64: str) -> bytes:
    """Convert base64 string to JPEG bytes."""
    if not b64:
        raise ValueError("Empty base64 string provided")

    try:
        header, data = (b64.split(',',1) if ',' in b64 else (None, b64))
        decoded = base64.b64decode(data)
        if not decoded:
            raise ValueError("Failed to decode base64 data")
        return decoded
    except Exception as e:
        logger.error(f"Error decoding base64: {e}")
        raise

# -----------------------------
# Save JPEG Bytes to File
# -----------------------------
def save_jpeg_bytes(jpeg_bytes: bytes, path: str):
    """Save JPEG bytes to file."""
    if not jpeg_bytes:
        raise ValueError("Empty JPEG bytes provided")

    try:
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        logger.debug(f"Saved JPEG to {path}")
    except Exception as e:
        logger.error(f"Error saving JPEG to {path}: {e}")
        raise
