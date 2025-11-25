import os

# Default Configuration
DEFAULT_OUT_DIR = "reports/session_test"
OBJECT_SAMPLE_MS = 1000
GAZE_FRAME_STEP = 2
YOLO_MODEL_PATH = "models/phone_detector.pt"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(levelname)s] %(message)s"

# Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
