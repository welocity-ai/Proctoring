import cv2
import logging
import time

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load Haar Cascade Model
# -----------------------------
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise FileNotFoundError("⚠️ Haar cascade model not found!")

logger.info("✅ Haar Cascade model loaded successfully")

# -----------------------------
# Initialize Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

logger.info("🎥 Webcam stream started (Press 'q' to quit)")

# -----------------------------
# Real-time Face Detection
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        logger.warning("⚠️ Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    logger.info(f"Detected {len(faces)} face(s)")

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Face Detection (Haar Cascade)", frame)

    # Press 's' to save a snapshot
    if cv2.waitKey(1) & 0xFF == ord('s'):
        timestamp = int(time.time())
        filename = f"output_detected_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"📸 Saved snapshot as {filename}")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("👋 Exiting live feed...")
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
logger.info("✅ Webcam released and all windows closed.")
