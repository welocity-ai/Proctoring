#!/usr/bin/env python3
"""
Profile Loading Times for Proctoring Analysis Components
---------------------------------------------------------
Measures the exact time taken to:
1. Import each major module
2. Load each ML model
3. Initialize each component

Run this to identify the biggest bottlenecks.
"""

import time
import sys
import os

def time_it(label, func):
    """Time a function and print results."""
    print(f"\n{'='*60}")
    print(f"⏱️  {label}")
    print(f"{'='*60}")
    start = time.perf_counter()
    result = func()
    duration = time.perf_counter() - start
    print(f"✓ Completed in {duration:.3f} seconds")
    return duration, result

def main():
    total_start = time.perf_counter()
    timings = {}
    
    print("\n" + "="*60)
    print("🔬 PROFILING PROCTORING ANALYSIS LOADING TIMES")
    print("="*60)
    
    # ==================== IMPORTS ====================
    print("\n\n📦 PHASE 1: MEASURING IMPORT TIMES\n")
    
    def import_cv2():
        import cv2
        return cv2
    timings['import_cv2'], cv2 = time_it("Importing OpenCV (cv2)", import_cv2)
    
    def import_torch():
        import torch
        return torch
    timings['import_torch'], torch = time_it("Importing PyTorch", import_torch)
    
    def import_mediapipe():
        import mediapipe as mp
        return mp
    timings['import_mediapipe'], mp = time_it("Importing MediaPipe", import_mediapipe)
    
    def import_ultralytics():
        from ultralytics import YOLO
        return YOLO
    timings['import_ultralytics'], YOLO = time_it("Importing Ultralytics YOLO", import_ultralytics)
    
    def import_pyannote():
        from pyannote.audio import Pipeline
        return Pipeline
    timings['import_pyannote'], Pipeline = time_it("Importing Pyannote.audio", import_pyannote)
    
    def import_other():
        import numpy
        from tqdm import tqdm
        import ffmpeg
        try:
            from fpdf2 import FPDF
        except ImportError:
            from fpdf import FPDF
        from jinja2 import Template
        return True
    timings['import_other'], _ = time_it("Importing Other Libraries (numpy, tqdm, ffmpeg, fpdf, jinja2)", import_other)
    
    # ==================== MODEL LOADING ====================
    print("\n\n🤖 PHASE 2: MEASURING MODEL LOADING TIMES\n")
    
    # 1. YOLO Phone Detector
    def load_yolo():
        model_path = "models/phone_detector.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        model = YOLO(model_path)
        return model
    timings['load_yolo'], yolo_model = time_it("Loading YOLO Phone Detector (52MB)", load_yolo)
    
    # 2. MediaPipe Face Mesh
    def load_mediapipe_face_mesh():
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        return face_mesh
    timings['load_mediapipe_face'], face_mesh = time_it("Loading MediaPipe Face Mesh", load_mediapipe_face_mesh)
    
    # 3. MediaPipe Face Detection
    def load_mediapipe_face_det():
        mp_face_det = mp.solutions.face_detection
        detector = mp_face_det.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4
        )
        return detector
    timings['load_mediapipe_det'], face_det = time_it("Loading MediaPipe Face Detection", load_mediapipe_face_det)
    
    # 4. Pyannote Speaker Diarization
    print("\n⚠️  Skipping Pyannote model loading (requires HF_TOKEN and downloads ~500MB)")
    print("   To test Pyannote loading time, set HF_TOKEN env var and uncomment code below")
    timings['load_pyannote'] = 0.0
    
    # Uncomment to test Pyannote (requires HF_TOKEN):
    # def load_pyannote():
    #     hf_token = os.getenv("HF_TOKEN")
    #     if not hf_token:
    #         print("❌ HF_TOKEN not set")
    #         return None
    #     pipeline = Pipeline.from_pretrained(
    #         "pyannote/speaker-diarization-3.0",
    #         use_auth_token=hf_token
    #     )
    #     if torch.backends.mps.is_available():
    #         device = torch.device("mps")
    #     elif torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     else:
    #         device = torch.device("cpu")
    #     pipeline.to(device)
    #     return pipeline
    # timings['load_pyannote'], pyannote_pipeline = time_it("Loading Pyannote Speaker Diarization", load_pyannote)
    
    # ==================== SUMMARY ====================
    total_duration = time.perf_counter() - total_start
    
    print("\n\n" + "="*60)
    print("📊 PROFILING SUMMARY")
    print("="*60)
    
    print("\n🔹 IMPORT TIMES:")
    import_total = sum([
        timings.get('import_cv2', 0),
        timings.get('import_torch', 0),
        timings.get('import_mediapipe', 0),
        timings.get('import_ultralytics', 0),
        timings.get('import_pyannote', 0),
        timings.get('import_other', 0)
    ])
    for key in ['import_cv2', 'import_torch', 'import_mediapipe', 'import_ultralytics', 'import_pyannote', 'import_other']:
        if key in timings:
            pct = (timings[key] / import_total * 100) if import_total > 0 else 0
            print(f"  {key:30s}: {timings[key]:6.3f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL IMPORT TIME':30s}: {import_total:6.3f}s")
    
    print("\n🔹 MODEL LOADING TIMES:")
    model_total = sum([
        timings.get('load_yolo', 0),
        timings.get('load_mediapipe_face', 0),
        timings.get('load_mediapipe_det', 0),
        timings.get('load_pyannote', 0)
    ])
    for key in ['load_yolo', 'load_mediapipe_face', 'load_mediapipe_det', 'load_pyannote']:
        if key in timings:
            pct = (timings[key] / model_total * 100) if model_total > 0 else 0
            print(f"  {key:30s}: {timings[key]:6.3f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL MODEL LOADING TIME':30s}: {model_total:6.3f}s")
    
    print(f"\n🔹 TOTAL PROFILING TIME: {total_duration:.3f}s")
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("="*60)
    
    # Find slowest import
    import_times = {
        'OpenCV': timings.get('import_cv2', 0),
        'PyTorch': timings.get('import_torch', 0),
        'MediaPipe': timings.get('import_mediapipe', 0),
        'Ultralytics': timings.get('import_ultralytics', 0),
        'Pyannote': timings.get('import_pyannote', 0)
    }
    slowest_import = max(import_times.items(), key=lambda x: x[1])
    
    if slowest_import[1] > 1.0:
        print(f"\n1. Slowest import: {slowest_import[0]} ({slowest_import[1]:.2f}s)")
        print("   → Consider lazy imports for this module")
    
    if timings.get('load_yolo', 0) > 2.0:
        print(f"\n2. YOLO model loading is slow ({timings['load_yolo']:.2f}s)")
        print("   → Consider using a lighter model or global caching")
    
    if import_total > 5.0:
        print(f"\n3. Total import time is high ({import_total:.2f}s)")
        print("   → Consider using lazy imports")
        print("   → Consider pre-compiling with PyInstaller/Nuitka")
    
    print("\n4. To enable Pyannote profiling:")
    print("   → Set HF_TOKEN environment variable")
    print("   → Uncomment the Pyannote loading code in this script")
    
    print("\n" + "="*60)
    print("✅ Profiling complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
