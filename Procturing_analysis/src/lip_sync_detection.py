# # src/lip_sync_detection.py

# import os
# import cv2
# import torch
# import numpy as np
# import wave
# import contextlib
# from tqdm import tqdm
# from datetime import timedelta

# # ---- Import from your existing modules ----
# from src.voice_diarization import get_speech_segments
# from src.face_and_gaze_analysis import extract_mouth_frames_for_window


# # ------------------------------------------------------------
# # Time formatting helper
# # ------------------------------------------------------------
# def _fmt_time(seconds: float) -> str:
#     td = timedelta(seconds=seconds)
#     return str(td)[:-3]


# # ------------------------------------------------------------
# # LipNet-Sync Model Definition
# # (Audio-Video Synchronization classifier)
# # ------------------------------------------------------------
# class LipNetSync(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         # 3D CNN for video (mouth ROI)
#         self.video_encoder = torch.nn.Sequential(
#             torch.nn.Conv3d(3, 32, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool3d((1, 2, 2)),

#             torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool3d((1, 2, 2)),

#             torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.AdaptiveAvgPool3d((8, 1, 1))
#         )

#         # Audio encoder (raw waveform 16k)
#         self.audio_encoder = torch.nn.Sequential(
#             torch.nn.Linear(16000, 2048),
#             torch.nn.ReLU(),
#             torch.nn.Linear(2048, 512),
#             torch.nn.ReLU()
#         )

#         # Final classifier
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(128 * 8 + 512, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 1),
#             torch.nn.Sigmoid()
#         )

#     def forward(self, video_tensor, audio_tensor):
#         v = self.video_encoder(video_tensor).flatten(1)
#         a = self.audio_encoder(audio_tensor)
#         x = torch.cat([v, a], dim=1)
#         return self.classifier(x)


# # ------------------------------------------------------------
# # Wrapper class for loading + inference
# # ------------------------------------------------------------
# class LipSyncDetector:

#     def __init__(self, checkpoint_path="src/models/lipnet_sync.pth"):
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(
#                 f"❌ Missing model file: {checkpoint_path}\n"
#                 "Download using:\n"
#                 "huggingface-cli download SylexCorp/lipnet-sync lipnet_sync.pth --local-dir src/models"
#             )

#         # Choose the best device
#         self.device = (
#             torch.device("mps") if torch.backends.mps.is_available() else
#             torch.device("cuda") if torch.cuda.is_available() else
#             torch.device("cpu")
#         )

#         print(f"[INFO] Loading LipNet-Sync model on: {self.device}")

#         self.model = LipNetSync().to(self.device)
#         state = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(state, strict=False)
#         self.model.eval()

#     def predict_sync(self, mouth_frames, audio_waveform):
#         """
#         mouth_frames: List of numpy arrays (RGB 96x96)
#         audio_waveform: float32 numpy array 16000 samples
#         """
#         # Not enough frames → cannot measure properly
#         if len(mouth_frames) < 4:
#             return 0.0

#         # Build video tensor: (1, 3, T, 96, 96)
#         vid = np.stack(mouth_frames, axis=0) / 255.0
#         vid = np.transpose(vid, (0, 3, 1, 2))  # (T, 3, H, W)
#         vid = torch.tensor(vid, dtype=torch.float32).unsqueeze(0).to(self.device)

#         # Audio tensor: (1, samples)
#         audio = torch.tensor(audio_waveform, dtype=torch.float32).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             score = self.model(vid, audio).squeeze().item()

#         return float(score)


# # ------------------------------------------------------------
# # MAIN LIP-SYNC DETECTION PIPELINE
# # ------------------------------------------------------------
# def detect_lip_sync(
#     video_path: str,
#     audio_path: str,
#     hf_token: str,
#     frame_step: int = 2,
#     min_window: float = 1.0,
#     threshold: float = 0.50,
#     only_primary: bool = True,
#     outdir: str = "reports/lipsync"
# ):

#     os.makedirs(outdir, exist_ok=True)
#     proof_dir = os.path.join(outdir, "proofs")
#     os.makedirs(proof_dir, exist_ok=True)

#     # --------------------------------------------------------
#     # 1. Use your existing diarization to get spoken segments
#     # --------------------------------------------------------
#     diar = get_speech_segments(audio_path, hf_token, min_noise_duration=0.2, only_primary=only_primary)
#     primary = diar.get("primary_speaker")
#     segments = diar.get("speaker_segments", {})

#     if only_primary:
#         segs = segments.get(primary, [])
#     else:
#         segs = [s | {"speaker": spk} for spk, arr in segments.items() for s in arr]

#     # --------------------------------------------------------
#     # 2. Load LipNet-Sync model
#     # --------------------------------------------------------
#     detector = LipSyncDetector()

#     results = []

#     # --------------------------------------------------------
#     # 3. Process each speech segment
#     # --------------------------------------------------------
#     for seg in tqdm(segs, desc="LipSync"):

#         start, end = seg["start"], seg["end"]
#         dur = end - start

#         if dur < min_window:
#             continue

#         # ---- Extract mouth frames using your existing helper ----
#         mouth_frames = extract_mouth_frames_for_window(
#             video_path,
#             start,
#             end,
#             frame_step=frame_step
#         )

#         # ---- Extract audio waveform for the same window ----
#         with contextlib.closing(wave.open(audio_path, "rb")) as wav:
#             sr = wav.getframerate()
#             s = int(start * sr)
#             e = int(end * sr)
#             wav.setpos(s)
#             audio = wav.readframes(e - s)
#             audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32)

#         # ---- Compute lip-sync score ----
#         score = detector.predict_sync(mouth_frames, audio)
#         match = score >= threshold

#         # Save proof image if mismatch
#         proof = None
#         if not match and len(mouth_frames) > 0:
#             mid = mouth_frames[len(mouth_frames)//2]
#             outpath = os.path.join(proof_dir, f"mismatch_{start:.2f}.jpg")
#             cv2.imwrite(outpath, cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))
#             proof = outpath

#         results.append({
#             "start": _fmt_time(start),
#             "end": _fmt_time(end),
#             "duration": round(dur, 3),
#             "sync_score": round(score, 3),
#             "lip_sync_match": match,
#             "proof_image": proof
#         })

#     return results


# # ------------------------------------------------------------
# # CLI TESTER
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     import argparse, json

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video", required=True)
#     parser.add_argument("--audio", required=True)
#     parser.add_argument("--hf-token", required=True)
#     parser.add_argument("--outdir", default="reports/lipsync")

#     args = parser.parse_args()

#     out = detect_lip_sync(
#         video_path=args.video,
#         audio_path=args.audio,
#         hf_token=args.hf_token,
#         outdir=args.outdir
#     )

#     print(json.dumps(out, indent=2))
