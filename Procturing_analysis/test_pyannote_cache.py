#!/usr/bin/env python3
"""
Test if Pyannote downloads models on each run or uses cache
"""

import time
import os
from pyannote.audio import Pipeline

print("="*60)
print("Testing Pyannote Model Caching")
print("="*60)

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("\n❌ HF_TOKEN not set. Cannot test Pyannote.")
    print("Set it with: export HF_TOKEN='your_token_here'")
    exit(1)

print(f"\n📍 Cache location: ~/.cache/torch/pyannote/")
print(f"📍 Current cache size: ", end="")
os.system("du -sh ~/.cache/torch/pyannote/ 2>/dev/null")

print("\n" + "="*60)
print("🔄 Loading Pyannote Pipeline (RUN 1)")
print("="*60)
start1 = time.perf_counter()
pipeline1 = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=hf_token
)
duration1 = time.perf_counter() - start1
print(f"✓ First load completed in {duration1:.3f} seconds")

print("\n" + "="*60)
print("🔄 Loading Pyannote Pipeline (RUN 2 - Should be cached)")
print("="*60)
start2 = time.perf_counter()
pipeline2 = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=hf_token
)
duration2 = time.perf_counter() - start2
print(f"✓ Second load completed in {duration2:.3f} seconds")

print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
print(f"First load:  {duration1:.3f}s")
print(f"Second load: {duration2:.3f}s")

if duration2 < duration1 * 0.5:
    print("\n✅ CACHED! Models are being reused from cache.")
    print(f"   Speed improvement: {duration1/duration2:.1f}x faster")
elif duration2 < duration1 * 0.9:
    print("\n⚠️  PARTIALLY CACHED. Some components may download each time.")
else:
    print("\n❌ NOT CACHED! Models are downloading on each load.")
    print("   This is a problem - it should be cached!")

print(f"\n📍 Final cache size: ", end="")
os.system("du -sh ~/.cache/torch/pyannote/ 2>/dev/null")

print("\n" + "="*60)
print("💡 NOTE: The pipeline OBJECT is different from the cached MODELS.")
print("   - Models (weights) are cached in ~/.cache/torch/pyannote/")
print("   - Pipeline object is created new each time (expected)")
print("   - As long as models are cached, no re-download happens!")
print("="*60)
