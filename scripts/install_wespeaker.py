#!/usr/bin/env python3
"""Download WeSpeaker ECAPA-TDNN-512 ONNX model and inspect its input/output specs.

Model: Wespeaker/wespeaker-ecapa-tdnn512-LM from HuggingFace
- 24.9 MB ONNX model
- 192-dim speaker embeddings
- Expects 80-dim Fbank features (25ms frame, 10ms shift, 16kHz)

Usage:
  python3 scripts/install_wespeaker.py              # download + inspect
  python3 scripts/install_wespeaker.py --inspect     # inspect only (already downloaded)
"""

import argparse
import os
import sys
import urllib.request

MODEL_URL = "https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/main/voxceleb_ECAPA512_LM.onnx"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "voxceleb_ECAPA512_LM.onnx")


def download_model():
    """Download the WeSpeaker ONNX model from HuggingFace."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    print(f"Downloading WeSpeaker ECAPA-TDNN-512 ONNX model...")
    print(f"  URL: {MODEL_URL}")
    print(f"  Destination: {MODEL_PATH}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress_hook)
    print()

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Download complete: {size_mb:.1f} MB")


def inspect_model():
    """Inspect the ONNX model's input/output specs."""
    import onnxruntime as ort

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run without --inspect to download first.")
        sys.exit(1)

    print(f"\nInspecting ONNX model: {MODEL_PATH}")
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    session = ort.InferenceSession(MODEL_PATH)

    print(f"\n  Inputs:")
    for inp in session.get_inputs():
        print(f"    name={inp.name}, shape={inp.shape}, type={inp.type}")

    print(f"\n  Outputs:")
    for out in session.get_outputs():
        print(f"    name={out.name}, shape={out.shape}, type={out.type}")

    # Test inference with dummy data
    import numpy as np
    print(f"\n  Test inference with dummy 80-dim Fbank (200 frames = 2s)...")
    dummy_input = np.random.randn(1, 200, 80).astype(np.float32)
    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: dummy_input})
        for i, out in enumerate(outputs):
            print(f"    output[{i}]: shape={out.shape}, dtype={out.dtype}")
            if out.ndim >= 1:
                print(f"    embedding dim: {out.shape[-1]}")
    except Exception as e:
        print(f"    Failed: {e}")

    # Test with different lengths to check dynamic shape support
    print(f"\n  Test variable-length input (150 frames = 1.5s)...")
    dummy_short = np.random.randn(1, 150, 80).astype(np.float32)
    try:
        outputs = session.run(None, {input_name: dummy_short})
        print(f"    output shape: {outputs[0].shape} — variable length supported!")
    except Exception as e:
        print(f"    Failed: {e} — fixed length only")

    print(f"\n  Test variable-length input (50 frames = 0.5s)...")
    dummy_tiny = np.random.randn(1, 50, 80).astype(np.float32)
    try:
        outputs = session.run(None, {input_name: dummy_tiny})
        print(f"    output shape: {outputs[0].shape} — short input supported!")
    except Exception as e:
        print(f"    Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download and inspect WeSpeaker ONNX model")
    parser.add_argument("--inspect", action="store_true", help="Inspect only (skip download)")
    args = parser.parse_args()

    if not args.inspect:
        download_model()
    inspect_model()


if __name__ == "__main__":
    main()
