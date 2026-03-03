#!/usr/bin/env python3
"""Install Python dependencies for audio validation (Phase 2).

Packages (~800MB-1GB total):
  - numpy / scipy: audio I/O + MFCC computation
  - faster-whisper: speech-to-text with word-level timestamps (CTranslate2, no PyTorch)
  - onnxruntime: needed by silero-vad Python wrapper
  - silero-vad: same VAD model the Android app uses
"""

import subprocess
import sys


def pip_install(*packages: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    print(f"  pip install {' '.join(packages)} ...")
    subprocess.check_call(cmd)


def main() -> None:
    print("=== Installing audio validation dependencies ===\n")

    # Core scientific stack
    print("[1/3] numpy + scipy")
    pip_install("numpy", "scipy")

    # Whisper (CTranslate2 backend — no PyTorch)
    print("[2/3] faster-whisper")
    pip_install("faster-whisper")

    # Silero VAD (ONNX)
    print("[3/3] onnxruntime + silero-vad")
    pip_install("onnxruntime", "silero-vad")

    # Verify
    print("\n=== Verification ===")
    import importlib
    ok = True
    for mod, label in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("faster_whisper", "faster-whisper"),
        ("onnxruntime", "onnxruntime"),
        ("silero_vad", "silero-vad"),
    ]:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "ok")
            print(f"  {label}: {v}")
        except ImportError:
            print(f"  {label}: FAILED")
            ok = False

    if ok:
        print("\nAll dependencies installed successfully.")
    else:
        print("\nSome dependencies failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
