#!/usr/bin/env python3
"""Install dependencies for real-world audio testing: speechbrain + matplotlib."""

import subprocess
import sys


def run(cmd, desc):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        sys.exit(1)
    print(f"  OK")


def main():
    pip = sys.executable + " -m pip"

    # speechbrain for ECAPA-TDNN speaker embeddings
    run(f"{pip} install speechbrain", "Installing speechbrain (ECAPA-TDNN speaker embeddings)")

    # matplotlib for visualization
    run(f"{pip} install matplotlib", "Installing matplotlib (timeline + trace plots)")

    # Verify imports
    print(f"\n{'=' * 60}")
    print(f"  Verifying imports")
    print(f"{'=' * 60}")

    try:
        # Patch torchaudio for speechbrain compatibility (2.10+ removed list_audio_backends)
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["ffmpeg"]
        import speechbrain  # noqa: F401
        print(f"  speechbrain {speechbrain.__version__} — OK")
    except ImportError as e:
        print(f"  speechbrain import FAILED: {e}")
        sys.exit(1)

    try:
        import matplotlib  # noqa: F401
        print(f"  matplotlib {matplotlib.__version__} — OK")
    except ImportError as e:
        print(f"  matplotlib import FAILED: {e}")
        sys.exit(1)

    # Quick check: ECAPA-TDNN model loads
    print(f"\n{'=' * 60}")
    print(f"  Testing ECAPA-TDNN model load")
    print(f"{'=' * 60}")
    try:
        # Patch already applied above for torchaudio compat
        from speechbrain.inference.speaker import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_ecapa",
            run_opts={"device": "cpu"},
        )
        print(f"  ECAPA-TDNN model loaded — OK")
        # Quick embedding test
        import torch
        dummy = torch.randn(1, 16000)  # 1 second of audio
        emb = classifier.encode_batch(dummy)
        print(f"  Embedding shape: {emb.shape} — OK")
    except Exception as e:
        print(f"  ECAPA-TDNN test FAILED: {e}")
        print(f"  (This may work later — model downloads on first use)")

    print(f"\n{'=' * 60}")
    print(f"  All dependencies installed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
