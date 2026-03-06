#!/usr/bin/env python3
"""Quick diagnostic: similarity gap distributions at different noise levels.

Prints percentiles of |sim_a - sim_b| for clean, 35 dB, 30 dB on both clips
(only segments after B is established). Used to calibrate adaptive noise thresholds.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import SAMPLE_RATE, load_audio, format_ms, WeSpeakerEmbedder
from strategy_comparison import run_pipeline_segments
from noise_robustness import add_gaussian_noise

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

SNR_LEVELS = ["clean", 40, 35, 30]


def get_post_b_gaps(segments):
    """Return list of |sim_a - sim_b| for segments after B is established."""
    gaps = []
    b_established = False
    for seg in segments:
        if seg["sim_b"] is not None and seg["sim_b"] > 0:
            b_established = True
        if b_established and seg["sim_b"] is not None and seg["sim_b"] > 0:
            gaps.append(abs(seg["sim_a"] - seg["sim_b"]))
    return gaps


def main():
    embedder = WeSpeakerEmbedder()

    for clip_num in [1, 2]:
        clip_path = CLIPS[clip_num]
        if not os.path.exists(clip_path):
            continue

        print(f"\n{'='*80}")
        print(f"  CLIP {clip_num} — Similarity gap distributions")
        print(f"{'='*80}")

        audio = load_audio(clip_path)

        for snr in SNR_LEVELS:
            if snr == "clean":
                test_audio = audio
                label = "Clean"
            else:
                test_audio = add_gaussian_noise(audio, snr)
                label = f"{snr} dB"

            segments = run_pipeline_segments(test_audio, embedder)
            gaps = get_post_b_gaps(segments)

            if not gaps:
                print(f"\n  {label}: B never established (0 post-B segments)")
                continue

            gaps = np.array(gaps)
            pcts = [10, 25, 50, 75, 90]
            pct_vals = np.percentile(gaps, pcts)

            print(f"\n  {label}: {len(gaps)} post-B segments")
            print(f"    Mean gap: {gaps.mean():.4f}  Std: {gaps.std():.4f}")
            print(f"    Percentiles: ", end="")
            for p, v in zip(pcts, pct_vals):
                print(f"p{p}={v:.4f}  ", end="")
            print()
            print(f"    Min: {gaps.min():.4f}  Max: {gaps.max():.4f}")

            # Also show min(sim_a, sim_b) distribution
            mins = []
            for seg in segments:
                if seg["sim_b"] is not None and seg["sim_b"] > 0:
                    mins.append(min(seg["sim_a"], seg["sim_b"]))
            if mins:
                mins = np.array(mins)
                print(f"    min(sim_a,sim_b): mean={mins.mean():.4f} std={mins.std():.4f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
