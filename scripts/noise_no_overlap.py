#!/usr/bin/env python3
"""Re-run SNR 20 and 30 dB with dual-assignment disabled vs enabled.

Compares dual_t=0.82 (overlap enabled) vs dual_t=1.0 (overlap disabled)
to isolate whether noise degradation comes from dual-assignment or core ID.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS, FRAME_SIZE, SAMPLE_RATE, SPEECH_SEGMENT_FRAMES, MIN_FLUSH_FRAMES,
    PipelineSimulator, SpeakerIdentifier, SpeechBrainDiarizer,
    WeSpeakerEmbedder, compare_timelines, create_silero_vad,
    extract_vad_segments, format_ms, load_audio,
)

from strategy_comparison import (
    run_pipeline_segments, apply_strategy, build_timeline, compute_metrics,
    compute_dual_metrics, relaxed_agreement,
)

from noise_robustness import add_gaussian_noise, run_speechbrain_on, GROUND_TRUTHS

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

SNR_LEVELS = [30, 20]
DUAL_CONFIGS = [
    ("dual=0.82", 0.82, 3),
    ("no overlap", 1.0, 0),
]


def analyze_with_dual(segments, baseline_dec, sb_tl, label_map, duration_ms, dual_t, dual_cap):
    """Compute metrics for a given dual-assignment config."""
    dm = compute_dual_metrics(segments, baseline_dec, dual_t, dual_cap)
    strict, relaxed = relaxed_agreement(
        segments, dm["segment_labels"], sb_tl, label_map, duration_ms)
    return {
        "speech_a": dm["speech_a"],
        "speech_b": dm["speech_b"],
        "overlap": dm["overlap_time"],
        "a_share": dm["a_share"],
        "changes": dm["speaker_changes"],
        "n_segments": len(segments),
        "strict_agree": strict,
        "relaxed_agree": relaxed,
    }


def main():
    print("=" * 105)
    print("  NOISE TEST: Dual-Assignment Enabled vs Disabled at SNR 20 & 30 dB")
    print("=" * 105)

    embedder = WeSpeakerEmbedder()

    for clip_num in [1, 2]:
        clip_path = CLIPS[clip_num]
        if not os.path.exists(clip_path):
            print(f"\nClip {clip_num}: not found")
            continue

        print(f"\n{'=' * 105}")
        print(f"  CLIP {clip_num}")
        print(f"{'=' * 105}")

        audio_clean = load_audio(clip_path)
        duration_ms = len(audio_clean) / SAMPLE_RATE * 1000

        gt = GROUND_TRUTHS[clip_num]
        gt_total = gt["speech_a"] + gt["speech_b"]
        gt_a_share = 100 * gt["speech_a"] / gt_total

        # Clean baseline (run pipeline once, compute both dual configs)
        print(f"  Running clean...")
        sb_tl = run_speechbrain_on(audio_clean)
        segments = run_pipeline_segments(audio_clean, embedder)
        baseline_dec = apply_strategy(segments, {"type": "baseline"})
        baseline_tl = build_timeline(segments, baseline_dec, duration_ms)
        baseline_speech = [(s, e, l) for s, e, l in baseline_tl if l != "silence"]
        cmp = compare_timelines(baseline_speech, sb_tl, duration_ms)
        label_map = cmp["label_map"]

        clean_results = {}
        for name, dt, dc in DUAL_CONFIGS:
            clean_results[name] = analyze_with_dual(
                segments, baseline_dec, sb_tl, label_map, duration_ms, dt, dc)

        # Noisy runs
        noisy_results = {}  # snr -> {config_name -> result}
        for snr in SNR_LEVELS:
            print(f"  Running SNR={snr} dB...")
            t0 = time.time()
            noisy_audio = add_gaussian_noise(audio_clean, snr)
            sb_tl_noisy = run_speechbrain_on(noisy_audio)
            segments_noisy = run_pipeline_segments(noisy_audio, embedder)
            baseline_dec_noisy = apply_strategy(segments_noisy, {"type": "baseline"})
            baseline_tl_noisy = build_timeline(segments_noisy, baseline_dec_noisy, duration_ms)
            baseline_speech_noisy = [(s, e, l) for s, e, l in baseline_tl_noisy if l != "silence"]
            cmp_noisy = compare_timelines(baseline_speech_noisy, sb_tl_noisy, duration_ms)
            lmap_noisy = cmp_noisy["label_map"]

            noisy_results[snr] = {}
            for name, dt, dc in DUAL_CONFIGS:
                noisy_results[snr][name] = analyze_with_dual(
                    segments_noisy, baseline_dec_noisy, sb_tl_noisy, lmap_noisy,
                    duration_ms, dt, dc)
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

        # Print table
        cw = 11
        lw = 16
        cols = ["Speech A", "Speech B", "Overlap", "A share", "Changes",
                "Strict SB", "Relaxed SB"]
        sep = "  " + "─" * (lw + cw * len(cols))
        header = f"  {'Config':>{lw}s}" + "".join(f"{c:>{cw}s}" for c in cols)

        print(f"\n{sep}")
        print(header)
        print(sep)

        def fmt_t(ms): return format_ms(ms)
        def fmt_p(v): return f"{v:.1f}%"

        def print_row(label, r):
            line = f"  {label:>{lw}s}"
            line += f"{fmt_t(r['speech_a']):>{cw}s}"
            line += f"{fmt_t(r['speech_b']):>{cw}s}"
            line += f"{fmt_t(r['overlap']):>{cw}s}"
            line += f"{fmt_p(r['a_share']):>{cw}s}"
            line += f"{str(r['changes']):>{cw}s}"
            line += f"{fmt_p(r['strict_agree']):>{cw}s}"
            line += f"{fmt_p(r['relaxed_agree']):>{cw}s}"
            print(line)

        # GT row
        line = f"  {'GT':>{lw}s}"
        line += f"{fmt_t(gt['speech_a']):>{cw}s}"
        line += f"{fmt_t(gt['speech_b']):>{cw}s}"
        line += f"{fmt_t(gt['overlap']):>{cw}s}"
        line += f"{fmt_p(gt_a_share):>{cw}s}"
        line += f"{'—':>{cw}s}" * 3
        print(line)
        print(sep)

        # Clean
        for name, _, _ in DUAL_CONFIGS:
            print_row(f"Clean {name}", clean_results[name])
        print(sep)

        # Per SNR
        for snr in SNR_LEVELS:
            for name, _, _ in DUAL_CONFIGS:
                print_row(f"{snr}dB {name}", noisy_results[snr][name])
            print(sep)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
