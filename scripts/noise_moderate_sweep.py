#!/usr/bin/env python3
"""Moderate noise tier test: 35 dB SNR with dual_t sweep.

Tests dual_t = [0.82, 0.83, 0.84, 0.85, 0.86] at SNR=35 dB (midpoint between
40 dB "fine" and 30 dB "broken") to find a moderate-noise overlap threshold.

Also runs clean baseline at each threshold for comparison, and saves 35 dB
noisy clips as WAV files for manual listening.
"""
import os
import struct
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
NOISY_DIR = os.path.join(CLIP_DIR, "noisy")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

SNR = 35
DUAL_THRESHOLDS = [0.82, 0.83, 0.84, 0.85, 0.86]
DUAL_CAP = 3


def save_wav(audio_int16, path):
    """Save int16 audio as 16kHz mono WAV."""
    data = audio_int16.tobytes()
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(data)))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # mono
        f.write(struct.pack('<I', SAMPLE_RATE))
        f.write(struct.pack('<I', SAMPLE_RATE * 2))
        f.write(struct.pack('<H', 2))   # block align
        f.write(struct.pack('<H', 16))  # bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', len(data)))
        f.write(data)


def compute_for_dual_t(segments, baseline_dec, sb_tl, label_map, duration_ms, dual_t):
    """Compute dual-assignment metrics for one threshold (post-hoc, no pipeline rerun)."""
    dm = compute_dual_metrics(segments, baseline_dec, dual_t, DUAL_CAP)
    strict, relaxed = relaxed_agreement(
        segments, dm["segment_labels"], sb_tl, label_map, duration_ms)
    return {
        "speech_a": dm["speech_a"],
        "speech_b": dm["speech_b"],
        "overlap": dm["overlap_time"],
        "a_share": dm["a_share"],
        "changes": dm["speaker_changes"],
        "strict_agree": strict,
        "relaxed_agree": relaxed,
    }


def main():
    print("=" * 105)
    print(f"  MODERATE NOISE TIER — 35 dB Dual-Threshold Sweep")
    print(f"  Thresholds: {DUAL_THRESHOLDS}, cap={DUAL_CAP}")
    print("=" * 105)

    os.makedirs(NOISY_DIR, exist_ok=True)
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

        # ── Clean: run pipeline + SpeechBrain once ──
        print(f"  Running clean pipeline...")
        t0 = time.time()
        sb_tl_clean = run_speechbrain_on(audio_clean)
        segments_clean = run_pipeline_segments(audio_clean, embedder)
        baseline_dec_clean = apply_strategy(segments_clean, {"type": "baseline"})
        baseline_tl_clean = build_timeline(segments_clean, baseline_dec_clean, duration_ms)
        baseline_speech_clean = [(s, e, l) for s, e, l in baseline_tl_clean if l != "silence"]
        cmp_clean = compare_timelines(baseline_speech_clean, sb_tl_clean, duration_ms)
        label_map_clean = cmp_clean["label_map"]
        print(f"  Clean done in {time.time() - t0:.1f}s")

        # Compute clean results for each threshold
        clean_results = {}
        for dt in DUAL_THRESHOLDS:
            clean_results[dt] = compute_for_dual_t(
                segments_clean, baseline_dec_clean, sb_tl_clean,
                label_map_clean, duration_ms, dt)
        # Also clean with no overlap
        clean_results["no_ovl"] = compute_for_dual_t(
            segments_clean, baseline_dec_clean, sb_tl_clean,
            label_map_clean, duration_ms, 1.0)

        # ── 35 dB: run pipeline + SpeechBrain once ──
        print(f"  Running 35 dB pipeline...")
        t0 = time.time()
        noisy_audio = add_gaussian_noise(audio_clean, SNR)

        # Save noisy clip
        wav_path = os.path.join(NOISY_DIR, f"clip{clip_num}_snr{SNR}dB.wav")
        save_wav(noisy_audio, wav_path)
        print(f"  Saved: {wav_path}")

        sb_tl_noisy = run_speechbrain_on(noisy_audio)
        segments_noisy = run_pipeline_segments(noisy_audio, embedder)
        baseline_dec_noisy = apply_strategy(segments_noisy, {"type": "baseline"})
        baseline_tl_noisy = build_timeline(segments_noisy, baseline_dec_noisy, duration_ms)
        baseline_speech_noisy = [(s, e, l) for s, e, l in baseline_tl_noisy if l != "silence"]
        cmp_noisy = compare_timelines(baseline_speech_noisy, sb_tl_noisy, duration_ms)
        label_map_noisy = cmp_noisy["label_map"]
        print(f"  35 dB done in {time.time() - t0:.1f}s")

        # Compute noisy results for each threshold
        noisy_results = {}
        for dt in DUAL_THRESHOLDS:
            noisy_results[dt] = compute_for_dual_t(
                segments_noisy, baseline_dec_noisy, sb_tl_noisy,
                label_map_noisy, duration_ms, dt)
        # Also noisy with no overlap
        noisy_results["no_ovl"] = compute_for_dual_t(
            segments_noisy, baseline_dec_noisy, sb_tl_noisy,
            label_map_noisy, duration_ms, 1.0)

        # ── Print table ──
        cw = 11
        lw = 18
        cols = ["Speech A", "Speech B", "Overlap", "A share", "Changes",
                "Strict SB", "Relaxed SB"]
        sep = "  " + "─" * (lw + cw * len(cols))
        header = f"  {'':>{lw}s}" + "".join(f"{c:>{cw}s}" for c in cols)

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

        # Clean rows
        for dt in DUAL_THRESHOLDS:
            print_row(f"Clean t={dt}", clean_results[dt])
        print_row("Clean no overlap", clean_results["no_ovl"])
        print(sep)

        # 35 dB rows
        for dt in DUAL_THRESHOLDS:
            print_row(f"35dB t={dt}", noisy_results[dt])
        print_row("35dB no overlap", noisy_results["no_ovl"])
        print(sep)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
