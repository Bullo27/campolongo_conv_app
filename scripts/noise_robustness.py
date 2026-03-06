#!/usr/bin/env python3
"""Noise robustness test: Gaussian noise at various SNR levels.

Tests how the neural speaker diarization pipeline degrades under additive
Gaussian noise. Compares our pipeline (with dual-assignment at t=0.82, c=3)
against SpeechBrain and manual ground truth at each SNR level.

SNR levels: 40, 30, 20, 15, 10, 5 dB
- 40 dB: barely perceptible noise
- 20-25 dB: typical phone-on-table scenario
- 10 dB: very noisy, near practical limit
- 5 dB: stress test
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

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

SNR_LEVELS = [40, 30, 20, 15, 10, 5]
DUAL_T = 0.82
DUAL_CAP = 3

# Ground truth (from scripts/ground_truths.md)
GROUND_TRUTHS = {
    1: {
        "speech_a": 171000, "speech_b": 35000, "overlap": 10000,
        "silence": 6000, "duration": 202000,
    },
    2: {
        "speech_a": 195000, "speech_b": 579000, "overlap": 35000,
        "silence": 12000, "duration": 751000,
    },
}


def add_gaussian_noise(audio_int16, snr_db, seed=42):
    """Add Gaussian noise to int16 audio at the specified SNR (dB)."""
    audio_f64 = audio_int16.astype(np.float64)
    signal_power = np.mean(audio_f64 ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.default_rng(seed).normal(0, np.sqrt(noise_power), len(audio_f64))
    return np.clip(audio_f64 + noise, -32768, 32767).astype(np.int16)


def run_speechbrain_on(audio):
    """Run SpeechBrain diarization on audio, return timeline."""
    diarizer = SpeechBrainDiarizer()
    vad_fn = create_silero_vad()
    sim = PipelineSimulator(vad_fn, smoothing_window=1,
                            speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                            b_confirm_frames=1)
    sim.run(audio)
    vad_speech = extract_vad_segments(sim.timeline)
    return diarizer.diarize(audio, vad_speech)


def run_full_analysis(audio, embedder, sb_tl, duration_ms):
    """Run our pipeline + dual-assignment, compute all metrics vs SpeechBrain.

    Returns (result_dict, label_map, segments) where result_dict has:
    speech_a, speech_b, overlap, a_share, changes, n_segments,
    strict_agree, relaxed_agree, silence.
    """
    segments = run_pipeline_segments(audio, embedder)

    # Baseline decisions
    baseline_dec = apply_strategy(segments, {"type": "baseline"})
    baseline_tl = build_timeline(segments, baseline_dec, duration_ms)
    baseline_speech = [(s, e, l) for s, e, l in baseline_tl if l != "silence"]

    # Label map from comparing baseline vs SB
    cmp = compare_timelines(baseline_speech, sb_tl, duration_ms)
    label_map = cmp["label_map"]

    # Dual-assignment metrics
    dm = compute_dual_metrics(segments, baseline_dec, DUAL_T, DUAL_CAP)
    strict, relaxed = relaxed_agreement(
        segments, dm["segment_labels"], sb_tl, label_map, duration_ms)

    # Total silence from timeline
    total_silence = sum(e - s for s, e, l in baseline_tl if l == "silence")

    result = {
        "speech_a": dm["speech_a"],
        "speech_b": dm["speech_b"],
        "overlap": dm["overlap_time"],
        "a_share": dm["a_share"],
        "changes": dm["speaker_changes"],
        "n_segments": len(segments),
        "strict_agree": strict,
        "relaxed_agree": relaxed,
        "silence": total_silence,
    }
    return result, label_map


def sb_metrics_from_tl(sb_tl, label_map, duration_ms):
    """Compute SpeechBrain metrics from its timeline + label map."""
    sb_mapped = sorted([(s, e, label_map.get(l, l)) for s, e, l in sb_tl], key=lambda x: x[0])
    sb_full = []
    prev = 0
    for s, e, l in sb_mapped:
        if s > prev:
            sb_full.append((prev, s, "silence"))
        sb_full.append((s, e, l))
        prev = e
    if prev < duration_ms:
        sb_full.append((prev, round(duration_ms), "silence"))
    return compute_metrics(sb_full)


def main():
    print("=" * 105)
    print("  NOISE ROBUSTNESS TEST — Gaussian Noise at Various SNR Levels")
    print(f"  Pipeline: t=0.80, bconf=2, buffer=47, margin=0, smoothing=1")
    print(f"  Dual-assignment: dual_t={DUAL_T}, cap={DUAL_CAP}")
    print(f"  SNR levels: {SNR_LEVELS} dB")
    print("=" * 105)

    embedder = WeSpeakerEmbedder()

    for clip_num in [1, 2]:
        clip_path = CLIPS.get(clip_num)
        if not clip_path or not os.path.exists(clip_path):
            print(f"\nClip {clip_num}: not found, skipping")
            continue

        print(f"\n{'=' * 105}")
        print(f"  CLIP {clip_num}")
        print(f"{'=' * 105}")

        audio_clean = load_audio(clip_path)
        duration_s = len(audio_clean) / SAMPLE_RATE
        duration_ms = duration_s * 1000
        print(f"  Duration: {format_ms(duration_ms)} ({duration_s:.1f}s)")

        gt = GROUND_TRUTHS[clip_num]
        gt_total = gt["speech_a"] + gt["speech_b"]
        gt_a_share = 100 * gt["speech_a"] / gt_total

        # ── Clean baseline (single pipeline run) ──
        print(f"\n  Running clean baseline...")
        t0 = time.time()
        sb_tl_clean = run_speechbrain_on(audio_clean)
        clean_result, clean_label_map = run_full_analysis(
            audio_clean, embedder, sb_tl_clean, duration_ms)
        clean_elapsed = time.time() - t0
        print(f"  Clean done in {clean_elapsed:.1f}s")

        sb_clean_m = sb_metrics_from_tl(sb_tl_clean, clean_label_map, duration_ms)

        # ── Noisy runs ──
        noisy_results = {}
        sb_noisy_metrics = {}
        for snr in SNR_LEVELS:
            print(f"  Running SNR={snr} dB...")
            t0 = time.time()
            noisy_audio = add_gaussian_noise(audio_clean, snr)

            sb_tl_noisy = run_speechbrain_on(noisy_audio)
            result, noisy_label_map = run_full_analysis(
                noisy_audio, embedder, sb_tl_noisy, duration_ms)
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s — A={format_ms(result['speech_a'])}, "
                  f"B={format_ms(result['speech_b'])}, overlap={format_ms(result['overlap'])}")

            noisy_results[snr] = result
            sb_noisy_metrics[snr] = sb_metrics_from_tl(
                sb_tl_noisy, noisy_label_map, duration_ms)

        # ── Print results table ──
        cw = 11  # column width
        lw = 14

        cols = ["Speech A", "Speech B", "Overlap", "A share", "Changes",
                "Segments", "Strict SB", "Relaxed SB"]

        sep = "  " + "─" * (lw + cw * len(cols))
        header = f"  {'SNR':>{lw}s}" + "".join(f"{c:>{cw}s}" for c in cols)

        print(f"\n  Our pipeline (dual_t={DUAL_T}, cap={DUAL_CAP}):")
        print(sep)
        print(header)
        print(sep)

        def fmt_time(ms):
            return format_ms(ms)

        def fmt_pct(v):
            return f"{v:.1f}%"

        def print_our_row(label, r):
            line = f"  {label:>{lw}s}"
            line += f"{fmt_time(r['speech_a']):>{cw}s}"
            line += f"{fmt_time(r['speech_b']):>{cw}s}"
            line += f"{fmt_time(r['overlap']):>{cw}s}"
            line += f"{fmt_pct(r['a_share']):>{cw}s}"
            line += f"{str(r['changes']):>{cw}s}"
            line += f"{str(r['n_segments']):>{cw}s}"
            line += f"{fmt_pct(r['strict_agree']):>{cw}s}"
            line += f"{fmt_pct(r['relaxed_agree']):>{cw}s}"
            print(line)

        # GT row
        line = f"  {'GT':>{lw}s}"
        line += f"{fmt_time(gt['speech_a']):>{cw}s}"
        line += f"{fmt_time(gt['speech_b']):>{cw}s}"
        line += f"{fmt_time(gt['overlap']):>{cw}s}"
        line += f"{fmt_pct(gt_a_share):>{cw}s}"
        line += f"{'—':>{cw}s}" * 4
        print(line)
        print(sep)

        # Clean row
        print_our_row("Clean", clean_result)

        # Noisy rows
        for snr in SNR_LEVELS:
            print_our_row(f"{snr} dB", noisy_results[snr])

        print(sep)

        # ── SpeechBrain section ──
        def print_sb_row(label, m):
            line = f"  {label:>{lw}s}"
            line += f"{fmt_time(m['speech_a']):>{cw}s}"
            line += f"{fmt_time(m['speech_b']):>{cw}s}"
            line += f"{'—':>{cw}s}"
            sb_total = m["speech_a"] + m["speech_b"]
            a_share = 100 * m["speech_a"] / sb_total if sb_total > 0 else 0
            line += f"{fmt_pct(a_share):>{cw}s}"
            line += f"{str(m['speaker_changes']):>{cw}s}"
            line += f"{'—':>{cw}s}" * 3
            print(line)

        print_sb_row("SB clean", sb_clean_m)
        for snr in SNR_LEVELS:
            print_sb_row(f"SB {snr}dB", sb_noisy_metrics[snr])

        print(sep)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
