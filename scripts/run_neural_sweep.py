#!/usr/bin/env python3
"""Sweep neural engine parameters (threshold, buffer, smoothing, bconf) on real audio clips.

Sweeps:
  --sim-threshold:    0.50 to 0.85 in 0.05 steps (8 values)
  --speech-buffer:    30, 47, 60 frames (~1.0s, ~1.5s, ~1.9s)
  --smoothing-window: 1, 4, 6
  --b-confirm-frames: 1, 2

Total combos per clip: 8 × 3 × 3 × 2 = 144
Runs on clips 1-2 with SpeechBrain reference for agreement %.

Usage:
  python3 scripts/run_neural_sweep.py                    # full sweep
  python3 scripts/run_neural_sweep.py --clips 1          # single clip
  python3 scripts/run_neural_sweep.py --no-diarize       # skip SpeechBrain (no agreement %)
  python3 scripts/run_neural_sweep.py --quick             # quick sweep (fewer combos)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS,
    NEURAL_SEGMENT_FRAMES,
    PipelineSimulator,
    SAMPLE_RATE,
    SPEECH_SEGMENT_FRAMES,
    SpeechBrainDiarizer,
    compare_timelines,
    create_silero_vad,
    extract_vad_segments,
    format_ms,
    load_audio,
)

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
RESULTS_DIR = os.path.join(CLIP_DIR, "results")

CLIPS = {
    1: "clip1_theater_conversation.mp3",
    2: "clip2_tv_study_interview.mp3",
}

CLIP_DESCRIPTIONS = {
    1: "Theater conversation (clean)",
    2: "TV studio interview (clean)",
}

# Sweep ranges
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
BUFFERS = [30, 47, 60]
SMOOTHING = [1, 4, 6]
BCONF = [1, 2]

# Quick sweep: fewer combos for fast iteration
QUICK_THRESHOLDS = [0.55, 0.65, 0.75, 0.85]
QUICK_BUFFERS = [47]
QUICK_SMOOTHING = [1, 6]
QUICK_BCONF = [2]

# Best MFCC baseline for comparison
MFCC_BEST = {
    1: {"agree": 72.4, "wtb": 63000, "wta": 553000, "changes": 78},
    2: {"agree": 75.3, "wtb": 235900, "wta": 316800, "changes": 113},
}


def get_sb_reference(audio, vad_fn, diarizer):
    """Run SpeechBrain once per clip, return (sb_timeline, sb_wta_ms, sb_wtb_ms, sb_tst_ms)."""
    if diarizer is None:
        return None, 0, 0, 0
    baseline = PipelineSimulator(vad_fn, smoothing_window=1,
                                 speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                 b_confirm_frames=1)
    baseline.run(audio)
    vad_segments = extract_vad_segments(baseline.timeline)
    sb_timeline = diarizer.diarize(audio, vad_segments)
    duration_ms = len(audio) / SAMPLE_RATE * 1000
    baseline_cmp = compare_timelines(baseline.timeline, sb_timeline, duration_ms)
    return (sb_timeline,
            baseline_cmp.get("sb_wta_ms", 0),
            baseline_cmp.get("sb_wtb_ms", 0),
            baseline_cmp.get("sb_tst_ms", 0))


def run_single_config(audio, vad_fn, threshold, buf, smooth, bconf, sb_timeline, duration_ms):
    """Run neural pipeline with given config, return metrics + comparison."""
    pipeline = PipelineSimulator(
        vad_fn,
        smoothing_window=smooth,
        speech_segment_frames=buf,
        b_confirm_frames=bconf,
        engine="neural",
        sim_threshold=threshold,
    )
    metrics = pipeline.run(audio)

    agreement_pct = 0
    if sb_timeline is not None:
        cmp = compare_timelines(pipeline.timeline, sb_timeline, duration_ms)
        agreement_pct = cmp.get("agreement_pct", 0)

    # Count speaker changes
    speaker_changes = 0
    prev_label = None
    for _, _, lbl in pipeline.timeline:
        if lbl in ("A", "B") and lbl != prev_label and prev_label in ("A", "B"):
            speaker_changes += 1
        if lbl in ("A", "B"):
            prev_label = lbl

    return metrics, agreement_pct, speaker_changes


def run_sweep(clip_nums, vad_fn, diarizer, thresholds, buffers, smoothing_vals, bconf_vals):
    """Run full parameter sweep, return list of result dicts."""
    results = []
    total_combos = len(thresholds) * len(buffers) * len(smoothing_vals) * len(bconf_vals)

    for clip_num in clip_nums:
        clip_path = os.path.join(CLIP_DIR, CLIPS[clip_num])
        if not os.path.exists(clip_path):
            print(f"WARNING: Clip {clip_num} not found, skipping.")
            continue

        print(f"\n{'=' * 70}")
        print(f"  NEURAL SWEEP — CLIP {clip_num}: {CLIP_DESCRIPTIONS.get(clip_num, '?')}")
        print(f"  {total_combos} configs to test")
        print(f"{'=' * 70}")

        audio = load_audio(clip_path)
        duration_ms = len(audio) / SAMPLE_RATE * 1000

        sb_timeline, sb_wta, sb_wtb, sb_tst = get_sb_reference(audio, vad_fn, diarizer)
        if sb_timeline:
            print(f"  SB ref: WTA={format_ms(sb_wta)}, WTB={format_ms(sb_wtb)}, TST={format_ms(sb_tst)}")

        combo_num = 0
        for thresh in thresholds:
            for buf in buffers:
                for smooth in smoothing_vals:
                    for bconf in bconf_vals:
                        combo_num += 1
                        label = f"t={thresh:.2f} buf={buf} sm={smooth} bc={bconf}"
                        latency_ms = round(buf * FRAME_MS * smooth)
                        print(f"  [{combo_num}/{total_combos}] {label} (lat~{latency_ms}ms) ... ",
                              end="", flush=True)
                        t0 = time.time()

                        metrics, agreement, changes = run_single_config(
                            audio, vad_fn, thresh, buf, smooth, bconf,
                            sb_timeline, duration_ms)
                        elapsed = time.time() - t0

                        row = {
                            "clip": clip_num,
                            "threshold": thresh,
                            "buffer": buf,
                            "smoothing": smooth,
                            "bconf": bconf,
                            "latency_ms": latency_ms,
                            "wta": metrics["wta"],
                            "wtb": metrics["wtb"],
                            "tst": metrics.get("tst", 0),
                            "trt": metrics["trt"],
                            "tct": metrics["tct"],
                            "bfst": metrics["bfst"],
                            "sb_wta": sb_wta,
                            "sb_wtb": sb_wtb,
                            "sb_tst": sb_tst,
                            "agreement_pct": round(agreement, 1),
                            "speaker_changes": changes,
                            "time_s": round(elapsed, 1),
                        }
                        results.append(row)

                        print(f"WTA={format_ms(row['wta'])} WTB={format_ms(row['wtb'])} "
                              f"Agree={row['agreement_pct']:.1f}% Ch={changes} ({elapsed:.1f}s)")

    return results


def print_top_results(results, clip_nums, n=15):
    """Print top N configs by average agreement across clips."""
    if not results:
        return

    # Group by config key
    by_config = {}
    for r in results:
        key = (r["threshold"], r["buffer"], r["smoothing"], r["bconf"])
        by_config.setdefault(key, []).append(r)

    # Compute average agreement, and collect per-clip details
    ranked = []
    for key, rows in by_config.items():
        avg_agree = sum(r["agreement_pct"] for r in rows) / len(rows)
        min_agree = min(r["agreement_pct"] for r in rows)
        avg_wtb = sum(r["wtb"] for r in rows) / len(rows)
        avg_changes = sum(r["speaker_changes"] for r in rows) / len(rows)
        ranked.append({
            "key": key,
            "avg_agree": avg_agree,
            "min_agree": min_agree,
            "avg_wtb": avg_wtb,
            "avg_changes": avg_changes,
            "rows": rows,
        })

    # Sort by avg agreement (descending), then min agreement, then fewer changes
    ranked.sort(key=lambda x: (-x["avg_agree"], -x["min_agree"], x["avg_changes"]))

    print(f"\n\n{'=' * 100}")
    print(f"  TOP {n} CONFIGS BY AVERAGE AGREEMENT")
    print(f"{'=' * 100}")

    header = (f"{'Rank':<5s} {'Thresh':>6s} {'Buf':>4s} {'Sm':>3s} {'BC':>3s} "
              f"{'Lat':>6s}  {'AvgAgr%':>7s} {'MinAgr%':>7s}  "
              f"{'AvgWTB':>8s} {'AvgCh':>6s}")
    print(header)
    print("-" * len(header))

    for i, entry in enumerate(ranked[:n], 1):
        thresh, buf, smooth, bconf = entry["key"]
        latency_ms = round(buf * FRAME_MS * smooth)
        print(f"{i:<5d} {thresh:>6.2f} {buf:>4d} {smooth:>3d} {bconf:>3d} "
              f"{latency_ms:>5d}ms  {entry['avg_agree']:>6.1f}%  {entry['min_agree']:>6.1f}%  "
              f"{format_ms(entry['avg_wtb']):>8s} {entry['avg_changes']:>6.1f}")

        # Per-clip details
        for r in entry["rows"]:
            mfcc_ref = MFCC_BEST.get(r["clip"], {})
            mfcc_agree = mfcc_ref.get("agree", 0)
            delta = r["agreement_pct"] - mfcc_agree
            print(f"      clip{r['clip']}: Agree={r['agreement_pct']:.1f}% "
                  f"(MFCC={mfcc_agree}%, delta={delta:+.1f}%) "
                  f"WTA={format_ms(r['wta'])} WTB={format_ms(r['wtb'])} "
                  f"Ch={r['speaker_changes']}")

    # Compare best neural vs MFCC
    if ranked:
        best = ranked[0]
        print(f"\n{'=' * 60}")
        print(f"  BEST NEURAL vs BEST MFCC")
        print(f"{'=' * 60}")
        thresh, buf, smooth, bconf = best["key"]
        print(f"  Neural: t={thresh:.2f} buf={buf} sm={smooth} bc={bconf} "
              f"avg_agree={best['avg_agree']:.1f}%")
        mfcc_avg = sum(MFCC_BEST[c]["agree"] for c in clip_nums if c in MFCC_BEST) / len(clip_nums)
        print(f"  MFCC:   buf=8 sm=6 bc=2 avg_agree={mfcc_avg:.1f}%")
        print(f"  Improvement: {best['avg_agree'] - mfcc_avg:+.1f}%")

    return ranked


def print_trt_invariant(results):
    """Check TRT = TCT + BFST invariant."""
    print(f"\nTRT = TCT + BFST check:")
    all_pass = True
    fail_count = 0
    for r in results:
        ok = abs(r["trt"] - r["tct"] - r["bfst"]) < 1
        if not ok:
            all_pass = False
            fail_count += 1

    if all_pass:
        print(f"  ALL {len(results)} configs PASS")
    else:
        print(f"  {fail_count}/{len(results)} configs FAIL")
        for r in results:
            if abs(r["trt"] - r["tct"] - r["bfst"]) >= 1:
                print(f"    clip{r['clip']} t={r['threshold']} buf={r['buffer']} "
                      f"sm={r['smoothing']} bc={r['bconf']}: "
                      f"TRT={format_ms(r['trt'])} TCT={format_ms(r['tct'])} "
                      f"BFST={format_ms(r['bfst'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep neural engine parameters on real audio clips")
    parser.add_argument("--clips", type=int, nargs="+", default=[1, 2],
                        help="Clip numbers to test (default: 1 2)")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip SpeechBrain diarization")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sweep (fewer combos)")
    parser.add_argument("--top", type=int, default=15,
                        help="Number of top configs to show (default: 15)")
    args = parser.parse_args()

    for c in args.clips:
        if c not in CLIPS:
            print(f"ERROR: Unknown clip {c}. Valid: {list(CLIPS.keys())}")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Select sweep ranges
    if args.quick:
        thresholds = QUICK_THRESHOLDS
        buffers = QUICK_BUFFERS
        smoothing_vals = QUICK_SMOOTHING
        bconf_vals = QUICK_BCONF
    else:
        thresholds = THRESHOLDS
        buffers = BUFFERS
        smoothing_vals = SMOOTHING
        bconf_vals = BCONF

    total = len(thresholds) * len(buffers) * len(smoothing_vals) * len(bconf_vals)
    print(f"=== Neural Engine Parameter Sweep ===")
    print(f"Clips: {args.clips}")
    print(f"Thresholds: {thresholds}")
    print(f"Buffers: {buffers} (frames)")
    print(f"Smoothing: {smoothing_vals}")
    print(f"B-confirm: {bconf_vals}")
    print(f"Total combos per clip: {total}")
    print(f"SpeechBrain: {'yes' if not args.no_diarize else 'no'}")

    # Initialize VAD
    print(f"\nInitializing Silero VAD (with debounce)...")
    vad_fn = create_silero_vad()

    # Initialize SpeechBrain
    diarizer = None
    if not args.no_diarize:
        print(f"Initializing SpeechBrain diarizer...")
        diarizer = SpeechBrainDiarizer()

    t_total = time.time()

    results = run_sweep(args.clips, vad_fn, diarizer,
                        thresholds, buffers, smoothing_vals, bconf_vals)

    total_time = time.time() - t_total
    print(f"\nTotal sweep time: {total_time:.1f}s ({total_time / 60:.1f}min)")

    # Analysis
    print_trt_invariant(results)
    print_top_results(results, args.clips, n=args.top)

    # Save results
    json_path = os.path.join(RESULTS_DIR, "neural_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
