#!/usr/bin/env python3
"""Sweep feature enrichment configs (CMVN, deltas, delta-deltas) on real audio clips.

Phase 2A: CMVN window sweep — [0, 15, 30, 50] with static features
Phase 2B: Full feature sweep — 6 configs (static/delta/delta+dd × CMVN on/off)

All tests use buf=8, smooth=6, bconf=2 (Wave 1 best config).

Usage:
  python3 scripts/run_feature_sweep.py                    # full sweep (2A + 2B)
  python3 scripts/run_feature_sweep.py --phase 2a         # CMVN sweep only
  python3 scripts/run_feature_sweep.py --phase 2b         # feature sweep only (uses best CMVN from 2A)
  python3 scripts/run_feature_sweep.py --phase 2b --cmvn-best 30  # feature sweep with explicit CMVN window
  python3 scripts/run_feature_sweep.py --clips 1           # single clip
  python3 scripts/run_feature_sweep.py --no-diarize        # skip SpeechBrain
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    DEFAULT_B_CONFIRM_FRAMES,
    FRAME_MS,
    PipelineSimulator,
    SPEECH_SEGMENT_FRAMES,
    SpeechBrainDiarizer,
    compare_timelines,
    create_silero_vad,
    extract_vad_segments,
    format_ms,
    load_audio,
    SAMPLE_RATE,
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

# Wave 1 best config (fixed for all feature sweep runs)
BUF = 8
SMOOTH = 6
BCONF = 2

# Phase 2A: CMVN window candidates
CMVN_WINDOWS = [0, 15, 30, 50]

# Phase 2B: Feature configs — (label, feature_mode, cmvn_window_placeholder)
# cmvn_window will be filled in with best from 2A (or 0)
FEATURE_CONFIGS_TEMPLATE = [
    ("baseline",       "static",   0),
    ("cmvn-only",      "static",   -1),  # -1 = use best CMVN
    ("delta-only",     "delta",    0),
    ("delta+cmvn",     "delta",    -1),
    ("delta+dd",       "delta+dd", 0),
    ("delta+dd+cmvn",  "delta+dd", -1),
]


def run_single_config(audio, vad_fn, feature_mode, cmvn_window, sb_timeline, duration_ms):
    """Run pipeline with given feature config, return metrics + comparison."""
    pipeline = PipelineSimulator(
        vad_fn,
        smoothing_window=SMOOTH,
        speech_segment_frames=BUF,
        b_confirm_frames=BCONF,
        feature_mode=feature_mode,
        cmvn_window=cmvn_window,
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

    return metrics, agreement_pct, speaker_changes, pipeline


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


def run_cmvn_sweep(clip_nums, vad_fn, diarizer):
    """Phase 2A: Sweep CMVN windows with static features."""
    results = []

    for clip_num in clip_nums:
        clip_path = os.path.join(CLIP_DIR, CLIPS[clip_num])
        if not os.path.exists(clip_path):
            print(f"WARNING: Clip {clip_num} not found, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  PHASE 2A \u2014 CMVN SWEEP \u2014 CLIP {clip_num}: {CLIP_DESCRIPTIONS.get(clip_num, '?')}")
        print(f"{'=' * 60}")

        audio = load_audio(clip_path)
        duration_ms = len(audio) / SAMPLE_RATE * 1000

        sb_timeline, sb_wta, sb_wtb, sb_tst = get_sb_reference(audio, vad_fn, diarizer)
        if sb_timeline:
            print(f"  SB: WTA={format_ms(sb_wta)}, WTB={format_ms(sb_wtb)}, TST={format_ms(sb_tst)}")

        for cmvn_win in CMVN_WINDOWS:
            label = f"cmvn={cmvn_win}"
            print(f"  {label} ... ", end="", flush=True)
            t0 = time.time()

            metrics, agreement, changes, _ = run_single_config(
                audio, vad_fn, "static", cmvn_win, sb_timeline, duration_ms)
            elapsed = time.time() - t0

            row = {
                "phase": "2A",
                "clip": clip_num,
                "label": label,
                "feature_mode": "static",
                "cmvn_window": cmvn_win,
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
                  f"Agree={row['agreement_pct']:.1f}% Changes={changes} ({elapsed:.1f}s)")

    return results


def pick_best_cmvn(cmvn_results):
    """Pick CMVN window with best average agreement across clips."""
    if not cmvn_results:
        return 0

    by_window = {}
    for r in cmvn_results:
        w = r["cmvn_window"]
        by_window.setdefault(w, []).append(r["agreement_pct"])

    best_window = 0
    best_avg = 0
    for w, pcts in sorted(by_window.items()):
        avg = sum(pcts) / len(pcts)
        print(f"  CMVN={w}: avg agreement={avg:.1f}%")
        if avg > best_avg:
            best_avg = avg
            best_window = w

    print(f"  -> Best CMVN window: {best_window} (avg agreement {best_avg:.1f}%)")
    return best_window


def run_feature_sweep(clip_nums, vad_fn, diarizer, best_cmvn):
    """Phase 2B: Sweep feature configs with best CMVN from 2A."""
    results = []

    # Build concrete configs from template
    configs = []
    for label, fmode, cmvn_placeholder in FEATURE_CONFIGS_TEMPLATE:
        cmvn = best_cmvn if cmvn_placeholder == -1 else cmvn_placeholder
        # Skip cmvn-only and *+cmvn variants if best_cmvn is 0 (CMVN didn't help)
        if cmvn_placeholder == -1 and best_cmvn == 0:
            print(f"  Skipping '{label}' (CMVN best=0, same as no-CMVN variant)")
            continue
        configs.append((label, fmode, cmvn))

    for clip_num in clip_nums:
        clip_path = os.path.join(CLIP_DIR, CLIPS[clip_num])
        if not os.path.exists(clip_path):
            print(f"WARNING: Clip {clip_num} not found, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  PHASE 2B \u2014 FEATURE SWEEP \u2014 CLIP {clip_num}: {CLIP_DESCRIPTIONS.get(clip_num, '?')}")
        print(f"{'=' * 60}")

        audio = load_audio(clip_path)
        duration_ms = len(audio) / SAMPLE_RATE * 1000

        sb_timeline, sb_wta, sb_wtb, sb_tst = get_sb_reference(audio, vad_fn, diarizer)
        if sb_timeline:
            print(f"  SB: WTA={format_ms(sb_wta)}, WTB={format_ms(sb_wtb)}, TST={format_ms(sb_tst)}")

        for label, fmode, cmvn_win in configs:
            dims = {"static": 12, "delta": 24, "delta+dd": 36}[fmode]
            tag = f"{label} ({fmode}, cmvn={cmvn_win}, {dims}d)"
            print(f"  {tag} ... ", end="", flush=True)
            t0 = time.time()

            metrics, agreement, changes, _ = run_single_config(
                audio, vad_fn, fmode, cmvn_win, sb_timeline, duration_ms)
            elapsed = time.time() - t0

            row = {
                "phase": "2B",
                "clip": clip_num,
                "label": label,
                "feature_mode": fmode,
                "cmvn_window": cmvn_win,
                "dims": dims,
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
                  f"Agree={row['agreement_pct']:.1f}% Changes={changes} ({elapsed:.1f}s)")

    return results


def print_results_table(results, title):
    """Print results as a formatted table."""
    if not results:
        return
    print(f"\n\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    header = (f"{'Clip':<6s} {'Label':<16s} {'Mode':<10s} {'CMVN':>5s} {'Dims':>5s}  "
              f"{'WTA':>7s} {'WTB':>7s} {'TST':>6s}  "
              f"{'SB_WTA':>7s} {'SB_WTB':>7s}  "
              f"{'Agree%':>7s}  {'Changes':>8s}")
    print(header)
    print("-" * len(header))

    current_clip = None
    for r in results:
        if r["clip"] != current_clip:
            if current_clip is not None:
                print()
            current_clip = r["clip"]

        dims = r.get("dims", 12)
        print(f"clip{r['clip']:<2d} {r['label']:<16s} {r['feature_mode']:<10s} "
              f"{r['cmvn_window']:>5d} {dims:>5d}  "
              f"{format_ms(r['wta']):>7s} {format_ms(r['wtb']):>7s} {format_ms(r['tst']):>6s}  "
              f"{format_ms(r['sb_wta']):>7s} {format_ms(r['sb_wtb']):>7s}  "
              f"{r['agreement_pct']:>6.1f}%  "
              f"{r['speaker_changes']:>8d}")

    print()

    # TRT invariant check
    print("TRT = TCT + BFST check:")
    all_pass = True
    for r in results:
        ok = abs(r["trt"] - r["tct"] - r["bfst"]) < 1
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  Clip {r['clip']} {r['label']}: "
              f"TRT={format_ms(r['trt'])} TCT={format_ms(r['tct'])} BFST={format_ms(r['bfst'])} {status}")

    if all_pass:
        print("  ALL PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep feature enrichment configs (CMVN, deltas, delta-deltas) on real audio clips")
    parser.add_argument("--clips", type=int, nargs="+", default=[1, 2],
                        help="Clip numbers to test (default: 1 2)")
    parser.add_argument("--phase", choices=["2a", "2b", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--cmvn-best", type=int, default=None,
                        help="Override best CMVN window for Phase 2B (skip 2A auto-pick)")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip SpeechBrain diarization")
    args = parser.parse_args()

    for c in args.clips:
        if c not in CLIPS:
            print(f"ERROR: Unknown clip {c}. Valid: {list(CLIPS.keys())}")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Feature Enrichment Sweep (Wave 2) ===")
    print(f"Clips: {args.clips}")
    print(f"Phase: {args.phase}")
    print(f"Fixed config: buf={BUF}, smooth={SMOOTH}, bconf={BCONF}")
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
    all_results = []
    best_cmvn = 0

    # Phase 2A: CMVN sweep
    if args.phase in ("2a", "all"):
        print(f"\n{'#' * 70}")
        print(f"  PHASE 2A: CMVN WINDOW SWEEP")
        print(f"  Windows: {CMVN_WINDOWS}")
        print(f"{'#' * 70}")

        cmvn_results = run_cmvn_sweep(args.clips, vad_fn, diarizer)
        all_results.extend(cmvn_results)
        print_results_table(cmvn_results, "PHASE 2A \u2014 CMVN SWEEP RESULTS")

        # Pick best CMVN
        print("\nSelecting best CMVN window:")
        best_cmvn = pick_best_cmvn(cmvn_results)

    # Override best CMVN if specified
    if args.cmvn_best is not None:
        best_cmvn = args.cmvn_best
        print(f"\nUsing override CMVN window: {best_cmvn}")

    # Phase 2B: Feature sweep
    if args.phase in ("2b", "all"):
        print(f"\n{'#' * 70}")
        print(f"  PHASE 2B: FEATURE MODE SWEEP")
        print(f"  Best CMVN: {best_cmvn}")
        print(f"{'#' * 70}")

        feature_results = run_feature_sweep(args.clips, vad_fn, diarizer, best_cmvn)
        all_results.extend(feature_results)
        print_results_table(feature_results, "PHASE 2B \u2014 FEATURE SWEEP RESULTS")

    total_time = time.time() - t_total
    print(f"\nTotal sweep time: {total_time:.1f}s")

    # Save all results
    json_path = os.path.join(RESULTS_DIR, "feature_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
