#!/usr/bin/env python3
"""Sweep buffer size, smoothing window, and B-confirmation on real audio clips.

Tests combinations of speech_segment_frames (buffer), smoothing_window, and
b_confirm_frames on clips 1 and 2, comparing against SpeechBrain ECAPA-TDNN.

Default test matrix (all with b_confirm_frames=2):
  baseline: buffer=5, smoothing=6
  combo 1:  buffer=8, smoothing=4
  combo 2:  buffer=8, smoothing=6
  combo 3:  buffer=10, smoothing=4
  combo 4:  buffer=10, smoothing=6

Usage:
  python3 scripts/run_smoothing_sweep.py
  python3 scripts/run_smoothing_sweep.py --clips 1
  python3 scripts/run_smoothing_sweep.py --no-diarize
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

# Default sweep combos: (buffer, smoothing, b_confirm)
DEFAULT_COMBOS = [
    (5, 6, 2),    # baseline: current best smoothing + B confirmation
    (8, 4, 2),    # larger buffer, smaller smoothing
    (8, 6, 2),    # larger buffer, same smoothing
    (10, 4, 2),   # largest buffer, smaller smoothing
    (10, 6, 2),   # largest buffer, same smoothing
]


def run_sweep(clip_nums, combos, vad_fn, diarizer):
    """Run the full sweep and return list of result dicts."""
    results = []

    for clip_num in clip_nums:
        clip_file = CLIPS[clip_num]
        clip_path = os.path.join(CLIP_DIR, clip_file)
        if not os.path.exists(clip_path):
            print(f"WARNING: Clip {clip_num} not found: {clip_path}, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  CLIP {clip_num}: {CLIP_DESCRIPTIONS.get(clip_num, '?')}")
        print(f"{'=' * 60}")

        # Load audio once per clip
        audio = load_audio(clip_path)
        duration_s = len(audio) / SAMPLE_RATE
        duration_ms = duration_s * 1000
        print(f"  Duration: {duration_s:.1f}s")

        # Run SpeechBrain once per clip
        sb_wta_ms = 0
        sb_wtb_ms = 0
        sb_tst_ms = 0
        sb_timeline = None
        if diarizer is not None:
            print(f"  Running SpeechBrain diarization (once per clip)...")
            baseline = PipelineSimulator(vad_fn, smoothing_window=1,
                                         speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                         b_confirm_frames=1)
            baseline.run(audio)
            vad_segments = extract_vad_segments(baseline.timeline)
            sb_timeline = diarizer.diarize(audio, vad_segments)
            baseline_cmp = compare_timelines(baseline.timeline, sb_timeline, duration_ms)
            sb_wta_ms = baseline_cmp.get("sb_wta_ms", 0)
            sb_wtb_ms = baseline_cmp.get("sb_wtb_ms", 0)
            sb_tst_ms = baseline_cmp.get("sb_tst_ms", 0)
            print(f"  SB: WTA={format_ms(sb_wta_ms)}, WTB={format_ms(sb_wtb_ms)}, TST={format_ms(sb_tst_ms)}")

        # Sweep combos
        for buf, smooth, bconf in combos:
            label = f"buf={buf} smooth={smooth} bconf={bconf}"
            approx_ms = round(buf * FRAME_MS)
            print(f"  {label} (~{approx_ms}ms buffer) ... ", end="", flush=True)
            t0 = time.time()

            pipeline = PipelineSimulator(vad_fn, smoothing_window=smooth,
                                         speech_segment_frames=buf,
                                         b_confirm_frames=bconf)
            metrics = pipeline.run(audio)

            # Compare with SB if available
            agreement_pct = 0
            if sb_timeline is not None:
                cmp = compare_timelines(pipeline.timeline, sb_timeline, duration_ms)
                agreement_pct = cmp.get("agreement_pct", 0)

            elapsed = time.time() - t0

            # Count speaker changes
            speaker_changes = 0
            prev_label = None
            for _, _, lbl in pipeline.timeline:
                if lbl in ("A", "B") and lbl != prev_label and prev_label in ("A", "B"):
                    speaker_changes += 1
                if lbl in ("A", "B"):
                    prev_label = lbl

            row = {
                "clip": clip_num,
                "buffer": buf,
                "smoothing": smooth,
                "b_confirm": bconf,
                "buffer_ms": approx_ms,
                "wta": metrics["wta"],
                "wtb": metrics["wtb"],
                "tst": metrics.get("tst", 0),
                "trt": metrics["trt"],
                "tct": metrics["tct"],
                "bfst": metrics["bfst"],
                "ovt": metrics.get("ovt", 0),
                "sb_wta": sb_wta_ms,
                "sb_wtb": sb_wtb_ms,
                "sb_tst": sb_tst_ms,
                "agreement_pct": round(agreement_pct, 1),
                "n_decisions": len(pipeline.similarity_trace),
                "speaker_changes": speaker_changes,
                "time_s": round(elapsed, 1),
            }
            results.append(row)

            print(f"WTA={format_ms(row['wta'])} WTB={format_ms(row['wtb'])} "
                  f"TST={format_ms(row['tst'])} Agree={row['agreement_pct']:.1f}% "
                  f"Changes={speaker_changes} ({elapsed:.1f}s)")

    return results


def print_summary_table(results):
    """Print the sweep results as a formatted table."""
    print(f"\n\n{'=' * 120}")
    print(f"  BUFFER / SMOOTHING SWEEP RESULTS")
    print(f"{'=' * 120}")

    header = (f"{'Clip':<6s} {'Buf':>4s} {'Smth':>5s} {'BCnf':>5s} {'~ms':>5s}  "
              f"{'WTA':>7s} {'WTB':>7s} {'TST':>6s}  "
              f"{'SB_WTA':>7s} {'SB_WTB':>7s} {'SB_TST':>6s}  "
              f"{'Agree%':>7s}  {'Changes':>8s}")
    print(header)
    print("-" * len(header))

    current_clip = None
    for r in results:
        if r["clip"] != current_clip:
            if current_clip is not None:
                print()
            current_clip = r["clip"]

        print(f"clip{r['clip']:<2d} {r['buffer']:>4d} {r['smoothing']:>5d} {r['b_confirm']:>5d} "
              f"{r['buffer_ms']:>4d}ms  "
              f"{format_ms(r['wta']):>7s} {format_ms(r['wtb']):>7s} {format_ms(r['tst']):>6s}  "
              f"{format_ms(r['sb_wta']):>7s} {format_ms(r['sb_wtb']):>7s} {format_ms(r['sb_tst']):>6s}  "
              f"{r['agreement_pct']:>6.1f}%  "
              f"{r['speaker_changes']:>8d}")

    print()

    # TRT invariant check
    print("TRT = TCT + BFST check:")
    for r in results:
        ok = abs(r["trt"] - r["tct"] - r["bfst"]) < 1
        print(f"  Clip {r['clip']} buf={r['buffer']} smooth={r['smoothing']} bconf={r['b_confirm']}: "
              f"TRT={format_ms(r['trt'])} TCT={format_ms(r['tct'])} BFST={format_ms(r['bfst'])} "
              f"{'PASS' if ok else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep buffer/smoothing/B-confirmation combos on real audio clips")
    parser.add_argument("--clips", type=int, nargs="+", default=[1, 2],
                        help="Clip numbers to test (default: 1 2)")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip SpeechBrain diarization")
    args = parser.parse_args()

    for c in args.clips:
        if c not in CLIPS:
            print(f"ERROR: Unknown clip {c}. Valid: {list(CLIPS.keys())}")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    combos = DEFAULT_COMBOS
    print(f"=== Buffer / Smoothing / B-Confirmation Sweep ===")
    print(f"Clips: {args.clips}")
    print(f"Combos ({len(combos)}):")
    for i, (buf, smooth, bconf) in enumerate(combos):
        label = "baseline" if i == 0 else f"combo {i}"
        print(f"  {label}: buffer={buf}, smoothing={smooth}, b_confirm={bconf}")
    print(f"SpeechBrain: {'yes' if not args.no_diarize else 'no'}")

    print(f"\nInitializing Silero VAD (with debounce)...")
    vad_fn = create_silero_vad()

    diarizer = None
    if not args.no_diarize:
        print(f"Initializing SpeechBrain diarizer...")
        diarizer = SpeechBrainDiarizer()

    t0 = time.time()
    results = run_sweep(args.clips, combos, vad_fn, diarizer)
    total_time = time.time() - t0

    print_summary_table(results)
    print(f"Total sweep time: {total_time:.1f}s")

    json_path = os.path.join(RESULTS_DIR, "buffer_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
