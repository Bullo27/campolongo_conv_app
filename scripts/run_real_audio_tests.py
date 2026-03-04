#!/usr/bin/env python3
"""Batch test runner for real-world audio clips.

Runs each clip through:
  1. Our pipeline (Silero VAD + MFCC speaker ID)
  2. SpeechBrain ECAPA-TDNN diarization (second opinion)
  3. Whisper transcription

Usage:
  python3 scripts/run_real_audio_tests.py                          # clips 1-3 (Phase B)
  python3 scripts/run_real_audio_tests.py --clips 1 2 3 4 5       # all clips
  python3 scripts/run_real_audio_tests.py --clips 1                # single clip
  python3 scripts/run_real_audio_tests.py --no-whisper             # skip whisper
  python3 scripts/run_real_audio_tests.py --no-diarize             # skip speechbrain
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    DEFAULT_B_CONFIRM_FRAMES,
    DEFAULT_CMVN_WINDOW,
    DEFAULT_FEATURE_MODE,
    DEFAULT_SILENCE_DEBOUNCE_MS,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_SPEECH_DEBOUNCE_MS,
    FEATURE_MODES,
    FRAME_MS,
    PipelineSimulator,
    SPEECH_SEGMENT_FRAMES,
    SpeechBrainDiarizer,
    compare_timelines,
    create_silero_vad,
    extract_vad_segments,
    format_ms,
    load_audio,
    run_whisper,
    SAMPLE_RATE,
)

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
RESULTS_DIR = os.path.join(CLIP_DIR, "results")

CLIPS = {
    1: "clip1_theater_conversation.mp3",
    2: "clip2_tv_study_interview.mp3",
    3: "clip3_interview_background_music.mp3",
    4: "clip4_street_interview_loud_background_chatter.mp3",
    5: "clip5_multi_location_interview_music_and_chatter_background.mp3",
}

CLIP_DESCRIPTIONS = {
    1: "Theater conversation (clean)",
    2: "TV studio interview (clean)",
    3: "Interview + background music",
    4: "Street interview + loud chatter",
    5: "Multi-location + music & chatter",
}


def process_clip(clip_num, audio_path, vad_fn, diarizer, run_whisper_flag,
                 smoothing_window=DEFAULT_SMOOTHING_WINDOW,
                 speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                 b_confirm_frames=DEFAULT_B_CONFIRM_FRAMES,
                 feature_mode=DEFAULT_FEATURE_MODE,
                 cmvn_window=DEFAULT_CMVN_WINDOW):
    """Process a single clip and return results dict."""
    print(f"\n{'#' * 70}")
    print(f"  CLIP {clip_num}: {CLIP_DESCRIPTIONS.get(clip_num, '?')}")
    print(f"  File: {os.path.basename(audio_path)}")
    print(f"{'#' * 70}")

    t0 = time.time()

    print(f"\n  Loading audio...")
    audio = load_audio(audio_path)
    duration_s = len(audio) / SAMPLE_RATE
    duration_ms = duration_s * 1000
    print(f"  Duration: {duration_s:.1f}s")

    print(f"  Running our pipeline...")
    pipeline = PipelineSimulator(vad_fn, smoothing_window=smoothing_window,
                                     speech_segment_frames=speech_segment_frames,
                                     b_confirm_frames=b_confirm_frames,
                                     feature_mode=feature_mode,
                                     cmvn_window=cmvn_window)
    pipeline_metrics = pipeline.run(audio)

    result = {
        "clip": clip_num,
        "file": os.path.basename(audio_path),
        "description": CLIP_DESCRIPTIONS.get(clip_num, ""),
        "duration_s": round(duration_s, 1),
        "pipeline_metrics": pipeline_metrics,
        "similarity_trace": [
            {"time_ms": t[0], "sim_a": round(t[1], 4), "sim_b": round(t[2], 4),
             "decision": t[3], "ambiguous": t[4]}
            for t in pipeline.similarity_trace
        ],
        "timeline": [
            {"start_ms": s, "end_ms": e, "label": l}
            for s, e, l in pipeline.timeline
        ],
    }

    if diarizer is not None:
        print(f"  Running speechbrain diarization...")
        vad_segments = extract_vad_segments(pipeline.timeline)
        sb_timeline = diarizer.diarize(audio, vad_segments)
        comparison = compare_timelines(pipeline.timeline, sb_timeline, duration_ms)
        result["sb_timeline"] = [
            {"start_ms": s, "end_ms": e, "label": l}
            for s, e, l in sb_timeline
        ]
        result["comparison"] = comparison
        print(f"  Agreement: {comparison['agreement_pct']:.1f}%")

    if run_whisper_flag:
        print(f"  Running Whisper transcription...")
        try:
            whisper_info = run_whisper(audio_path)
            result["whisper"] = whisper_info
        except Exception as e:
            print(f"  Whisper failed: {e}")

    elapsed = time.time() - t0
    result["processing_time_s"] = round(elapsed, 1)
    print(f"  Processing time: {elapsed:.1f}s")

    return result


def print_summary_table(results):
    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 100}")

    header = (f"{'Clip':<6s} {'Dur':>5s}  {'Agree%':>7s}  "
              f"{'OurWTA':>7s} {'OurWTB':>7s}  "
              f"{'SB_WTA':>7s} {'SB_WTB':>7s}  "
              f"{'OVT':>6s} {'Ambig%':>7s}  "
              f"{'#Decs':>6s}")
    print(header)
    print("-" * len(header))

    for r in results:
        clip = r["clip"]
        dur = f"{r['duration_s']:.0f}s"
        pm = r["pipeline_metrics"]
        our_wta = format_ms(pm["wta"])
        our_wtb = format_ms(pm["wtb"])
        ovt = format_ms(pm.get("ovt", 0))

        comp = r.get("comparison", {})
        agree = f"{comp.get('agreement_pct', 0):.1f}%" if comp else "N/A"
        sb_wta = format_ms(comp.get("sb_wta_ms", 0)) if comp else "N/A"
        sb_wtb = format_ms(comp.get("sb_wtb_ms", 0)) if comp else "N/A"

        n_decisions = len(r.get("similarity_trace", []))
        n_ambig = sum(1 for t in r.get("similarity_trace", []) if t["ambiguous"])
        ambig_pct = f"{n_ambig / n_decisions * 100:.1f}%" if n_decisions > 0 else "N/A"

        print(f"clip{clip:<2d} {dur:>5s}  {agree:>7s}  "
              f"{our_wta:>7s} {our_wtb:>7s}  "
              f"{sb_wta:>7s} {sb_wtb:>7s}  "
              f"{ovt:>6s} {ambig_pct:>7s}  "
              f"{n_decisions:>6d}")

    print()

    for r in results:
        clip = r["clip"]
        pm = r["pipeline_metrics"]
        print(f"  Clip {clip}: TRT={format_ms(pm['trt'])}  "
              f"TCT={format_ms(pm['tct'])}  BFST={format_ms(pm['bfst'])}  "
              f"TST={format_ms(pm.get('tst', 0))}  "
              f"TRT=TCT+BFST: {'PASS' if abs(pm['trt'] - pm['tct'] - pm['bfst']) < 1 else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Batch test runner for real audio clips")
    parser.add_argument("--clips", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--no-whisper", action="store_true")
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--silence-debounce-ms", type=float, default=DEFAULT_SILENCE_DEBOUNCE_MS)
    parser.add_argument("--speech-debounce-ms", type=float, default=DEFAULT_SPEECH_DEBOUNCE_MS)
    parser.add_argument("--no-debounce", action="store_true")
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--speech-buffer", type=int, default=SPEECH_SEGMENT_FRAMES)
    parser.add_argument("--b-confirm-frames", type=int, default=DEFAULT_B_CONFIRM_FRAMES)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=DEFAULT_FEATURE_MODE)
    parser.add_argument("--cmvn-window", type=int, default=DEFAULT_CMVN_WINDOW)
    args = parser.parse_args()

    for c in args.clips:
        if c not in CLIPS:
            print(f"ERROR: Unknown clip number {c}. Valid: {list(CLIPS.keys())}")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Real-World Audio Test Runner ===")
    print(f"Clips: {args.clips}")
    print(f"Diarize: {'yes' if not args.no_diarize else 'no'}")
    print(f"Whisper: {'yes' if not args.no_whisper else 'no'}")
    print(f"Results dir: {RESULTS_DIR}")

    sil_db = 0 if args.no_debounce else args.silence_debounce_ms
    spe_db = 0 if args.no_debounce else args.speech_debounce_ms
    print(f"\nInitializing Silero VAD (threshold={args.vad_threshold}, "
          f"silence_debounce={sil_db}ms, speech_debounce={spe_db}ms)...")
    vad_fn = create_silero_vad(
        vad_threshold=args.vad_threshold,
        silence_debounce_ms=sil_db,
        speech_debounce_ms=spe_db,
    )

    diarizer = None
    if not args.no_diarize:
        print(f"Initializing SpeechBrain diarizer...")
        diarizer = SpeechBrainDiarizer()

    results = []
    for clip_num in args.clips:
        clip_file = CLIPS[clip_num]
        clip_path = os.path.join(CLIP_DIR, clip_file)
        if not os.path.exists(clip_path):
            print(f"\nWARNING: Clip file not found: {clip_path}, skipping.")
            continue

        result = process_clip(clip_num, clip_path, vad_fn, diarizer, not args.no_whisper,
                              smoothing_window=args.smoothing_window,
                              speech_segment_frames=args.speech_buffer,
                              b_confirm_frames=args.b_confirm_frames,
                              feature_mode=args.feature_mode,
                              cmvn_window=args.cmvn_window)
        results.append(result)

        json_path = os.path.join(RESULTS_DIR, f"clip{clip_num}_results.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {json_path}")

    print_summary_table(results)

    combined_path = os.path.join(RESULTS_DIR, "combined_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Combined results saved: {combined_path}")


if __name__ == "__main__":
    main()
