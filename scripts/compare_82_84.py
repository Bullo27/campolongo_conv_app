#!/usr/bin/env python3
"""Quick comparison: dual_t=0.82 vs 0.84 on clean clips against ground truth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    SAMPLE_RATE, WeSpeakerEmbedder, load_audio, format_ms,
    PipelineSimulator, SpeechBrainDiarizer, create_silero_vad,
    extract_vad_segments, compare_timelines, SPEECH_SEGMENT_FRAMES,
)
from strategy_comparison import (
    run_pipeline_segments, apply_strategy, build_timeline,
    compute_dual_metrics, relaxed_agreement,
)

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}
GT = {
    1: {"speech_a": 171000, "speech_b": 35000, "overlap": 10000, "silence": 6000},
    2: {"speech_a": 195000, "speech_b": 579000, "overlap": 35000, "silence": 12000},
}

def main():
    embedder = WeSpeakerEmbedder()

    for cn in [1, 2]:
        path = CLIPS[cn]
        if not os.path.exists(path):
            continue
        audio = load_audio(path)
        dur_ms = len(audio) / SAMPLE_RATE * 1000
        gt = GT[cn]

        print(f"\n{'='*90}")
        print(f"  CLIP {cn} — {dur_ms/1000:.1f}s")
        print(f"{'='*90}")

        # Pipeline
        segments = run_pipeline_segments(audio, embedder)
        baseline_dec = apply_strategy(segments, {"type": "baseline"})
        baseline_tl = build_timeline(segments, baseline_dec, dur_ms)

        # SpeechBrain
        diarizer = SpeechBrainDiarizer()
        vad_fn = create_silero_vad()
        sim = PipelineSimulator(vad_fn, smoothing_window=1,
                                speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                b_confirm_frames=1)
        sim.run(audio)
        vad_speech = extract_vad_segments(sim.timeline)
        sb_tl = diarizer.diarize(audio, vad_speech)

        # Label map
        baseline_speech = [(s, e, l) for s, e, l in baseline_tl if l != "silence"]
        cmp = compare_timelines(baseline_speech, sb_tl, dur_ms)
        label_map = cmp["label_map"]

        # Compare thresholds
        print(f"\n  {'':>14s} {'Speech A':>10s} {'Speech B':>10s} {'Overlap':>10s} {'A share':>8s} {'Strict SB':>10s} {'Relaxed SB':>11s}")
        print(f"  {'─'*65}")

        # Ground truth
        gt_total = gt["speech_a"] + gt["speech_b"]
        print(f"  {'Ground Truth':>14s} {format_ms(gt['speech_a']):>10s} {format_ms(gt['speech_b']):>10s} "
              f"{format_ms(gt['overlap']):>10s} {100*gt['speech_a']/gt_total:>7.1f}% {'—':>10s} {'—':>11s}")

        for dual_t in [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 1.0]:
            dm = compute_dual_metrics(segments, baseline_dec, dual_t, 3)
            strict, rel = relaxed_agreement(
                segments, dm["segment_labels"], sb_tl, label_map, dur_ms)
            label = f"t={dual_t:.2f}" if dual_t < 1.0 else "No overlap"
            print(f"  {label:>14s} {format_ms(dm['speech_a']):>10s} {format_ms(dm['speech_b']):>10s} "
                  f"{format_ms(dm['overlap_time']):>10s} {dm['a_share']:>7.1f}% "
                  f"{strict:>9.1f}% {rel:>10.1f}%")

        print(f"  {'─'*65}")

    print(f"\nDone.")

if __name__ == "__main__":
    main()
