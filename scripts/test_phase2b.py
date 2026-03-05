#!/usr/bin/env python3
"""Phase 2B test: min segment length for B establishment + bconf=2.

Tests both clips with the fix and reports:
1. Per-segment log (first 25 segments) showing flush segment handling
2. Initial portion vs ground truth
3. Full clip vs SpeechBrain
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS, FRAME_SIZE, SAMPLE_RATE, SPEECH_SEGMENT_FRAMES, MIN_FLUSH_FRAMES,
    PipelineSimulator, SpeakerIdentifier, SpeechBrainDiarizer,
    WeSpeakerEmbedder, compare_timelines, create_silero_vad,
    extract_vad_segments, format_ms, load_audio,
)

import numpy as np

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")

CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

# Ground truths (user-verified)
GROUND_TRUTHS = {
    1: [
        (0, 7000, "A"),
        (7000, 20000, "B"),
        (20000, 25000, "mixed"),
        (25000, 50000, "A"),
    ],
    2: [
        (0, 7000, "A"),
        (7000, 19000, "B"),
    ],
}

THRESHOLD = 0.80
BUFFER = 47
BCONF = 2
SMOOTHING = 2


def run_verbose_pipeline(audio, vad_fn, embedder, threshold, buffer_frames, bconf):
    """Run neural pipeline with per-segment logging, including flush segment info."""
    speaker_id = SpeakerIdentifier(
        threshold=threshold, margin=0.0, b_confirm_frames=bconf,
        min_frames_for_b=buffer_frames,
    )

    n_frames = len(audio) // FRAME_SIZE
    speech_buffer = []
    speech_buffer_start_frame = 0
    timeline = []
    segment_log = []

    def identify_and_log(current_frame):
        nonlocal speech_buffer
        mid_frame = (speech_buffer_start_frame + current_frame) // 2
        time_ms = round(mid_frame * FRAME_MS)
        n_buffered = len(speech_buffer)
        is_flush = n_buffered < buffer_frames

        raw_audio = np.concatenate(speech_buffer)
        speech_buffer = []
        embedding = embedder.extract_embedding(raw_audio)
        raw_speaker = speaker_id.identify(embedding, n_frames=n_buffered)

        start_ms = round(speech_buffer_start_frame * FRAME_MS)
        end_ms = round(current_frame * FRAME_MS)
        entry = {
            "idx": len(segment_log),
            "start_ms": start_ms, "end_ms": end_ms, "mid_ms": time_ms,
            "n_frames": n_buffered,
            "duration_ms": round(n_buffered * FRAME_MS),
            "sim_a": round(speaker_id.last_sim_a, 4),
            "sim_b": round(speaker_id.last_sim_b, 4),
            "decision": raw_speaker,
            "ref_b_exists": speaker_id.ref_b is not None,
            "b_candidate_count": speaker_id._b_candidate_count,
            "is_flush": is_flush,
        }
        segment_log.append(entry)
        return raw_speaker

    def update_timeline(frame_idx, label, start_frame=None):
        if start_frame is not None:
            start_ms = round(start_frame * FRAME_MS)
        else:
            start_ms = round(frame_idx * FRAME_MS)
        end_ms = round((frame_idx + 1) * FRAME_MS)
        if timeline and timeline[-1][2] == label:
            timeline[-1] = (timeline[-1][0], end_ms, label)
        else:
            timeline.append((start_ms, end_ms, label))

    for i in range(n_frames):
        start = i * FRAME_SIZE
        frame = audio[start:start + FRAME_SIZE]
        is_speech = vad_fn(frame)

        if is_speech:
            if not speech_buffer:
                speech_buffer_start_frame = i
            speech_buffer.append(frame)
            if len(speech_buffer) >= buffer_frames:
                buf_start = speech_buffer_start_frame
                speaker = identify_and_log(i)
                update_timeline(i, speaker, start_frame=buf_start)
        else:
            if len(speech_buffer) >= MIN_FLUSH_FRAMES:
                buf_start = speech_buffer_start_frame
                speaker = identify_and_log(i)
                update_timeline(i, speaker, start_frame=buf_start)
            speech_buffer.clear()
            update_timeline(i, "silence")

    if len(speech_buffer) >= MIN_FLUSH_FRAMES:
        buf_start = speech_buffer_start_frame
        speaker = identify_and_log(n_frames - 1)
        update_timeline(n_frames - 1, speaker, start_frame=buf_start)

    return segment_log, timeline


def main():
    print(f"=== Phase 2B Test: min_frames_for_b={BUFFER}, bconf={BCONF}, t={THRESHOLD} ===\n")

    embedder = WeSpeakerEmbedder()

    for clip_num in [1, 2]:
        clip_path = CLIPS.get(clip_num)
        if not clip_path or not os.path.exists(clip_path):
            print(f"\nClip {clip_num}: not found, skipping")
            continue

        print(f"\n{'=' * 70}")
        print(f"  CLIP {clip_num}")
        print(f"{'=' * 70}")

        audio = load_audio(clip_path)
        duration_s = len(audio) / SAMPLE_RATE
        duration_ms = duration_s * 1000
        print(f"  Duration: {format_ms(duration_ms)}")

        # Our pipeline (verbose)
        print(f"  Running neural pipeline...")
        vad_fn = create_silero_vad()
        t0 = time.time()
        seg_log, our_tl = run_verbose_pipeline(
            audio, vad_fn, embedder, THRESHOLD, BUFFER, BCONF)
        elapsed = time.time() - t0
        print(f"  Pipeline done in {elapsed:.1f}s, {len(seg_log)} segments")

        # Print first 25 segments with flush indicator
        print(f"\n  {'Seg':>3s} {'Start':>7s} {'End':>7s} {'Dur':>5s} {'Frm':>4s} "
              f"{'SimA':>6s} {'SimB':>6s} {'Dec':>3s} {'RefB':>4s} {'BCnd':>4s} {'Flush':>5s}")
        print(f"  {'-' * 70}")
        for e in seg_log[:25]:
            print(f"  {e['idx']:>3d} {format_ms(e['start_ms']):>7s} {format_ms(e['end_ms']):>7s} "
                  f"{format_ms(e['duration_ms']):>5s} {e['n_frames']:>4d} "
                  f"{e['sim_a']:>6.3f} {e['sim_b']:>6.3f} "
                  f"  {e['decision']} {'yes' if e['ref_b_exists'] else 'no':>4s} "
                  f"{e['b_candidate_count']:>4d} "
                  f"{'FLUSH' if e['is_flush'] else '':>5s}")
        if len(seg_log) > 25:
            print(f"  ... ({len(seg_log) - 25} more segments)")

        # Count flush segments
        n_flush = sum(1 for e in seg_log if e['is_flush'])
        print(f"\n  Flush segments: {n_flush}/{len(seg_log)} "
              f"({100*n_flush/len(seg_log):.1f}%)")

        # Our metrics
        our_wta = sum(e - s for s, e, l in our_tl if l == "A")
        our_wtb = sum(e - s for s, e, l in our_tl if l == "B")
        our_sil = sum(e - s for s, e, l in our_tl if l == "silence")

        # B establishment point
        b_est_seg = None
        for e in seg_log:
            if e['ref_b_exists'] and e['decision'] == 'B':
                b_est_seg = e
                break

        if b_est_seg:
            print(f"\n  B established at segment {b_est_seg['idx']}: "
                  f"{format_ms(b_est_seg['start_ms'])}-{format_ms(b_est_seg['end_ms'])} "
                  f"(sim_a={b_est_seg['sim_a']:.3f})")
        else:
            print(f"\n  B never established!")

        # SpeechBrain
        print(f"  Running SpeechBrain...")
        diarizer = SpeechBrainDiarizer()
        vad_fn2 = create_silero_vad()
        baseline = PipelineSimulator(vad_fn2, smoothing_window=1,
                                     speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                     b_confirm_frames=1)
        baseline.run(audio)
        vad_speech = extract_vad_segments(baseline.timeline)
        sb_tl = diarizer.diarize(audio, vad_speech)

        cmp = compare_timelines(
            [(s, e, l) for s, e, l in our_tl if l != "silence"],
            sb_tl, duration_ms)
        sb_wta = cmp.get("sb_wta_ms", 0)
        sb_wtb = cmp.get("sb_wtb_ms", 0)

        # Summary
        print(f"\n  {'=' * 60}")
        print(f"  RESULTS: Clip {clip_num}")
        print(f"  {'=' * 60}")
        print(f"  {'':>15s} {'WTA':>8s} {'WTB':>8s} {'Silence':>8s}")
        print(f"  {'Our pipeline':>15s} {format_ms(our_wta):>8s} {format_ms(our_wtb):>8s} "
              f"{format_ms(our_sil):>8s}")
        print(f"  {'SpeechBrain':>15s} {format_ms(sb_wta):>8s} {format_ms(sb_wtb):>8s}")
        print(f"\n  Agreement with SB: {cmp['agreement_pct']:.1f}%")

        # Ground truth comparison
        gt = GROUND_TRUTHS.get(clip_num, [])
        if gt:
            gt_a = sum(e - s for s, e, l in gt if l == "A")
            gt_b = sum(e - s for s, e, l in gt if l == "B")
            gt_end = max(e for _, e, _ in gt)
            print(f"\n  Ground truth (first {gt_end/1000:.0f}s): "
                  f"A={format_ms(gt_a)}, B={format_ms(gt_b)}")

            # Check initial portion classification
            gt_end_ms = gt_end
            our_a_initial = sum(
                min(e, gt_end_ms) - max(s, 0)
                for s, e, l in our_tl if l == "A" and s < gt_end_ms
            )
            our_b_initial = sum(
                min(e, gt_end_ms) - max(s, 0)
                for s, e, l in our_tl if l == "B" and s < gt_end_ms
            )
            print(f"  Our initial {gt_end/1000:.0f}s: "
                  f"A={format_ms(max(0, our_a_initial))}, B={format_ms(max(0, our_b_initial))}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
