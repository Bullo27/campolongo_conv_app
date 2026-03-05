#!/usr/bin/env python3
"""Smoothing window sweep: compare smoothing=1,2,3,4,5 vs SpeechBrain for clips 1 & 2.

Reports per smoothing config:
- Speaking time per speaker (A, B)
- Silence split by last-active speaker
- Speaker changes count
- A/B share percentages
- SpeechBrain agreement
- B establishment point
- Ground truth comparison (initial portion)
"""
import os
import sys
import time
from collections import deque

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
SMOOTHING_VALUES = [1, 2, 3, 4, 5]


def split_silence(timeline):
    """Split silence segments by last-active speaker."""
    speech_a = 0
    speech_b = 0
    silence_a = 0
    silence_b = 0
    silence_initial = 0
    last_speaker = None
    speaker_changes = 0
    prev_speaker = None

    for start, end, label in timeline:
        dur = end - start
        if label == "A":
            speech_a += dur
            if prev_speaker == "B":
                speaker_changes += 1
            prev_speaker = "A"
            last_speaker = "A"
        elif label == "B":
            speech_b += dur
            if prev_speaker == "A":
                speaker_changes += 1
            prev_speaker = "B"
            last_speaker = "B"
        elif label == "silence":
            if last_speaker is None:
                silence_initial += dur
            elif last_speaker == "A":
                silence_a += dur
            elif last_speaker == "B":
                silence_b += dur

    total_silence = silence_initial + silence_a + silence_b
    return {
        "speech_a": speech_a,
        "speech_b": speech_b,
        "silence_a": silence_a,
        "silence_b": silence_b,
        "silence_initial": silence_initial,
        "total_silence": total_silence,
        "speaker_changes": speaker_changes,
    }


def run_our_pipeline(audio, embedder, smoothing=1):
    """Run our neural pipeline with Phase 2C params + majority-vote smoothing."""
    vad_fn = create_silero_vad()
    speaker_id = SpeakerIdentifier(
        threshold=THRESHOLD, margin=0.0, b_confirm_frames=BCONF,
        min_frames_for_b=BUFFER,
    )

    n_frames = len(audio) // FRAME_SIZE
    speech_buffer = []
    speech_buffer_start_frame = 0
    timeline = []
    b_est_time = None
    segment_count = 0

    # Smoothing state
    decision_buffer = deque(maxlen=smoothing)
    last_smoothed = "A"

    def identify_and_clear(current_frame):
        nonlocal speech_buffer, b_est_time, segment_count, last_smoothed
        n_buffered = len(speech_buffer)
        raw_audio = np.concatenate(speech_buffer)
        speech_buffer = []
        embedding = embedder.extract_embedding(raw_audio)

        had_b = speaker_id.ref_b is not None
        raw_speaker = speaker_id.identify(embedding, n_frames=n_buffered)
        segment_count += 1

        if not had_b and speaker_id.ref_b is not None:
            mid_frame = (speech_buffer_start_frame + current_frame) // 2
            b_est_time = round(mid_frame * FRAME_MS)

        # Majority-vote smoothing
        decision_buffer.append(raw_speaker)
        if smoothing <= 1:
            speaker = raw_speaker
        else:
            count_a = sum(1 for d in decision_buffer if d == "A")
            count_b = len(decision_buffer) - count_a
            if count_a > count_b:
                speaker = "A"
            elif count_b > count_a:
                speaker = "B"
            else:
                speaker = last_smoothed  # tie → keep current
        last_smoothed = speaker
        return speaker

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
            if len(speech_buffer) >= BUFFER:
                buf_start = speech_buffer_start_frame
                speaker = identify_and_clear(i)
                update_timeline(i, speaker, start_frame=buf_start)
        else:
            if len(speech_buffer) >= MIN_FLUSH_FRAMES:
                buf_start = speech_buffer_start_frame
                speaker = identify_and_clear(i)
                update_timeline(i, speaker, start_frame=buf_start)
            speech_buffer.clear()
            update_timeline(i, "silence")

    if len(speech_buffer) >= MIN_FLUSH_FRAMES:
        buf_start = speech_buffer_start_frame
        speaker = identify_and_clear(n_frames - 1)
        update_timeline(n_frames - 1, speaker, start_frame=buf_start)

    return timeline, b_est_time, segment_count


def main():
    cols = [f"sm={s}" for s in SMOOTHING_VALUES] + ["SpeechBrain"]
    col_w = 12
    label_w = 26

    print("=" * 95)
    print("  SMOOTHING WINDOW SWEEP — Phase 2C Neural Pipeline")
    print(f"  Params: t={THRESHOLD}, bconf={BCONF}, buffer={BUFFER}, margin=0")
    print(f"  Smoothing values: {SMOOTHING_VALUES}")
    print("=" * 95)

    embedder = WeSpeakerEmbedder()

    for clip_num in [1, 2]:
        clip_path = CLIPS.get(clip_num)
        if not clip_path or not os.path.exists(clip_path):
            print(f"\nClip {clip_num}: not found, skipping")
            continue

        print(f"\n{'=' * 95}")
        print(f"  CLIP {clip_num}")
        print(f"{'=' * 95}")

        audio = load_audio(clip_path)
        duration_s = len(audio) / SAMPLE_RATE
        duration_ms = duration_s * 1000
        print(f"  Total duration: {format_ms(duration_ms)} ({duration_s:.1f}s)")

        # ── SpeechBrain (run once) ──
        print(f"\n  Running SpeechBrain...")
        diarizer = SpeechBrainDiarizer()
        vad_fn_sb = create_silero_vad()
        baseline = PipelineSimulator(vad_fn_sb, smoothing_window=1,
                                     speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                     b_confirm_frames=1)
        baseline.run(audio)
        vad_speech = extract_vad_segments(baseline.timeline)
        sb_tl = diarizer.diarize(audio, vad_speech)

        # ── Run our pipeline for each smoothing value ──
        results = {}  # smoothing_val -> {timeline, b_est, seg_count, metrics, cmp, elapsed}
        for s in SMOOTHING_VALUES:
            print(f"  Running our pipeline (smoothing={s})...")
            t0 = time.time()
            tl, b_est, seg_cnt = run_our_pipeline(audio, embedder, smoothing=s)
            elapsed = time.time() - t0

            metrics = split_silence(tl)

            # Compare with SB (use unsmoothed speech segments for label mapping from sm=1)
            cmp = compare_timelines(
                [(st, en, lb) for st, en, lb in tl if lb != "silence"],
                sb_tl, duration_ms)
            lmap = cmp.get("label_map", {})

            results[s] = {
                "timeline": tl, "b_est": b_est, "seg_count": seg_cnt,
                "metrics": metrics, "cmp": cmp, "lmap": lmap, "elapsed": elapsed,
            }

        # ── Build SB metrics (use label map from sm=1 for consistency) ──
        lmap_ref = results[1]["lmap"]
        sb_tl_mapped = [(st, en, lmap_ref.get(lb, lb)) for st, en, lb in sb_tl]
        sb_tl_full = []
        prev_end = 0
        for st, en, lb in sorted(sb_tl_mapped, key=lambda x: x[0]):
            if st > prev_end:
                sb_tl_full.append((prev_end, st, "silence"))
            sb_tl_full.append((st, en, lb))
            prev_end = en
        if prev_end < duration_ms:
            sb_tl_full.append((prev_end, duration_ms, "silence"))
        sb_metrics = split_silence(sb_tl_full)

        # ── Print comparison table ──
        header = f"  {'METRIC':<{label_w}s}" + "".join(f"{c:>{col_w}s}" for c in cols)
        sep = "  " + "─" * (label_w + col_w * len(cols))

        print(f"\n{sep}")
        print(header)
        print(sep)

        def row_time(label, vals):
            line = f"  {label:<{label_w}s}"
            for v in vals:
                line += f"{format_ms(v):>{col_w}s}"
            print(line)

        def row_str(label, vals):
            line = f"  {label:<{label_w}s}"
            for v in vals:
                line += f"{str(v):>{col_w}s}"
            print(line)

        def row_pct(label, vals):
            line = f"  {label:<{label_w}s}"
            for v in vals:
                line += f"{v:>{col_w - 1}.1f}%"
            print(line)

        # Speech times
        row_time("Speech A", [results[s]["metrics"]["speech_a"] for s in SMOOTHING_VALUES] + [sb_metrics["speech_a"]])
        row_time("Speech B", [results[s]["metrics"]["speech_b"] for s in SMOOTHING_VALUES] + [sb_metrics["speech_b"]])
        print(sep)

        # Silence
        row_time("Silence (A's turn)", [results[s]["metrics"]["silence_a"] for s in SMOOTHING_VALUES] + [sb_metrics["silence_a"]])
        row_time("Silence (B's turn)", [results[s]["metrics"]["silence_b"] for s in SMOOTHING_VALUES] + [sb_metrics["silence_b"]])
        row_time("Silence (initial)", [results[s]["metrics"]["silence_initial"] for s in SMOOTHING_VALUES] + [sb_metrics["silence_initial"]])
        row_time("Total silence", [results[s]["metrics"]["total_silence"] for s in SMOOTHING_VALUES] + [sb_metrics["total_silence"]])
        print(sep)

        # Speaker changes
        row_str("Speaker changes (A↔B)", [results[s]["metrics"]["speaker_changes"] for s in SMOOTHING_VALUES] + [sb_metrics["speaker_changes"]])
        row_str("Total segments", [results[s]["seg_count"] for s in SMOOTHING_VALUES] + [len(sb_tl)])
        print(sep)

        # Percentages
        pct_vals = []
        for s in SMOOTHING_VALUES:
            m = results[s]["metrics"]
            total = m["speech_a"] + m["speech_b"]
            pct_vals.append(100 * m["speech_a"] / total if total > 0 else 0)
        sb_total = sb_metrics["speech_a"] + sb_metrics["speech_b"]
        pct_vals.append(100 * sb_metrics["speech_a"] / sb_total if sb_total > 0 else 0)
        row_pct("A share of speech", pct_vals)

        pct_vals_b = []
        for s in SMOOTHING_VALUES:
            m = results[s]["metrics"]
            total = m["speech_a"] + m["speech_b"]
            pct_vals_b.append(100 * m["speech_b"] / total if total > 0 else 0)
        pct_vals_b.append(100 * sb_metrics["speech_b"] / sb_total if sb_total > 0 else 0)
        row_pct("B share of speech", pct_vals_b)
        print(sep)

        # SB agreement
        agree_vals = [f"{results[s]['cmp']['agreement_pct']:.1f}%" for s in SMOOTHING_VALUES] + ["—"]
        row_str("SB agreement", agree_vals)

        # B establishment
        b_vals = [format_ms(results[s]["b_est"]) if results[s]["b_est"] is not None else "never" for s in SMOOTHING_VALUES] + ["—"]
        row_str("B established at", b_vals)

        # Processing time
        time_vals = [f"{results[s]['elapsed']:.1f}s" for s in SMOOTHING_VALUES] + ["—"]
        row_str("Processing time", time_vals)
        print(sep)

        # ── Ground truth comparison ──
        gt = GROUND_TRUTHS.get(clip_num, [])
        if gt:
            gt_end = max(e for _, e, _ in gt)
            gt_a = sum(e - s for s, e, l in gt if l == "A")
            gt_b = sum(e - s for s, e, l in gt if l == "B")

            print(f"\n  Ground truth (first {gt_end / 1000:.0f}s):")
            print(sep)
            print(header)
            print(sep)

            # GT row (same across all columns)
            row_time("GT: A", [gt_a] * (len(SMOOTHING_VALUES) + 1))
            row_time("GT: B", [gt_b] * (len(SMOOTHING_VALUES) + 1))

            # Our initial portion per smoothing
            our_a_inits = []
            our_b_inits = []
            for s in SMOOTHING_VALUES:
                tl = results[s]["timeline"]
                a_init = sum(min(e, gt_end) - max(st, 0) for st, e, lb in tl if lb == "A" and st < gt_end)
                b_init = sum(min(e, gt_end) - max(st, 0) for st, e, lb in tl if lb == "B" and st < gt_end)
                our_a_inits.append(max(0, a_init))
                our_b_inits.append(max(0, b_init))

            sb_a_init = sum(min(e, gt_end) - max(st, 0) for st, e, lb in sb_tl_full if lb == "A" and st < gt_end)
            sb_b_init = sum(min(e, gt_end) - max(st, 0) for st, e, lb in sb_tl_full if lb == "B" and st < gt_end)

            row_time("Detected: A", our_a_inits + [max(0, sb_a_init)])
            row_time("Detected: B", our_b_inits + [max(0, sb_b_init)])
            print(sep)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
