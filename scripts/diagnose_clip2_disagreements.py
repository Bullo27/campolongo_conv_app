#!/usr/bin/env python3
"""Part 1: Clip 2 disagreement diagnostic — where/why do we disagree with SpeechBrain?

For each of our pipeline segments, records sim_a, sim_b, our decision, and SB's
majority label in that time range. Then prints aggregate analysis:
- Disagreement by confidence band (|sim_a - sim_b|)
- Direction (we-B/SB-A vs we-A/SB-B)
- 30s time-bucket agreement rates
- Contiguous disagreement clusters
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
CLIP_PATH = os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3")

THRESHOLD = 0.80
BUFFER = 47
BCONF = 2


def run_pipeline_with_segment_log(audio, embedder):
    """Run our neural pipeline (sm=1, margin=0) and return per-segment data."""
    vad_fn = create_silero_vad()
    speaker_id = SpeakerIdentifier(
        threshold=THRESHOLD, margin=0.0, b_confirm_frames=BCONF,
        min_frames_for_b=BUFFER,
    )

    n_frames = len(audio) // FRAME_SIZE
    speech_buffer = []
    speech_buffer_start_frame = 0
    timeline = []
    segments = []  # per-segment log

    def identify_and_log(current_frame):
        nonlocal speech_buffer
        n_buffered = len(speech_buffer)
        is_flush = n_buffered < BUFFER
        raw_audio = np.concatenate(speech_buffer)
        speech_buffer = []
        embedding = embedder.extract_embedding(raw_audio)

        had_b = speaker_id.ref_b is not None
        decision = speaker_id.identify(embedding, n_frames=n_buffered)

        start_ms = round(speech_buffer_start_frame * FRAME_MS)
        end_ms = round(current_frame * FRAME_MS)

        segments.append({
            "idx": len(segments),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "n_frames": n_buffered,
            "is_flush": is_flush,
            "decision": decision,
            "sim_a": speaker_id.last_sim_a,
            "sim_b": speaker_id.last_sim_b,
            "confidence": abs(speaker_id.last_sim_a - speaker_id.last_sim_b),
            "ref_b_exists": speaker_id.ref_b is not None,
            "b_just_established": (not had_b and speaker_id.ref_b is not None),
        })
        return decision

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

    return segments, timeline


def build_sb_frame_labels(sb_timeline, label_map, total_frames):
    """Build a frame-level array of mapped SB labels (A/B/None)."""
    labels = [None] * total_frames
    for start_ms, end_ms, raw_label in sb_timeline:
        mapped = label_map.get(raw_label, raw_label)
        start_f = int(start_ms / FRAME_MS)
        end_f = int(end_ms / FRAME_MS)
        for f in range(max(0, start_f), min(total_frames, end_f)):
            labels[f] = mapped
    return labels


def get_sb_majority_label(sb_frame_labels, start_ms, end_ms):
    """Get SB's majority label in a time range."""
    start_f = int(start_ms / FRAME_MS)
    end_f = int(end_ms / FRAME_MS)
    count_a = 0
    count_b = 0
    for f in range(max(0, start_f), min(len(sb_frame_labels), end_f)):
        if sb_frame_labels[f] == "A":
            count_a += 1
        elif sb_frame_labels[f] == "B":
            count_b += 1
    if count_a == 0 and count_b == 0:
        return None  # no SB speech in this range
    return "A" if count_a >= count_b else "B"


def main():
    print("=" * 80)
    print("  CLIP 2 DISAGREEMENT DIAGNOSTIC (Part 1)")
    print(f"  Params: t={THRESHOLD}, bconf={BCONF}, buffer={BUFFER}, margin=0, smoothing=1")
    print("=" * 80)

    if not os.path.exists(CLIP_PATH):
        print(f"  ERROR: Clip not found at {CLIP_PATH}")
        return

    audio = load_audio(CLIP_PATH)
    duration_s = len(audio) / SAMPLE_RATE
    duration_ms = duration_s * 1000
    total_frames = len(audio) // FRAME_SIZE
    print(f"  Duration: {format_ms(duration_ms)} ({duration_s:.1f}s)")

    # ── Run our pipeline ──
    print(f"\n  Running our pipeline...")
    embedder = WeSpeakerEmbedder()
    t0 = time.time()
    segments, our_tl = run_pipeline_with_segment_log(audio, embedder)
    elapsed_ours = time.time() - t0
    print(f"  Done in {elapsed_ours:.1f}s — {len(segments)} segments")

    # ── Run SpeechBrain ──
    print(f"  Running SpeechBrain...")
    diarizer = SpeechBrainDiarizer()
    vad_fn_sb = create_silero_vad()
    baseline = PipelineSimulator(vad_fn_sb, smoothing_window=1,
                                 speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                 b_confirm_frames=1)
    baseline.run(audio)
    vad_speech = extract_vad_segments(baseline.timeline)
    sb_tl = diarizer.diarize(audio, vad_speech)

    # Get label map via compare_timelines
    our_speech = [(s, e, l) for s, e, l in our_tl if l != "silence"]
    cmp = compare_timelines(our_speech, sb_tl, duration_ms)
    label_map = cmp["label_map"]
    print(f"  SB label map: {label_map}")
    print(f"  Overall SB agreement: {cmp['agreement_pct']:.1f}%")

    # Build frame-level SB labels
    sb_frame_labels = build_sb_frame_labels(sb_tl, label_map, total_frames)

    # ── Annotate each segment with SB's majority label ──
    for seg in segments:
        seg["sb_label"] = get_sb_majority_label(sb_frame_labels, seg["start_ms"], seg["end_ms"])
        if seg["sb_label"] is not None:
            seg["agrees"] = (seg["decision"] == seg["sb_label"])
        else:
            seg["agrees"] = None  # SB has no speech here

    # Filter to segments where both have an opinion
    compared = [s for s in segments if s["agrees"] is not None]
    matches = [s for s in compared if s["agrees"]]
    mismatches = [s for s in compared if not s["agrees"]]

    print(f"\n  Segments total: {len(segments)}")
    print(f"  Segments with SB coverage: {len(compared)}")
    print(f"  Matches: {len(matches)} ({100*len(matches)/len(compared):.1f}%)")
    print(f"  Mismatches: {len(mismatches)} ({100*len(mismatches)/len(compared):.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # 1. DISAGREEMENT BY CONFIDENCE BAND
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  1. DISAGREEMENTS BY CONFIDENCE BAND (|sim_a - sim_b|)")
    print(f"{'=' * 80}")

    bands = [
        ("< 0.02 (very low)", lambda c: c < 0.02),
        ("0.02–0.05 (low)", lambda c: 0.02 <= c < 0.05),
        ("0.05–0.10 (medium)", lambda c: 0.05 <= c < 0.10),
        (">= 0.10 (high)", lambda c: c >= 0.10),
    ]

    print(f"\n  {'Band':<25s} {'Match':>8s} {'Mismatch':>10s} {'Total':>8s} {'Mis %':>8s}")
    print(f"  {'─' * 65}")

    for band_name, band_fn in bands:
        band_match = [s for s in matches if band_fn(s["confidence"])]
        band_mismatch = [s for s in mismatches if band_fn(s["confidence"])]
        band_total = len(band_match) + len(band_mismatch)
        mis_pct = 100 * len(band_mismatch) / band_total if band_total > 0 else 0
        print(f"  {band_name:<25s} {len(band_match):>8d} {len(band_mismatch):>10d} "
              f"{band_total:>8d} {mis_pct:>7.1f}%")

    # Also show cumulative: what % of mismatches are below various thresholds
    print(f"\n  Cumulative: what fraction of mismatches have confidence below threshold?")
    for t in [0.02, 0.03, 0.05, 0.10, 0.15]:
        n_below = sum(1 for s in mismatches if s["confidence"] < t)
        pct = 100 * n_below / len(mismatches) if mismatches else 0
        print(f"    conf < {t:.2f}: {n_below}/{len(mismatches)} mismatches ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # 2. DIRECTION OF DISAGREEMENTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  2. DIRECTION OF DISAGREEMENTS")
    print(f"{'=' * 80}")

    we_b_sb_a = [s for s in mismatches if s["decision"] == "B" and s["sb_label"] == "A"]
    we_a_sb_b = [s for s in mismatches if s["decision"] == "A" and s["sb_label"] == "B"]

    print(f"\n  We say B, SB says A: {len(we_b_sb_a)} segments")
    print(f"  We say A, SB says B: {len(we_a_sb_b)} segments")

    if we_b_sb_a:
        confs = [s["confidence"] for s in we_b_sb_a]
        print(f"\n  We-B/SB-A confidence stats:")
        print(f"    mean={np.mean(confs):.4f}, median={np.median(confs):.4f}, "
              f"min={np.min(confs):.4f}, max={np.max(confs):.4f}")
        low_conf = sum(1 for c in confs if c < 0.05)
        print(f"    {low_conf}/{len(confs)} have confidence < 0.05")

    if we_a_sb_b:
        confs = [s["confidence"] for s in we_a_sb_b]
        print(f"\n  We-A/SB-B confidence stats:")
        print(f"    mean={np.mean(confs):.4f}, median={np.median(confs):.4f}, "
              f"min={np.min(confs):.4f}, max={np.max(confs):.4f}")
        low_conf = sum(1 for c in confs if c < 0.05)
        print(f"    {low_conf}/{len(confs)} have confidence < 0.05")

    # ══════════════════════════════════════════════════════════════════
    # 3. TIME-BUCKET ANALYSIS (30s buckets)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  3. AGREEMENT BY 30s TIME BUCKETS")
    print(f"{'=' * 80}")

    bucket_size_ms = 30000
    n_buckets = int(duration_ms / bucket_size_ms) + 1

    print(f"\n  {'Bucket':<15s} {'Match':>7s} {'Mis':>5s} {'Total':>7s} {'Agree%':>8s} "
          f"{'WeB/SBA':>8s} {'WeA/SBB':>8s} {'AvgConf':>8s}")
    print(f"  {'─' * 75}")

    for b in range(n_buckets):
        bstart = b * bucket_size_ms
        bend = (b + 1) * bucket_size_ms
        bucket_segs = [s for s in compared
                       if s["start_ms"] < bend and s["end_ms"] > bstart]
        if not bucket_segs:
            continue
        b_match = sum(1 for s in bucket_segs if s["agrees"])
        b_mis = len(bucket_segs) - b_match
        agree_pct = 100 * b_match / len(bucket_segs) if bucket_segs else 0
        b_we_b_sb_a = sum(1 for s in bucket_segs if not s["agrees"]
                          and s["decision"] == "B" and s["sb_label"] == "A")
        b_we_a_sb_b = sum(1 for s in bucket_segs if not s["agrees"]
                          and s["decision"] == "A" and s["sb_label"] == "B")
        avg_conf = np.mean([s["confidence"] for s in bucket_segs])
        label = f"{format_ms(bstart)}-{format_ms(bend)}"
        print(f"  {label:<15s} {b_match:>7d} {b_mis:>5d} {len(bucket_segs):>7d} "
              f"{agree_pct:>7.1f}% {b_we_b_sb_a:>8d} {b_we_a_sb_b:>8d} {avg_conf:>8.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 4. DISAGREEMENT CLUSTERS (contiguous runs of mismatches)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  4. DISAGREEMENT CLUSTERS (contiguous mismatch runs)")
    print(f"{'=' * 80}")

    # Build ordered list of agree/disagree for compared segments
    clusters = []
    current_cluster = None
    for seg in compared:
        if not seg["agrees"]:
            if current_cluster is None:
                current_cluster = {
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "segments": [seg],
                }
            else:
                current_cluster["end_ms"] = seg["end_ms"]
                current_cluster["segments"].append(seg)
        else:
            if current_cluster is not None:
                clusters.append(current_cluster)
                current_cluster = None
    if current_cluster is not None:
        clusters.append(current_cluster)

    print(f"\n  Total clusters: {len(clusters)}")
    size_1 = sum(1 for c in clusters if len(c["segments"]) == 1)
    size_2 = sum(1 for c in clusters if len(c["segments"]) == 2)
    size_3plus = sum(1 for c in clusters if len(c["segments"]) >= 3)
    print(f"  Size 1 (isolated): {size_1}")
    print(f"  Size 2: {size_2}")
    print(f"  Size 3+: {size_3plus}")

    # Show clusters of size >= 2
    if any(len(c["segments"]) >= 2 for c in clusters):
        print(f"\n  Clusters of size >= 2:")
        print(f"  {'Time range':<20s} {'Size':>5s} {'Direction':>12s} {'AvgConf':>8s}")
        print(f"  {'─' * 50}")
        for c in clusters:
            if len(c["segments"]) < 2:
                continue
            n_we_b_sb_a = sum(1 for s in c["segments"]
                              if s["decision"] == "B" and s["sb_label"] == "A")
            n_we_a_sb_b = len(c["segments"]) - n_we_b_sb_a
            if n_we_b_sb_a > n_we_a_sb_b:
                direction = "We-B/SB-A"
            elif n_we_a_sb_b > n_we_b_sb_a:
                direction = "We-A/SB-B"
            else:
                direction = "mixed"
            avg_conf = np.mean([s["confidence"] for s in c["segments"]])
            label = f"{format_ms(c['start_ms'])}-{format_ms(c['end_ms'])}"
            print(f"  {label:<20s} {len(c['segments']):>5d} {direction:>12s} {avg_conf:>8.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 4B. ISOLATED MISMATCH CONTEXT (N-1 / N+1 labels in our pipeline)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  4B. ISOLATED MISMATCH CONTEXT — OUR PIPELINE'S SURROUNDING LABELS")
    print(f"{'=' * 80}")

    # Build a lookup from segment idx to the segment's position in the compared list
    seg_by_idx = {s["idx"]: s for s in segments}

    isolated_clusters = [c for c in clusters if len(c["segments"]) == 1]
    transition_boundary = []  # N-1 and N+1 have different labels in our pipeline
    mid_run = []              # N-1 and N+1 have same label in our pipeline
    edge_cases = []           # first or last segment, can't determine context

    for ic in isolated_clusters:
        seg = ic["segments"][0]
        idx = seg["idx"]
        prev_seg = seg_by_idx.get(idx - 1)
        next_seg = seg_by_idx.get(idx + 1)

        prev_label = prev_seg["decision"] if prev_seg else None
        next_label = next_seg["decision"] if next_seg else None

        seg["prev_label"] = prev_label
        seg["next_label"] = next_label

        if prev_label is None or next_label is None:
            seg["context_type"] = "edge"
            edge_cases.append(seg)
        elif prev_label == next_label:
            seg["context_type"] = "mid-run"
            seg["surrounding_label"] = prev_label
            mid_run.append(seg)
        else:
            seg["context_type"] = "transition"
            transition_boundary.append(seg)

    print(f"\n  Isolated mismatches: {len(isolated_clusters)}")
    print(f"  Transition boundary (N-1 != N+1): {len(transition_boundary)}")
    print(f"  Mid-run (N-1 == N+1):             {len(mid_run)}")
    print(f"  Edge (first/last segment):         {len(edge_cases)}")

    # ── Transition boundary details ──
    if transition_boundary:
        print(f"\n  TRANSITION BOUNDARY mismatches ({len(transition_boundary)}):")
        print(f"  {'Seg':>4s} {'Start':>7s} {'End':>7s} {'Frm':>4s} {'Ours':>5s} {'SB':>4s} "
              f"{'Conf':>6s} {'Prev':>5s} {'Next':>5s} {'Flush':>5s}")
        print(f"  {'─' * 60}")
        for seg in transition_boundary:
            print(f"  {seg['idx']:>4d} {format_ms(seg['start_ms']):>7s} "
                  f"{format_ms(seg['end_ms']):>7s} {seg['n_frames']:>4d} "
                  f"  {seg['decision']}    {seg['sb_label']} "
                  f"{seg['confidence']:>6.3f} "
                  f"  {seg['prev_label']}     {seg['next_label']} "
                  f"{'FLUSH' if seg['is_flush'] else '':>5s}")

    # ── Mid-run details ──
    if mid_run:
        print(f"\n  MID-RUN mismatches ({len(mid_run)}):")
        print(f"  {'Seg':>4s} {'Start':>7s} {'End':>7s} {'Frm':>4s} {'Ours':>5s} {'SB':>4s} "
              f"{'Conf':>6s} {'Surround':>9s} {'Dur(ms)':>8s} {'Flush':>5s}")
        print(f"  {'─' * 70}")
        for seg in mid_run:
            dur_ms = seg['end_ms'] - seg['start_ms']
            print(f"  {seg['idx']:>4d} {format_ms(seg['start_ms']):>7s} "
                  f"{format_ms(seg['end_ms']):>7s} {seg['n_frames']:>4d} "
                  f"  {seg['decision']}    {seg['sb_label']} "
                  f"{seg['confidence']:>6.3f} "
                  f"    {seg['surrounding_label']}+{seg['surrounding_label']} "
                  f"{dur_ms:>8.0f} "
                  f"{'FLUSH' if seg['is_flush'] else '':>5s}")

        # Subcategorize mid-run by direction
        mid_we_b_sb_a = [s for s in mid_run if s["decision"] == "B" and s["sb_label"] == "A"]
        mid_we_a_sb_b = [s for s in mid_run if s["decision"] == "A" and s["sb_label"] == "B"]
        print(f"\n  Mid-run direction: {len(mid_we_b_sb_a)} we-B/SB-A, {len(mid_we_a_sb_b)} we-A/SB-B")

        # How many mid-run are flush (short) segments?
        mid_flush = [s for s in mid_run if s["is_flush"]]
        mid_full = [s for s in mid_run if not s["is_flush"]]
        print(f"  Mid-run flush (short): {len(mid_flush)}, full-length: {len(mid_full)}")

        if mid_run:
            durs = [s['end_ms'] - s['start_ms'] for s in mid_run]
            print(f"  Mid-run durations: mean={np.mean(durs):.0f}ms, "
                  f"median={np.median(durs):.0f}ms, min={np.min(durs):.0f}ms, max={np.max(durs):.0f}ms")

    # ══════════════════════════════════════════════════════════════════
    # 5. DETAILED MISMATCH LIST (first 40)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  5. DETAILED MISMATCH LIST (first 40)")
    print(f"{'=' * 80}")

    print(f"\n  {'Seg':>4s} {'Start':>7s} {'End':>7s} {'Frm':>4s} {'Ours':>5s} {'SB':>4s} "
          f"{'SimA':>6s} {'SimB':>6s} {'Conf':>6s} {'Flush':>5s}")
    print(f"  {'─' * 65}")
    for seg in mismatches[:40]:
        print(f"  {seg['idx']:>4d} {format_ms(seg['start_ms']):>7s} "
              f"{format_ms(seg['end_ms']):>7s} {seg['n_frames']:>4d} "
              f"  {seg['decision']}    {seg['sb_label']} "
              f"{seg['sim_a']:>6.3f} {seg['sim_b']:>6.3f} "
              f"{seg['confidence']:>6.3f} "
              f"{'FLUSH' if seg['is_flush'] else '':>5s}")
    if len(mismatches) > 40:
        print(f"  ... ({len(mismatches) - 40} more)")

    # ══════════════════════════════════════════════════════════════════
    # 6. SUMMARY STATS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  6. SUMMARY")
    print(f"{'=' * 80}")

    # How much speech time do mismatches represent?
    mismatch_speech_ms = sum(s["end_ms"] - s["start_ms"] for s in mismatches)
    total_speech_ms = sum(s["end_ms"] - s["start_ms"] for s in compared)
    print(f"\n  Mismatch speech time: {format_ms(mismatch_speech_ms)} "
          f"({100*mismatch_speech_ms/total_speech_ms:.1f}% of compared speech)")

    # How much of the A-share gap do we-B/SB-A mismatches explain?
    we_b_sb_a_ms = sum(s["end_ms"] - s["start_ms"] for s in we_b_sb_a)
    we_a_sb_b_ms = sum(s["end_ms"] - s["start_ms"] for s in we_a_sb_b)
    net_gap_ms = we_b_sb_a_ms - we_a_sb_b_ms
    print(f"  We-B/SB-A speech time: {format_ms(we_b_sb_a_ms)}")
    print(f"  We-A/SB-B speech time: {format_ms(we_a_sb_b_ms)}")
    print(f"  Net gap (would shift A share if fixed): {format_ms(net_gap_ms)}")
    print(f"  As % of total speech: {100*net_gap_ms/total_speech_ms:.1f}%")

    # Flush vs full segment mismatches
    flush_mis = [s for s in mismatches if s["is_flush"]]
    full_mis = [s for s in mismatches if not s["is_flush"]]
    print(f"\n  Flush segment mismatches: {len(flush_mis)}")
    print(f"  Full segment mismatches: {len(full_mis)}")

    # Confidence distribution of all mismatches
    if mismatches:
        confs = [s["confidence"] for s in mismatches]
        print(f"\n  All mismatches confidence: mean={np.mean(confs):.4f}, "
              f"median={np.median(confs):.4f}, std={np.std(confs):.4f}")
        print(f"    P25={np.percentile(confs, 25):.4f}, P75={np.percentile(confs, 75):.4f}")

    # Compare with matches
    if matches:
        confs_m = [s["confidence"] for s in matches]
        print(f"  All matches confidence:    mean={np.mean(confs_m):.4f}, "
              f"median={np.median(confs_m):.4f}, std={np.std(confs_m):.4f}")

    print(f"\n{'=' * 80}")
    print("  KEY QUESTION: Are mismatches predominantly low-confidence?")
    if mismatches:
        low_conf_pct = 100 * sum(1 for s in mismatches if s["confidence"] < 0.05) / len(mismatches)
        print(f"  → {low_conf_pct:.1f}% of mismatches have confidence < 0.05")
        if low_conf_pct > 60:
            print("  → YES — confidence-gating or hysteresis likely to help")
        elif low_conf_pct > 40:
            print("  → MIXED — some benefit possible, but many high-confidence mismatches too")
        else:
            print("  → NO — most mismatches are high-confidence (embedding/reference issue)")
    print(f"{'=' * 80}")

    print("\nDone.")


if __name__ == "__main__":
    main()
