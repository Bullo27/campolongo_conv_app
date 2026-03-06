#!/usr/bin/env python3
"""Strategy comparison: baseline vs conf-gate vs hysteresis vs combined + dual-assignment.

Runs pipeline once per clip, captures raw segment decisions + similarities,
then replays with each strategy to produce modified timelines.
Reports full metrics for each strategy + SpeechBrain on both clips.

Dual-assignment: when both sim_a >= dual_t AND sim_b >= dual_t, credit
segment to both speakers. Primary speaker unchanged for state/silence/changes.
Includes relaxed SB agreement (match if SB label in any of our assigned labels).
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

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

THRESHOLD = 0.80
BUFFER = 47
BCONF = 2

STRATEGIES = [
    ("Baseline", {"type": "baseline"}),
    ("CG t=0.03", {"type": "conf-gate", "threshold": 0.03}),
    ("CG t=0.05", {"type": "conf-gate", "threshold": 0.05}),
    ("Hyst N=2", {"type": "hysteresis", "n": 2}),
    ("Comb .03/2", {"type": "combined", "threshold": 0.03, "n": 2}),
    ("Comb .05/2", {"type": "combined", "threshold": 0.05, "n": 2}),
]


def run_pipeline_segments(audio, embedder):
    """Run pipeline once, return list of segments with raw decisions + similarities."""
    vad_fn = create_silero_vad()
    speaker_id = SpeakerIdentifier(
        threshold=THRESHOLD, margin=0.0, b_confirm_frames=BCONF,
        min_frames_for_b=BUFFER,
    )

    n_frames = len(audio) // FRAME_SIZE
    speech_buffer = []
    speech_buffer_start_frame = 0
    segments = []

    def identify_and_log(current_frame):
        nonlocal speech_buffer
        n_buffered = len(speech_buffer)
        raw_audio = np.concatenate(speech_buffer)
        speech_buffer = []
        embedding = embedder.extract_embedding(raw_audio)

        had_b = speaker_id.ref_b is not None
        decision = speaker_id.identify(embedding, n_frames=n_buffered)

        segments.append({
            "start_ms": round(speech_buffer_start_frame * FRAME_MS),
            "end_ms": round((current_frame + 1) * FRAME_MS),
            "decision": decision,
            "sim_a": speaker_id.last_sim_a,
            "sim_b": speaker_id.last_sim_b,
            "confidence": abs(speaker_id.last_sim_a - speaker_id.last_sim_b),
            "n_frames": n_buffered,
        })
        return decision

    for i in range(n_frames):
        start = i * FRAME_SIZE
        frame = audio[start:start + FRAME_SIZE]
        is_speech = vad_fn(frame)

        if is_speech:
            if not speech_buffer:
                speech_buffer_start_frame = i
            speech_buffer.append(frame)
            if len(speech_buffer) >= BUFFER:
                identify_and_log(i)
        else:
            if len(speech_buffer) >= MIN_FLUSH_FRAMES:
                identify_and_log(i)
            speech_buffer.clear()

    if len(speech_buffer) >= MIN_FLUSH_FRAMES:
        identify_and_log(n_frames - 1)

    return segments


def apply_strategy(segments, params):
    """Apply post-hoc strategy to raw segment decisions. Return modified list."""
    decisions = []
    current = "A"
    consec_other = 0
    stype = params["type"]

    for seg in segments:
        raw = seg["decision"]
        conf = seg["confidence"]

        if stype == "baseline":
            decisions.append(raw)
            current = raw
            consec_other = 0

        elif stype == "conf-gate":
            t = params["threshold"]
            if conf >= t:
                decisions.append(raw)
                current = raw
                consec_other = 0
            else:
                decisions.append(current)

        elif stype == "hysteresis":
            n = params["n"]
            if raw != current:
                consec_other += 1
                if consec_other >= n:
                    current = raw
                    consec_other = 0
                decisions.append(current)
            else:
                consec_other = 0
                decisions.append(raw)

        elif stype == "combined":
            t = params["threshold"]
            n = params["n"]
            if conf >= t:
                # High confidence: trust raw immediately
                if raw != current:
                    consec_other = 0
                current = raw
                decisions.append(raw)
            else:
                # Low confidence: require N consecutive to switch
                if raw != current:
                    consec_other += 1
                    if consec_other >= n:
                        current = raw
                        consec_other = 0
                    decisions.append(current)
                else:
                    consec_other = 0
                    decisions.append(raw)

    return decisions


def build_timeline(segments, decisions, duration_ms):
    """Build timeline from segments + modified decisions, filling silence gaps."""
    entries = []
    prev_end = 0
    for seg, dec in zip(segments, decisions):
        s, e = seg["start_ms"], seg["end_ms"]
        if s > prev_end:
            # Silence gap
            if entries and entries[-1][2] == "silence":
                entries[-1] = (entries[-1][0], s, "silence")
            else:
                entries.append((prev_end, s, "silence"))
        # Speech segment
        if entries and entries[-1][2] == dec:
            entries[-1] = (entries[-1][0], e, dec)
        else:
            entries.append((s, e, dec))
        prev_end = e
    # Trailing silence
    end = round(duration_ms)
    if prev_end < end:
        if entries and entries[-1][2] == "silence":
            entries[-1] = (entries[-1][0], end, "silence")
        else:
            entries.append((prev_end, end, "silence"))
    return entries


def compute_metrics(timeline):
    """Compute speech/silence/changes metrics from a timeline."""
    speech_a = speech_b = 0
    silence_a = silence_b = silence_init = 0
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
                silence_init += dur
            elif last_speaker == "A":
                silence_a += dur
            else:
                silence_b += dur

    total = speech_a + speech_b
    return {
        "speech_a": speech_a,
        "speech_b": speech_b,
        "a_share": 100 * speech_a / total if total > 0 else 0,
        "b_share": 100 * speech_b / total if total > 0 else 0,
        "silence_a": silence_a,
        "silence_b": silence_b,
        "silence_init": silence_init,
        "total_silence": silence_a + silence_b + silence_init,
        "speaker_changes": speaker_changes,
    }


def frame_level_agreement(our_tl, sb_tl, label_map, duration_ms):
    """Compute frame-level agreement using a fixed label map."""
    n_frames = int(duration_ms / FRAME_MS) + 1
    our_labels = [None] * n_frames
    sb_labels = [None] * n_frames

    for start, end, label in our_tl:
        if label == "silence":
            continue
        for f in range(max(0, int(start / FRAME_MS)), min(n_frames, int(end / FRAME_MS))):
            our_labels[f] = label

    for start, end, label in sb_tl:
        mapped = label_map.get(label, label)
        for f in range(max(0, int(start / FRAME_MS)), min(n_frames, int(end / FRAME_MS))):
            sb_labels[f] = mapped

    total = agreed = 0
    for f in range(n_frames):
        if our_labels[f] is not None and sb_labels[f] is not None:
            total += 1
            if our_labels[f] == sb_labels[f]:
                agreed += 1

    return round(100 * agreed / total, 1) if total > 0 else 0.0


# ── Dual-assignment configs ──
DUAL_THRESHOLDS = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85]
DUAL_CAPS = [3]


def compute_dual_metrics(segments, baseline_decisions, dual_t, cap):
    """Compute dual-assignment metrics on top of baseline decisions.

    Returns dict with speech_a, speech_b, overlap_time, dual_count,
    a_share, b_share, speaker_changes, and per-segment label sets.
    """
    speech_a = 0
    speech_b = 0
    overlap_time = 0
    dual_count = 0
    consec_dual = 0
    speaker_changes = 0
    prev_speaker = None

    # Per-segment: list of sets of assigned labels
    segment_labels = []

    for seg, primary in zip(segments, baseline_decisions):
        dur = seg["end_ms"] - seg["start_ms"]
        sim_a = seg["sim_a"]
        sim_b = seg["sim_b"]
        other = "B" if primary == "A" else "A"

        both_above = (sim_a >= dual_t and sim_b >= dual_t)
        is_dual = both_above and consec_dual < cap

        # Primary always gets credit
        if primary == "A":
            speech_a += dur
        else:
            speech_b += dur

        if is_dual:
            # Other speaker also gets credit
            if other == "A":
                speech_a += dur
            else:
                speech_b += dur
            overlap_time += dur
            dual_count += 1
            consec_dual += 1
            segment_labels.append({primary, other})
        else:
            if not both_above:
                consec_dual = 0  # reset only when condition not met
            segment_labels.append({primary})

        # Speaker changes based on primary only
        if prev_speaker is not None and primary != prev_speaker:
            speaker_changes += 1
        prev_speaker = primary

    total = speech_a + speech_b
    return {
        "speech_a": speech_a,
        "speech_b": speech_b,
        "overlap_time": overlap_time,
        "dual_count": dual_count,
        "a_share": 100 * speech_a / total if total > 0 else 0,
        "b_share": 100 * speech_b / total if total > 0 else 0,
        "speaker_changes": speaker_changes,
        "segment_labels": segment_labels,
    }


def relaxed_agreement(segments, segment_labels, sb_tl, label_map, duration_ms):
    """Compute both strict and relaxed frame-level SB agreement.

    Strict: frame matches if primary label == SB mapped label.
    Relaxed: frame matches if SB mapped label is in ANY of our assigned labels.
    """
    n_frames = int(duration_ms / FRAME_MS) + 1

    # Build our frame-level accepted label sets
    our_accepted = [None] * n_frames  # None = silence/no speech
    for seg, labels in zip(segments, segment_labels):
        f_start = max(0, int(seg["start_ms"] / FRAME_MS))
        f_end = min(n_frames, int(seg["end_ms"] / FRAME_MS))
        for f in range(f_start, f_end):
            our_accepted[f] = labels

    # Build SB frame labels (mapped)
    sb_labels = [None] * n_frames
    for start, end, label in sb_tl:
        mapped = label_map.get(label, label)
        for f in range(max(0, int(start / FRAME_MS)), min(n_frames, int(end / FRAME_MS))):
            sb_labels[f] = mapped

    total = strict_agreed = relaxed_agreed = 0
    for f in range(n_frames):
        if our_accepted[f] is not None and sb_labels[f] is not None:
            total += 1
            sb_lbl = sb_labels[f]
            if sb_lbl in our_accepted[f]:
                relaxed_agreed += 1
            # Strict: check primary only (first element added = primary,
            # but sets are unordered; need to check differently)
            # Actually for strict, we use the baseline timeline agreement
            # which is already computed. But let's compute it here too
            # by checking if there's only one label or if sb matches.

    # For strict, we need the primary labels. Let's build that separately.
    our_primary = [None] * n_frames
    for seg, labels in zip(segments, segment_labels):
        # Primary is the baseline decision
        primary = seg["decision"]
        f_start = max(0, int(seg["start_ms"] / FRAME_MS))
        f_end = min(n_frames, int(seg["end_ms"] / FRAME_MS))
        for f in range(f_start, f_end):
            our_primary[f] = primary

    strict_total = strict_agreed = 0
    for f in range(n_frames):
        if our_primary[f] is not None and sb_labels[f] is not None:
            strict_total += 1
            if our_primary[f] == sb_labels[f]:
                strict_agreed += 1

    strict_pct = round(100 * strict_agreed / strict_total, 1) if strict_total > 0 else 0.0
    relaxed_pct = round(100 * relaxed_agreed / total, 1) if total > 0 else 0.0
    return strict_pct, relaxed_pct


def run_dual_sweep(segments, baseline_decisions, sb_tl, label_map, duration_ms):
    """Run all dual-assignment configs, return list of results."""
    results = []
    for dual_t in DUAL_THRESHOLDS:
        for cap in DUAL_CAPS:
            cap_label = "inf" if cap >= 999 else str(cap)
            name = f"t={dual_t} c={cap_label}"

            dm = compute_dual_metrics(segments, baseline_decisions, dual_t, cap)
            strict, relaxed = relaxed_agreement(
                segments, dm["segment_labels"], sb_tl, label_map, duration_ms)

            results.append({
                "name": name,
                "dual_t": dual_t,
                "cap": cap,
                "speech_a": dm["speech_a"],
                "speech_b": dm["speech_b"],
                "overlap_time": dm["overlap_time"],
                "dual_count": dm["dual_count"],
                "a_share": dm["a_share"],
                "b_share": dm["b_share"],
                "speaker_changes": dm["speaker_changes"],
                "strict_agree": strict,
                "relaxed_agree": relaxed,
            })
    return results


def main():
    print("=" * 120)
    print("  STRATEGY COMPARISON — Full Metrics Report")
    print(f"  Pipeline: t={THRESHOLD}, bconf={BCONF}, buffer={BUFFER}, margin=0")
    print("=" * 120)

    embedder = WeSpeakerEmbedder()

    for clip_num in [1, 2]:
        clip_path = CLIPS.get(clip_num)
        if not clip_path or not os.path.exists(clip_path):
            print(f"\nClip {clip_num}: not found, skipping")
            continue

        print(f"\n{'=' * 120}")
        print(f"  CLIP {clip_num}")
        print(f"{'=' * 120}")

        audio = load_audio(clip_path)
        duration_s = len(audio) / SAMPLE_RATE
        duration_ms = duration_s * 1000
        print(f"  Duration: {format_ms(duration_ms)}")

        # ── Run our pipeline once ──
        print(f"  Running pipeline...")
        t0 = time.time()
        segments = run_pipeline_segments(audio, embedder)
        print(f"  Done: {len(segments)} segments in {time.time() - t0:.1f}s")

        # ── Run SpeechBrain once ──
        print(f"  Running SpeechBrain...")
        diarizer = SpeechBrainDiarizer()
        vad_fn_sb = create_silero_vad()
        baseline_sim = PipelineSimulator(vad_fn_sb, smoothing_window=1,
                                         speech_segment_frames=SPEECH_SEGMENT_FRAMES,
                                         b_confirm_frames=1)
        baseline_sim.run(audio)
        vad_speech = extract_vad_segments(baseline_sim.timeline)
        sb_tl = diarizer.diarize(audio, vad_speech)

        # ── Get label map from baseline ──
        baseline_dec = apply_strategy(segments, {"type": "baseline"})
        baseline_tl = build_timeline(segments, baseline_dec, duration_ms)
        baseline_speech = [(s, e, l) for s, e, l in baseline_tl if l != "silence"]
        cmp = compare_timelines(baseline_speech, sb_tl, duration_ms)
        label_map = cmp["label_map"]
        print(f"  Label map: {label_map}")

        # ── Build SB full timeline for metrics ──
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
        sb_metrics = compute_metrics(sb_full)

        # ── Compute each strategy ──
        results = []  # (name, metrics, sb_agreement)
        for name, params in STRATEGIES:
            decisions = apply_strategy(segments, params)
            tl = build_timeline(segments, decisions, duration_ms)
            metrics = compute_metrics(tl)
            agreement = frame_level_agreement(tl, sb_tl, label_map, duration_ms)
            results.append((name, metrics, agreement))

        results.append(("SpeechBrain", sb_metrics, None))

        # ── Print table ──
        cols = [r[0] for r in results]
        cw = 12  # column width
        lw = 22  # label width

        header = f"  {'METRIC':<{lw}s}" + "".join(f"{c:>{cw}s}" for c in cols)
        sep = "  " + "─" * (lw + cw * len(cols))

        print(f"\n{sep}")
        print(header)
        print(sep)

        def row_time(label, vals):
            print(f"  {label:<{lw}s}" + "".join(f"{format_ms(v):>{cw}s}" for v in vals))

        def row_str(label, vals):
            print(f"  {label:<{lw}s}" + "".join(f"{str(v):>{cw}s}" for v in vals))

        def row_pct(label, vals):
            print(f"  {label:<{lw}s}" + "".join(f"{v:>{cw - 1}.1f}%" for v in vals))

        # Speech
        row_time("Speech A", [r[1]["speech_a"] for r in results])
        row_time("Speech B", [r[1]["speech_b"] for r in results])
        print(sep)

        # Silence
        row_time("Silence (A's turn)", [r[1]["silence_a"] for r in results])
        row_time("Silence (B's turn)", [r[1]["silence_b"] for r in results])
        row_time("Silence (initial)", [r[1]["silence_init"] for r in results])
        row_time("Total silence", [r[1]["total_silence"] for r in results])
        print(sep)

        # Speaker changes
        row_str("Speaker changes", [r[1]["speaker_changes"] for r in results])
        print(sep)

        # Shares
        row_pct("A share", [r[1]["a_share"] for r in results])
        row_pct("B share", [r[1]["b_share"] for r in results])
        print(sep)

        # SB agreement
        agree_vals = [f"{r[2]:.1f}%" if r[2] is not None else "—" for r in results]
        row_str("SB agreement", agree_vals)
        print(sep)

        # ── Dual-Assignment Overlap Sweep ──
        print(f"\n\n{'=' * 120}")
        print(f"  DUAL-ASSIGNMENT SWEEP — Clip {clip_num}")
        print(f"  When both sim_a >= dual_t AND sim_b >= dual_t, credit segment to both speakers")
        print(f"{'=' * 120}")

        baseline_dec = apply_strategy(segments, {"type": "baseline"})
        dual_results = run_dual_sweep(
            segments, baseline_dec, sb_tl, label_map, duration_ms)

        # Table header
        dcw = 11  # dual column width
        dlw = 14  # dual label width
        dual_cols = ["Baseline"] + [r["name"] for r in dual_results] + ["SpeechBrain"]
        d_header = f"  {'':>{dlw}s}" + "".join(f"{c:>{dcw}s}" for c in dual_cols)
        d_sep = "  " + "─" * (dlw + dcw * len(dual_cols))

        print(f"\n{d_sep}")
        print(d_header)
        print(d_sep)

        def d_row_time(label, vals):
            print(f"  {label:>{dlw}s}" + "".join(f"{format_ms(v):>{dcw}s}" for v in vals))

        def d_row_str(label, vals):
            print(f"  {label:>{dlw}s}" + "".join(f"{str(v):>{dcw}s}" for v in vals))

        def d_row_pct(label, vals):
            print(f"  {label:>{dlw}s}" + "".join(f"{v:>{dcw - 1}.1f}%" for v in vals))

        # Baseline metrics for reference
        bl_m = results[0][1]  # first result is baseline
        sb_m = sb_metrics

        # Speech A
        d_row_time("Speech A", [bl_m["speech_a"]] + [r["speech_a"] for r in dual_results] + [sb_m["speech_a"]])
        d_row_time("Speech B", [bl_m["speech_b"]] + [r["speech_b"] for r in dual_results] + [sb_m["speech_b"]])
        d_row_time("Overlap", [0] + [r["overlap_time"] for r in dual_results] + [0])
        print(d_sep)

        # Shares
        d_row_pct("A share", [bl_m["a_share"]] + [r["a_share"] for r in dual_results] + [sb_m["a_share"]])
        d_row_pct("B share", [bl_m["b_share"]] + [r["b_share"] for r in dual_results] + [sb_m["b_share"]])
        print(d_sep)

        # Dual segments count
        d_row_str("Dual segs", ["0"] + [str(r["dual_count"]) for r in dual_results] + ["—"])
        d_row_str("Changes", [str(bl_m["speaker_changes"])] + [str(r["speaker_changes"]) for r in dual_results] + [str(sb_m["speaker_changes"])])
        print(d_sep)

        # Agreement
        bl_agree = results[0][2]  # baseline agreement
        d_row_str("Strict SB", [f"{bl_agree:.1f}%"] + [f"{r['strict_agree']:.1f}%" for r in dual_results] + ["—"])
        d_row_str("Relaxed SB", [f"{bl_agree:.1f}%"] + [f"{r['relaxed_agree']:.1f}%" for r in dual_results] + ["—"])
        print(d_sep)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
