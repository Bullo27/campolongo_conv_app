#!/usr/bin/env python3
"""First-principles diagnostic on clip 2's first 19 seconds.

Ground truth (user-verified):
  - ~0-7s: Speaker A (after brief breath at start)
  - ~7-19s: Speaker B (continuous)

Runs:
  1. VAD: shows exactly which frames are speech
  2. Our neural pipeline with verbose per-segment logging
  3. SpeechBrain offline diarization
  4. Embedding similarity matrix (within-speaker vs cross-speaker)
  5. Timeline PNG visualization

Usage:
  python3 scripts/diagnose_clip2.py
  python3 scripts/diagnose_clip2.py --threshold 0.65 --buffer 47
  python3 scripts/diagnose_clip2.py --duration 19  # first N seconds
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS,
    FRAME_SIZE,
    SAMPLE_RATE,
    SPEECH_SEGMENT_FRAMES,
    MIN_FLUSH_FRAMES,
    SpeakerIdentifier,
    SpeechBrainDiarizer,
    WeSpeakerEmbedder,
    create_silero_vad,
    extract_vad_segments,
    format_ms,
    load_audio,
)

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIP2_PATH = os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3")
RESULTS_DIR = os.path.join(CLIP_DIR, "results")

# Ground truth for first 19s of clip 2
GROUND_TRUTH = [
    (0, 7000, "A"),
    (7000, 19000, "B"),
]


def extract_first_n_seconds(audio_int16, duration_s):
    """Extract first N seconds of audio."""
    n_samples = int(duration_s * SAMPLE_RATE)
    return audio_int16[:n_samples]


def run_vad(audio_int16, vad_fn):
    """Run VAD frame by frame, return per-frame speech/silence and segments."""
    n_frames = len(audio_int16) // FRAME_SIZE
    vad_results = []  # list of (frame_idx, time_ms, is_speech)

    for i in range(n_frames):
        start = i * FRAME_SIZE
        frame = audio_int16[start:start + FRAME_SIZE]
        is_speech = vad_fn(frame)
        vad_results.append((i, round(i * FRAME_MS), is_speech))

    # Build segments
    segments = []
    current_label = None
    seg_start = 0
    for i, time_ms, is_speech in vad_results:
        label = "speech" if is_speech else "silence"
        if label != current_label:
            if current_label is not None:
                segments.append((seg_start, time_ms, current_label))
            current_label = label
            seg_start = time_ms
    if current_label is not None:
        segments.append((seg_start, round(n_frames * FRAME_MS), current_label))

    return vad_results, segments


def run_neural_pipeline_verbose(audio_int16, vad_fn, embedder, threshold, buffer_frames,
                                 smoothing, bconf):
    """Run neural pipeline with detailed per-segment logging.

    Returns:
        segment_log: list of dicts with per-segment info
        timeline: list of (start_ms, end_ms, label)
        all_embeddings: list of (mid_time_ms, embedding, label) for similarity analysis
    """
    speaker_id = SpeakerIdentifier(
        threshold=threshold,
        margin=0.0,
        b_confirm_frames=bconf,
        min_frames_for_b=buffer_frames,
    )

    n_frames = len(audio_int16) // FRAME_SIZE
    speech_buffer = []
    speech_buffer_start_frame = 0
    timeline = []
    segment_log = []
    all_embeddings = []

    def identify_and_log(current_frame):
        nonlocal speech_buffer
        mid_frame = (speech_buffer_start_frame + current_frame) // 2
        time_ms = round(mid_frame * FRAME_MS)
        n_buffered = len(speech_buffer)

        # Extract embedding
        raw_audio = np.concatenate(speech_buffer)
        speech_buffer = []
        embedding = embedder.extract_embedding(raw_audio)

        # Get state before identification
        ref_a_before = speaker_id.ref_a.copy() if speaker_id.ref_a is not None else None
        ref_b_before = speaker_id.ref_b.copy() if speaker_id.ref_b is not None else None

        # Identify (pass n_frames so short flush segments can't trigger B)
        raw_speaker = speaker_id.identify(embedding, n_frames=n_buffered)

        # Log everything
        start_ms = round(speech_buffer_start_frame * FRAME_MS)
        end_ms = round(current_frame * FRAME_MS)

        entry = {
            "segment_idx": len(segment_log),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "mid_ms": time_ms,
            "n_frames": n_buffered,
            "duration_ms": round(n_buffered * FRAME_MS),
            "sim_a": round(speaker_id.last_sim_a, 4),
            "sim_b": round(speaker_id.last_sim_b, 4),
            "decision": raw_speaker,
            "ref_a_exists": speaker_id.ref_a is not None,
            "ref_b_exists": speaker_id.ref_b is not None,
            "b_candidate_count": speaker_id._b_candidate_count,
            "is_first_segment": ref_a_before is None,
        }
        segment_log.append(entry)
        all_embeddings.append((time_ms, embedding.copy(), raw_speaker))

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
        frame = audio_int16[start:start + FRAME_SIZE]
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

    # Flush remaining
    if len(speech_buffer) >= MIN_FLUSH_FRAMES:
        buf_start = speech_buffer_start_frame
        speaker = identify_and_log(n_frames - 1)
        update_timeline(n_frames - 1, speaker, start_frame=buf_start)

    return segment_log, timeline, all_embeddings


def compute_similarity_matrix(all_embeddings):
    """Compute pairwise cosine similarity between all segment embeddings."""
    n = len(all_embeddings)
    if n < 2:
        return None

    embeddings = np.array([e[1] for e in all_embeddings])
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms
    # Cosine similarity matrix
    sim_matrix = normalized @ normalized.T
    return sim_matrix


def print_segment_log(segment_log):
    """Print detailed per-segment log."""
    print(f"\n{'=' * 95}")
    print(f"  PER-SEGMENT LOG (neural pipeline)")
    print(f"{'=' * 95}")
    header = (f"{'Seg':>3s} {'Start':>7s} {'End':>7s} {'Dur':>6s} "
              f"{'SimA':>7s} {'SimB':>7s} {'Dec':>4s} "
              f"{'RefA':>4s} {'RefB':>4s} {'BCand':>5s} {'Notes'}")
    print(header)
    print("-" * 95)

    for e in segment_log:
        notes = []
        if e["is_first_segment"]:
            notes.append("ANCHOR (sets ref_A)")
        if not e["ref_b_exists"] and e["decision"] == "A" and e["sim_a"] < 0.8:
            notes.append(f"low sim but no B yet")
        if e["b_candidate_count"] > 0:
            notes.append(f"B-candidate streak")

        print(f"{e['segment_idx']:>3d} "
              f"{format_ms(e['start_ms']):>7s} "
              f"{format_ms(e['end_ms']):>7s} "
              f"{format_ms(e['duration_ms']):>6s} "
              f"{e['sim_a']:>7.4f} "
              f"{e['sim_b']:>7.4f} "
              f"  {e['decision']:>1s}  "
              f"{'yes' if e['ref_a_exists'] else 'no':>4s} "
              f"{'yes' if e['ref_b_exists'] else 'no':>4s} "
              f"{e['b_candidate_count']:>5d} "
              f"{'  '.join(notes)}")


def print_similarity_analysis(all_embeddings, sim_matrix):
    """Analyze within-speaker vs cross-speaker cosine similarities."""
    if sim_matrix is None or len(all_embeddings) < 3:
        print("\n  Not enough segments for similarity analysis.")
        return

    n = len(all_embeddings)
    times = [e[0] for e in all_embeddings]

    # Classify segments by ground truth time
    gt_labels = []
    for time_ms, _, _ in all_embeddings:
        if time_ms < 7000:
            gt_labels.append("A")
        else:
            gt_labels.append("B")

    a_indices = [i for i, l in enumerate(gt_labels) if l == "A"]
    b_indices = [i for i, l in enumerate(gt_labels) if l == "B"]

    print(f"\n{'=' * 80}")
    print(f"  EMBEDDING SIMILARITY ANALYSIS (ground truth labels)")
    print(f"{'=' * 80}")
    print(f"  Segments in A region (0-7s): {len(a_indices)}")
    print(f"  Segments in B region (7-19s): {len(b_indices)}")

    # Within-A similarities
    within_a = []
    for i in range(len(a_indices)):
        for j in range(i + 1, len(a_indices)):
            within_a.append(sim_matrix[a_indices[i], a_indices[j]])

    # Within-B similarities
    within_b = []
    for i in range(len(b_indices)):
        for j in range(i + 1, len(b_indices)):
            within_b.append(sim_matrix[b_indices[i], b_indices[j]])

    # Cross similarities (A vs B)
    cross = []
    for ai in a_indices:
        for bi in b_indices:
            cross.append(sim_matrix[ai, bi])

    def stats(arr, name):
        if not arr:
            print(f"  {name}: no pairs")
            return
        arr = np.array(arr)
        print(f"  {name}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}")

    print()
    stats(within_a, "Within-A (same speaker)")
    stats(within_b, "Within-B (same speaker)")
    stats(cross,    "Cross A↔B (diff speakers)")

    if within_a and within_b and cross:
        within_all = within_a + within_b
        within_mean = np.mean(within_all)
        cross_mean = np.mean(cross)
        gap = within_mean - cross_mean
        print(f"\n  Within-speaker mean: {within_mean:.4f}")
        print(f"  Cross-speaker mean:  {cross_mean:.4f}")
        print(f"  GAP (within - cross): {gap:.4f}")
        if gap > 0.15:
            print(f"  → GOOD separation — model CAN distinguish these speakers")
            print(f"  → Optimal threshold should be ~{(within_mean + cross_mean) / 2:.2f}")
        elif gap > 0.05:
            print(f"  → MARGINAL separation — model struggles but may work with tuning")
        else:
            print(f"  → POOR separation — timbres too similar for this model")

    # Print the full similarity matrix
    print(f"\n  PAIRWISE SIMILARITY MATRIX:")
    print(f"       ", end="")
    for i in range(n):
        gt = gt_labels[i]
        print(f" S{i}{gt:>1s} ", end="")
    print()
    for i in range(n):
        gt_i = gt_labels[i]
        print(f"  S{i}{gt_i:>1s}  ", end="")
        for j in range(n):
            val = sim_matrix[i, j]
            # Highlight: green for within-speaker, red for cross-speaker
            if i == j:
                print(f" 1.00 ", end="")
            else:
                print(f" {val:.2f} ", end="")
        print(f"  [{format_ms(times[i])}]")


def generate_timeline_png(vad_segments, our_timeline, sb_timeline, all_embeddings,
                          sim_matrix, segment_log, duration_s, threshold, buffer_frames,
                          output_path):
    """Generate a comprehensive timeline PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_panels = 5  # VAD, ground truth, ours, speechbrain, similarity trace
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [0.6, 0.6, 0.6, 0.6, 2.0]})

    fig.suptitle(f"Clip 2 — First {duration_s:.0f}s Diagnostic\n"
                 f"Neural: threshold={threshold}, buffer={buffer_frames} frames "
                 f"({round(buffer_frames * FRAME_MS)}ms)",
                 fontsize=13, fontweight="bold", y=0.98)

    colors_a = "#2196F3"
    colors_b = "#FF5722"
    colors_sil = "#E0E0E0"

    def plot_timeline_bar(ax, timeline, title, label_map=None):
        for start_ms, end_ms, label in timeline:
            t0 = start_ms / 1000
            t1 = end_ms / 1000
            mapped = label_map.get(label, label) if label_map else label
            if mapped == "A":
                color, alpha = colors_a, 0.7
            elif mapped == "B":
                color, alpha = colors_b, 0.7
            elif mapped in ("speech",):
                color, alpha = "#9E9E9E", 0.4
            else:
                color, alpha = colors_sil, 0.15
            ax.axvspan(t0, t1, color=color, alpha=alpha)
        ax.set_xlim(0, duration_s)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, alpha=0.15, axis="x")

    # Panel 0: VAD
    plot_timeline_bar(axes[0], vad_segments, "VAD (speech/silence)")

    # Panel 1: Ground truth
    plot_timeline_bar(axes[1], GROUND_TRUTH, "Ground Truth (user-verified)")

    # Panel 2: Our neural pipeline
    plot_timeline_bar(axes[2], our_timeline,
                      f"Our Neural Pipeline (t={threshold}, buf={buffer_frames})")

    # Panel 3: SpeechBrain
    if sb_timeline:
        # Determine best label mapping
        from validate_audio import compare_timelines
        duration_ms = duration_s * 1000
        cmp = compare_timelines(
            [(s, e, l) for s, e, l in our_timeline if l != "silence"],
            sb_timeline, duration_ms
        )
        label_map = cmp.get("label_map", {})
        plot_timeline_bar(axes[3], sb_timeline, "SpeechBrain (offline clustering)",
                          label_map=label_map)
    else:
        axes[3].text(0.5, 0.5, "No SpeechBrain data", ha="center", va="center",
                     transform=axes[3].transAxes, fontsize=11, color="gray")
        axes[3].set_xlim(0, duration_s)

    # Add legend to first panel
    patches = [
        mpatches.Patch(color=colors_a, alpha=0.7, label="Speaker A"),
        mpatches.Patch(color=colors_b, alpha=0.7, label="Speaker B"),
        mpatches.Patch(color=colors_sil, alpha=0.15, label="Silence"),
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8, ncol=3)

    # Panel 4: Similarity trace + embedding annotations
    ax_sim = axes[4]
    if segment_log:
        times_s = [e["mid_ms"] / 1000 for e in segment_log]
        sim_a_vals = [e["sim_a"] for e in segment_log]
        sim_b_vals = [e["sim_b"] for e in segment_log]
        decisions = [e["decision"] for e in segment_log]

        ax_sim.plot(times_s, sim_a_vals, "o-", color=colors_a, markersize=6,
                    linewidth=1.5, alpha=0.8, label="sim_A (vs ref_A)")
        # Only plot sim_B where ref_B exists
        sim_b_valid = [(t, s) for t, s, e in zip(times_s, sim_b_vals, segment_log)
                       if e["ref_b_exists"]]
        if sim_b_valid:
            tb, sb = zip(*sim_b_valid)
            ax_sim.plot(tb, sb, "s-", color=colors_b, markersize=6,
                        linewidth=1.5, alpha=0.8, label="sim_B (vs ref_B)")

        # Threshold line
        ax_sim.axhline(y=threshold, color="gray", linestyle="--", alpha=0.6,
                        label=f"threshold ({threshold})")

        # Annotate each segment with its decision
        for i, e in enumerate(segment_log):
            t = e["mid_ms"] / 1000
            y_val = max(e["sim_a"], e["sim_b"] if e["ref_b_exists"] else 0)
            color = colors_a if e["decision"] == "A" else colors_b
            ax_sim.annotate(f"S{i}→{e['decision']}", (t, y_val + 0.03),
                           fontsize=7, ha="center", color=color, fontweight="bold")

        # Ground truth regions as background
        ax_sim.axvspan(0, 7, color=colors_a, alpha=0.05)
        ax_sim.axvspan(7, duration_s, color=colors_b, alpha=0.05)
        ax_sim.text(3.5, 1.05, "GT: Speaker A", ha="center", fontsize=8, color=colors_a)
        ax_sim.text(13, 1.05, "GT: Speaker B", ha="center", fontsize=8, color=colors_b)

    ax_sim.set_xlabel("Time (s)", fontsize=10)
    ax_sim.set_ylabel("Cosine Similarity", fontsize=10)
    ax_sim.set_xlim(0, duration_s)
    ax_sim.set_ylim(-0.1, 1.15)
    ax_sim.legend(loc="lower right", fontsize=8)
    ax_sim.set_title("Per-Segment Similarity to Speaker References", fontsize=10, loc="left")
    ax_sim.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved PNG: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnose clip 2 first 19s")
    parser.add_argument("--duration", type=float, default=19.0,
                        help="Duration in seconds to analyze (default: 19)")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Cosine similarity threshold (default: 0.65)")
    parser.add_argument("--buffer", type=int, default=47,
                        help="Speech buffer frames (default: 47)")
    parser.add_argument("--smoothing", type=int, default=1,
                        help="Smoothing window (default: 1 = none)")
    parser.add_argument("--bconf", type=int, default=1,
                        help="B-confirm frames (default: 1)")
    parser.add_argument("--no-speechbrain", action="store_true",
                        help="Skip SpeechBrain diarization")
    args = parser.parse_args()

    print(f"=== Clip 2 Diagnostic — First {args.duration}s ===")
    print(f"Config: threshold={args.threshold}, buffer={args.buffer}, "
          f"smoothing={args.smoothing}, bconf={args.bconf}")

    # Load and trim audio
    print(f"\nLoading clip 2...")
    audio_full = load_audio(CLIP2_PATH)
    audio = extract_first_n_seconds(audio_full, args.duration)
    duration_ms = len(audio) / SAMPLE_RATE * 1000
    print(f"  Audio: {len(audio)} samples ({duration_ms:.0f}ms)")

    # 1. VAD
    print(f"\nRunning VAD...")
    vad_fn = create_silero_vad()
    vad_results, vad_segments = run_vad(audio, vad_fn)
    speech_frames = sum(1 for _, _, s in vad_results if s)
    total_frames = len(vad_results)
    print(f"  {speech_frames}/{total_frames} frames = speech "
          f"({speech_frames * FRAME_MS:.0f}ms / {total_frames * FRAME_MS:.0f}ms)")
    print(f"\n  VAD segments:")
    for start, end, label in vad_segments:
        print(f"    {format_ms(start):>7s} - {format_ms(end):>7s}  {label}")

    # 2. Neural pipeline (verbose)
    print(f"\nRunning neural pipeline...")
    embedder = WeSpeakerEmbedder()

    # We need a fresh VAD for the pipeline (VAD has internal state)
    vad_fn2 = create_silero_vad()
    segment_log, our_timeline, all_embeddings = run_neural_pipeline_verbose(
        audio, vad_fn2, embedder,
        threshold=args.threshold,
        buffer_frames=args.buffer,
        smoothing=args.smoothing,
        bconf=args.bconf,
    )

    print_segment_log(segment_log)

    # 3. Similarity matrix analysis
    sim_matrix = compute_similarity_matrix(all_embeddings)
    print_similarity_analysis(all_embeddings, sim_matrix)

    # 4. SpeechBrain
    sb_timeline = None
    if not args.no_speechbrain:
        print(f"\nRunning SpeechBrain offline diarization...")
        diarizer = SpeechBrainDiarizer()
        # Get VAD segments for SB
        speech_segs = [(s, e) for s, e, l in vad_segments if l == "speech"]
        sb_timeline = diarizer.diarize(audio, speech_segs)
        if sb_timeline:
            print(f"\n  SpeechBrain timeline:")
            for start, end, label in sb_timeline:
                print(f"    {format_ms(start):>7s} - {format_ms(end):>7s}  {label}")

    # 5. Summary vs ground truth
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: OUR PIPELINE vs GROUND TRUTH")
    print(f"{'=' * 70}")
    our_wta = sum(e - s for s, e, l in our_timeline if l == "A")
    our_wtb = sum(e - s for s, e, l in our_timeline if l == "B")
    gt_wta = sum(e - s for s, e, l in GROUND_TRUTH if l == "A")
    gt_wtb = sum(e - s for s, e, l in GROUND_TRUTH if l == "B")
    print(f"  Ground truth: A={format_ms(gt_wta)}, B={format_ms(gt_wtb)}")
    print(f"  Our pipeline: A={format_ms(our_wta)}, B={format_ms(our_wtb)}")
    if sb_timeline:
        sb_a = sum(e - s for s, e, l in sb_timeline if "0" in l)
        sb_b = sum(e - s for s, e, l in sb_timeline if "1" in l)
        print(f"  SpeechBrain:  label0={format_ms(sb_a)}, label1={format_ms(sb_b)}")

    # 6. Generate PNG
    os.makedirs(RESULTS_DIR, exist_ok=True)
    png_path = os.path.join(RESULTS_DIR, "clip2_diagnostic.png")
    generate_timeline_png(
        vad_segments, our_timeline, sb_timeline, all_embeddings,
        sim_matrix, segment_log, args.duration, args.threshold, args.buffer,
        png_path,
    )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
