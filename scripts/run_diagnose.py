#!/usr/bin/env python3
"""Run diagnostic on clips 1 and 2: per-segment analysis + SpeechBrain comparison.

Clip 1 (trimmed to start at original 1:45):
  Ground truth (approximate):
    0-7s:    Speaker A (brief silence in middle)
    7-20s:   Speaker B
    20-25s:  Rapid A/B alternation (brief sentences)
    25s+:    Speaker A speaks longer

Clip 2:
  Ground truth (first 19s):
    0-7s:    Speaker A (brief breath at start)
    7-19s:   Speaker B
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS, SAMPLE_RATE, SPEECH_SEGMENT_FRAMES, MIN_FLUSH_FRAMES,
    PipelineSimulator, SpeakerIdentifier, SpeechBrainDiarizer,
    WeSpeakerEmbedder, compare_timelines, create_silero_vad,
    extract_vad_segments, format_ms, load_audio,
)

import numpy as np

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
RESULTS_DIR = os.path.join(CLIP_DIR, "results")

CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

# Ground truths (approximate, user-verified)
GROUND_TRUTHS = {
    1: [  # Clip 1 trimmed (starts at original 1:45)
        (0, 7000, "A"),
        (7000, 20000, "B"),
        (20000, 25000, "mixed"),  # rapid alternation
        (25000, 50000, "A"),      # A speaks longer (approximate end)
    ],
    2: [  # Clip 2 (first 19s)
        (0, 7000, "A"),
        (7000, 19000, "B"),
    ],
}

THRESHOLD = 0.80
BUFFER = 47
BCONF = 2
SMOOTHING = 2


def run_vad_segments(audio, vad_fn):
    """Run VAD, return segments list."""
    from validate_audio import FRAME_SIZE
    n_frames = len(audio) // FRAME_SIZE
    segments = []
    current_label = None
    seg_start = 0

    for i in range(n_frames):
        start = i * FRAME_SIZE
        frame = audio[start:start + FRAME_SIZE]
        is_speech = vad_fn(frame)
        label = "speech" if is_speech else "silence"
        if label != current_label:
            if current_label is not None:
                segments.append((seg_start, round(i * FRAME_MS), current_label))
            current_label = label
            seg_start = round(i * FRAME_MS)
    if current_label is not None:
        segments.append((seg_start, round(n_frames * FRAME_MS), current_label))
    return segments


def run_verbose_pipeline(audio, vad_fn, embedder, threshold, buffer_frames, bconf):
    """Run neural pipeline with per-segment logging. Returns segment_log, timeline, embeddings."""
    from validate_audio import FRAME_SIZE
    speaker_id = SpeakerIdentifier(threshold=threshold, margin=0.0, b_confirm_frames=bconf,
                                    min_frames_for_b=buffer_frames)

    n_frames = len(audio) // FRAME_SIZE
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
            "is_first": speaker_id.ref_a is not None and len(segment_log) == 0,
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

    return segment_log, timeline, all_embeddings


def generate_png(clip_num, duration_s, vad_segments, our_timeline, sb_timeline,
                 segment_log, ground_truth, threshold, buffer_frames):
    """Generate timeline PNG for a clip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from validate_audio import compare_timelines as ct

    # Determine how many seconds to show in detail (first 50s) and full clip
    detail_end = min(50, duration_s)

    n_panels = 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 14),
                             gridspec_kw={"height_ratios": [0.5, 0.5, 0.5, 0.5, 2.0]})

    fig.suptitle(f"Clip {clip_num} Diagnostic — {duration_s:.0f}s total, detail: first {detail_end:.0f}s\n"
                 f"Neural: threshold={threshold}, buffer={buffer_frames} ({round(buffer_frames * FRAME_MS)}ms)",
                 fontsize=12, fontweight="bold", y=0.98)

    ca, cb, cs = "#2196F3", "#FF5722", "#E0E0E0"

    def plot_bar(ax, timeline, title, label_map=None, xlim=None):
        for start_ms, end_ms, label in timeline:
            t0, t1 = start_ms / 1000, end_ms / 1000
            mapped = label_map.get(label, label) if label_map else label
            if mapped == "A":
                color, alpha = ca, 0.7
            elif mapped == "B":
                color, alpha = cb, 0.7
            elif mapped in ("speech", "mixed"):
                color, alpha = "#9E9E9E", 0.4
            else:
                color, alpha = cs, 0.15
            ax.axvspan(t0, t1, color=color, alpha=alpha)
        lim = xlim or (0, duration_s)
        ax.set_xlim(*lim)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, alpha=0.15, axis="x")

    # Panel 0: Ground truth
    plot_bar(axes[0], ground_truth, "Ground Truth (user-verified)", xlim=(0, detail_end))

    # Panel 1: VAD
    plot_bar(axes[1], vad_segments, "VAD (speech/silence)", xlim=(0, detail_end))

    # Panel 2: Our pipeline (detail)
    plot_bar(axes[2], our_timeline,
             f"Our Pipeline (t={threshold}, buf={buffer_frames})", xlim=(0, detail_end))

    # Panel 3: SpeechBrain (detail)
    if sb_timeline:
        duration_ms = duration_s * 1000
        cmp = ct([(s, e, l) for s, e, l in our_timeline if l != "silence"],
                 sb_timeline, duration_ms)
        lmap = cmp.get("label_map", {})
        plot_bar(axes[3], sb_timeline, "SpeechBrain (offline clustering)",
                 label_map=lmap, xlim=(0, detail_end))
    else:
        axes[3].set_xlim(0, detail_end)
        axes[3].text(0.5, 0.5, "No SB data", ha="center", va="center",
                     transform=axes[3].transAxes)

    # Legend on first panel
    patches = [
        mpatches.Patch(color=ca, alpha=0.7, label="Speaker A"),
        mpatches.Patch(color=cb, alpha=0.7, label="Speaker B"),
        mpatches.Patch(color=cs, alpha=0.15, label="Silence"),
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

    # Panel 4: Similarity trace (full clip)
    ax = axes[4]
    if segment_log:
        times_s = [e["mid_ms"] / 1000 for e in segment_log]
        sim_a = [e["sim_a"] for e in segment_log]
        sim_b_all = [(e["mid_ms"] / 1000, e["sim_b"]) for e in segment_log if e["ref_b_exists"]]

        ax.plot(times_s, sim_a, ".-", color=ca, markersize=2, linewidth=0.5, alpha=0.7, label="sim_A")
        if sim_b_all:
            tb, sb_v = zip(*sim_b_all)
            ax.plot(tb, sb_v, ".-", color=cb, markersize=2, linewidth=0.5, alpha=0.7, label="sim_B")

        ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"threshold ({threshold})")

        # Mark ground truth regions
        for start, end, label in ground_truth:
            if label == "A":
                ax.axvspan(start / 1000, end / 1000, color=ca, alpha=0.04)
            elif label == "B":
                ax.axvspan(start / 1000, end / 1000, color=cb, alpha=0.04)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Similarity Trace (full clip)", fontsize=9, loc="left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS_DIR, f"clip{clip_num}_diagnostic.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    clips_to_run = [1, 2]
    if len(sys.argv) > 1:
        clips_to_run = [int(x) for x in sys.argv[1:]]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Diagnostic: clips {clips_to_run} ===")
    print(f"Config: threshold={THRESHOLD}, buffer={BUFFER}, smoothing={SMOOTHING}, bconf={BCONF}")

    embedder = WeSpeakerEmbedder()

    for clip_num in clips_to_run:
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

        # VAD
        vad_fn1 = create_silero_vad()
        vad_segs = run_vad_segments(audio, vad_fn1)
        speech_ms = sum(e - s for s, e, l in vad_segs if l == "speech")
        print(f"  VAD: {format_ms(speech_ms)} speech / {format_ms(duration_ms)} total")

        # Our pipeline (verbose on first 30s worth of segments)
        print(f"  Running neural pipeline...")
        vad_fn2 = create_silero_vad()
        t0 = time.time()
        seg_log, our_tl, all_emb = run_verbose_pipeline(
            audio, vad_fn2, embedder, THRESHOLD, BUFFER, BCONF)
        elapsed = time.time() - t0
        print(f"  Pipeline done in {elapsed:.1f}s, {len(seg_log)} segments")

        # Print first ~25 segments
        print(f"\n  {'Seg':>3s} {'Start':>7s} {'End':>7s} {'Dur':>5s} "
              f"{'SimA':>6s} {'SimB':>6s} {'Dec':>3s} {'RefB':>4s}")
        print(f"  {'-' * 55}")
        for e in seg_log[:25]:
            print(f"  {e['idx']:>3d} {format_ms(e['start_ms']):>7s} {format_ms(e['end_ms']):>7s} "
                  f"{format_ms(e['duration_ms']):>5s} "
                  f"{e['sim_a']:>6.3f} {e['sim_b']:>6.3f} "
                  f"  {e['decision']} {'yes' if e['ref_b_exists'] else 'no':>4s}")
        if len(seg_log) > 25:
            print(f"  ... ({len(seg_log) - 25} more segments)")

        # Our metrics
        our_wta = sum(e - s for s, e, l in our_tl if l == "A")
        our_wtb = sum(e - s for s, e, l in our_tl if l == "B")
        our_sil = sum(e - s for s, e, l in our_tl if l == "silence")

        # Speaker changes
        changes = 0
        prev = None
        for _, _, lbl in our_tl:
            if lbl in ("A", "B") and lbl != prev and prev in ("A", "B"):
                changes += 1
            if lbl in ("A", "B"):
                prev = lbl

        # SpeechBrain
        print(f"\n  Running SpeechBrain...")
        diarizer = SpeechBrainDiarizer()
        vad_fn3 = create_silero_vad()
        baseline = PipelineSimulator(vad_fn3, smoothing_window=1,
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
        lmap = cmp.get("label_map", {})

        # SB silence
        sb_speech = sum(e - s for s, e, l in sb_tl)
        sb_sil = duration_ms - sb_speech

        # Summary
        print(f"\n  {'=' * 60}")
        print(f"  COMPARISON: Clip {clip_num}")
        print(f"  {'=' * 60}")
        print(f"  {'':>15s} {'WTA':>8s} {'WTB':>8s} {'Silence':>8s} {'Changes':>8s}")
        print(f"  {'Our pipeline':>15s} {format_ms(our_wta):>8s} {format_ms(our_wtb):>8s} "
              f"{format_ms(our_sil):>8s} {changes:>8d}")
        print(f"  {'SpeechBrain':>15s} {format_ms(sb_wta):>8s} {format_ms(sb_wtb):>8s} "
              f"{format_ms(sb_sil):>8s} {'—':>8s}")
        print(f"\n  Agreement: {cmp['agreement_pct']:.1f}%  "
              f"({cmp['agreed_frames']}/{cmp['total_speech_frames']} frames)")
        print(f"  Label map: {lmap}")

        # Ground truth comparison (first portion)
        gt = GROUND_TRUTHS.get(clip_num, [])
        if gt:
            gt_a = sum(e - s for s, e, l in gt if l == "A")
            gt_b = sum(e - s for s, e, l in gt if l == "B")
            print(f"\n  Ground truth (first {max(e for _, e, _ in gt)/1000:.0f}s): "
                  f"A={format_ms(gt_a)}, B={format_ms(gt_b)}")

        # Generate PNG
        generate_png(clip_num, duration_s, vad_segs, our_tl, sb_tl,
                     seg_log, gt, THRESHOLD, BUFFER)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
