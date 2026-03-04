#!/usr/bin/env python3
"""Visualize real-world audio test results.

Reads per-clip JSON results from test clips/results/ and generates PNG visualizations:
  - Timeline comparison: our pipeline (top) vs speechbrain (bottom), disagreements highlighted
  - Similarity trace: sim_A and sim_B over time, threshold lines, ambiguous regions

Usage:
  python3 scripts/visualize_results.py                          # all available clips
  python3 scripts/visualize_results.py --clips 1 2              # specific clips
  python3 scripts/visualize_results.py --clips 1 --no-show      # save only
"""

import argparse
import json
import os
import sys

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips", "results")


def load_clip_results(clip_num):
    path = os.path.join(RESULTS_DIR, f"clip{clip_num}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def timeline_to_arrays(timeline, duration_ms, resolution_ms=32.0):
    """Convert a timeline to a frame-level label array for plotting."""
    n_frames = int(duration_ms / resolution_ms) + 1
    labels = np.zeros(n_frames)
    times = np.arange(n_frames) * resolution_ms / 1000

    for seg in timeline:
        start_f = int(seg["start_ms"] / resolution_ms)
        end_f = int(seg["end_ms"] / resolution_ms)
        label = seg["label"]
        val = 0
        if label == "A" or label == "SB_0":
            val = 1
        elif label == "B" or label == "SB_1":
            val = -1
        elif label == "silence":
            val = 0
        for f in range(max(0, start_f), min(n_frames, end_f)):
            labels[f] = val

    return times, labels


def _fmt(ms):
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    m, s = divmod(ms / 1000, 60)
    return f"{int(m)}m{s:04.1f}s"


def _compute_sb_metrics(result):
    comp = result.get("comparison", {})
    sb_tl = result.get("sb_timeline", [])
    duration_ms = result["duration_s"] * 1000

    sb_wta = comp.get("sb_wta_ms", 0)
    sb_wtb = comp.get("sb_wtb_ms", 0)
    sb_speech = sum(s["end_ms"] - s["start_ms"] for s in sb_tl)
    sb_silence = duration_ms - sb_speech

    return {
        "wta": sb_wta,
        "wtb": sb_wtb,
        "speech_total": sb_speech,
        "silence": sb_silence,
    }


def visualize_clip(result, label_map=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    clip_num = result["clip"]
    duration_s = result["duration_s"]
    duration_ms = duration_s * 1000

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 2.2, 1.8], hspace=0.35)

    fig.suptitle(f"Clip {clip_num}: {result.get('description', '')} ({duration_s:.0f}s)",
                 fontsize=14, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(gs[0])
    times, our_labels = timeline_to_arrays(result["timeline"], duration_ms)
    _plot_timeline(ax1, times, our_labels, "Our Pipeline (MFCC)")

    ax2 = fig.add_subplot(gs[1])
    if "sb_timeline" in result and result["sb_timeline"]:
        sb_timeline = result["sb_timeline"]
        if label_map:
            sb_timeline = [dict(s) for s in sb_timeline]
            for seg in sb_timeline:
                mapped = label_map.get(seg["label"], seg["label"])
                seg["label"] = mapped

        _, sb_labels = timeline_to_arrays(sb_timeline, duration_ms)
        _plot_timeline(ax2, times, sb_labels, "SpeechBrain (ECAPA-TDNN)")

        min_len = min(len(our_labels), len(sb_labels))
        disagree = np.zeros(min_len, dtype=bool)
        for i in range(min_len):
            if our_labels[i] != 0 and sb_labels[i] != 0 and our_labels[i] != sb_labels[i]:
                disagree[i] = True
        if np.any(disagree):
            disagree_times = times[:min_len][disagree]
            for t in disagree_times:
                ax1.axvline(x=t, color="red", alpha=0.1, linewidth=0.5)
                ax2.axvline(x=t, color="red", alpha=0.1, linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "No SpeechBrain data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")
        ax2.set_xlim(0, duration_s)

    ax3 = fig.add_subplot(gs[2])
    trace = result.get("similarity_trace", [])
    if trace:
        trace_times = [t["time_ms"] / 1000 for t in trace]
        sim_a = [t["sim_a"] for t in trace]
        sim_b = [t["sim_b"] for t in trace]
        ambiguous = [t["ambiguous"] for t in trace]

        ax3.plot(trace_times, sim_a, color="#2196F3", alpha=0.7, linewidth=0.8, label="sim_A")
        ax3.plot(trace_times, sim_b, color="#FF5722", alpha=0.7, linewidth=0.8, label="sim_B")

        for i, (t, amb) in enumerate(zip(trace_times, ambiguous)):
            if amb:
                ax3.axvspan(t - 0.016, t + 0.016, color="yellow", alpha=0.3)

        ax3.axhline(y=0.80, color="gray", linestyle="--", alpha=0.5, label="threshold (0.80)")
        ax3.axhline(y=0.60, color="gray", linestyle=":", alpha=0.3, label="OVT threshold (0.60)")

        ax3.set_ylabel("Cosine Similarity")
        ax3.set_xlabel("Time (s)")
        ax3.set_xlim(0, duration_s)
        ax3.set_ylim(-1, 1.1)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.set_title("Similarity Trace", fontsize=10)
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(0.5, 0.5, "No similarity trace data", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="gray")

    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    _draw_metrics_table(ax4, result, trace)

    out_path = os.path.join(RESULTS_DIR, f"clip{clip_num}_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    return fig


def _draw_metrics_table(ax, result, trace):
    pm = result["pipeline_metrics"]
    comp = result.get("comparison", {})
    has_sb = bool(comp)
    sb = _compute_sb_metrics(result) if has_sb else {}
    duration_ms = result["duration_s"] * 1000

    if has_sb:
        col_labels = ["Metric", "Our Pipeline", "SpeechBrain", "Difference"]
    else:
        col_labels = ["Metric", "Our Pipeline"]

    rows = []

    def add_row(label, our_val, sb_val=None):
        our_str = _fmt(our_val)
        if has_sb and sb_val is not None:
            sb_str = _fmt(sb_val)
            diff = our_val - sb_val
            diff_str = f"{'+' if diff >= 0 else ''}{_fmt(abs(diff))}"
            if diff < 0:
                diff_str = f"-{_fmt(abs(diff))}"
            rows.append([label, our_str, sb_str, diff_str])
        elif has_sb:
            rows.append([label, our_str, "\u2014", "\u2014"])
        else:
            rows.append([label, our_str])

    add_row("Speaker A Time (WTA)", pm["wta"], sb.get("wta"))
    add_row("Speaker B Time (WTB)", pm["wtb"], sb.get("wtb"))
    add_row("Total Speech", pm["wta"] + pm["wtb"], sb.get("speech_total"))
    add_row("Silence Time A (STA)", pm["sta"])
    add_row("Silence Time B (STB)", pm["stb"])
    add_row("Silence Time Mixed (STM)", pm["stm"])
    add_row("Conv Time A (CTA)", pm["cta"])
    add_row("Conv Time B (CTB)", pm["ctb"])
    add_row("Total Conv Time (TCT)", pm["tct"])
    add_row("Total Silence Time (TST)", pm["tst"], sb.get("silence"))
    add_row("Before/Final Silence (BFST)", pm["bfst"])
    add_row("Overlap Time (OVT)", pm.get("ovt", 0))
    add_row("Total Recording Time (TRT)", pm["trt"])

    n_cols = len(col_labels)
    if n_cols == 4:
        col_widths = [0.32, 0.2, 0.2, 0.18]
    else:
        col_widths = [0.45, 0.25]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(ha="left", fontweight="bold")
        table[i, 0].set_facecolor("#ECEFF1")

    key_rows = {"Speaker A Time (WTA)", "Speaker B Time (WTB)", "Total Conv Time (TCT)",
                "Overlap Time (OVT)"}
    for i, row in enumerate(rows, start=1):
        if row[0] in key_rows:
            for j in range(1, n_cols):
                table[i, j].set_facecolor("#E3F2FD")

    if n_cols == 4:
        for i, row in enumerate(rows, start=1):
            if row[3] != "\u2014":
                table[i, 3].set_text_props(fontweight="bold")

    summary_parts = []
    if has_sb:
        summary_parts.append(f"Frame-level agreement: {comp.get('agreement_pct', 0):.1f}%")
        summary_parts.append(f"({comp.get('agreed_frames', 0)}/{comp.get('total_speech_frames', 0)} frames)")
    if trace:
        n_amb = sum(1 for t in trace if t["ambiguous"])
        summary_parts.append(f"Ambiguous decisions: {n_amb}/{len(trace)} ({n_amb/max(len(trace),1)*100:.1f}%)")
    ovt = pm.get("ovt", 0)
    tct = max(pm.get("tct", 1), 1)
    summary_parts.append(f"OVT/TCT: {ovt/tct*100:.1f}%")

    invariant_ok = abs(pm["trt"] - pm["tct"] - pm["bfst"]) < 1
    summary_parts.append(f"TRT=TCT+BFST: {'PASS' if invariant_ok else 'FAIL'}")

    ax.set_title("Metrics Comparison", fontsize=10, pad=10)
    ax.text(0.5, -0.02, "   |   ".join(summary_parts),
            ha="center", va="top", transform=ax.transAxes, fontsize=9, color="#546E7A")


def _plot_timeline(ax, times, labels, title):
    import matplotlib.patches as mpatches

    colors = {1: "#2196F3", -1: "#FF5722", 0: "#E0E0E0"}

    for i in range(len(labels) - 1):
        val = labels[i]
        color = colors.get(val, "#E0E0E0")
        ax.axvspan(times[i], times[i + 1], color=color, alpha=0.6 if val != 0 else 0.2)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])

    patches = [
        mpatches.Patch(color="#2196F3", alpha=0.6, label="Speaker A"),
        mpatches.Patch(color="#FF5722", alpha=0.6, label="Speaker B"),
        mpatches.Patch(color="#E0E0E0", alpha=0.2, label="Silence"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=3)


def main():
    parser = argparse.ArgumentParser(description="Visualize real-world audio test results")
    parser.add_argument("--clips", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"=== Visualize Results ===")
    print(f"Results dir: {RESULTS_DIR}")

    for clip_num in args.clips:
        result = load_clip_results(clip_num)
        if result is None:
            print(f"\n  Clip {clip_num}: no results found, skipping")
            continue

        print(f"\n  Clip {clip_num}: generating visualization...")

        label_map = None
        if "comparison" in result:
            label_map = result["comparison"].get("label_map")

        visualize_clip(result, label_map)

    if args.show:
        plt.show()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
