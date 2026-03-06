#!/usr/bin/env python3
"""Compare 2-tier adaptive with LOW=0.82 vs LOW=0.84 as default tier.

Runs the same scenarios, sweeps independently for each default, then
prints side-by-side end-to-end comparison including clean-clip baseline.
"""
import os, sys, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS, FRAME_SIZE, SAMPLE_RATE, SPEECH_SEGMENT_FRAMES,
    PipelineSimulator, SpeakerIdentifier, SpeechBrainDiarizer,
    WeSpeakerEmbedder, create_silero_vad, extract_vad_segments,
    compare_timelines, format_ms, load_audio,
)
from strategy_comparison import (
    run_pipeline_segments, apply_strategy, build_timeline,
    compute_dual_metrics, relaxed_agreement,
)
from noise_adaptive_test import create_mixed_noise_clip

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}
GT = {
    1: {"speech_a": 171000, "speech_b": 35000, "overlap": 10000},
    2: {"speech_a": 195000, "speech_b": 579000, "overlap": 35000},
}

DUAL_CAP = 3

SCENARIOS = {
    "A": {"desc": "30→Clean",
           1: [(0,100,30),(100,202,None)],
           2: [(0,375,30),(375,751,None)]},
    "B": {"desc": "35→Clean",
           1: [(0,100,35),(100,202,None)],
           2: [(0,375,35),(375,751,None)]},
    "C": {"desc": "30→35",
           1: [(0,100,30),(100,202,35)],
           2: [(0,375,30),(375,751,35)]},
    "D": {"desc": "35→30",
           1: [(0,100,35),(100,202,30)],
           2: [(0,375,35),(375,751,30)]},
    "E": {"desc": "30→35→Cl",
           2: [(0,250,30),(250,500,35),(500,751,None)]},
}

# Sweep ranges
T_HIGH_RANGE = [0.73, 0.74, 0.75, 0.76, 0.77, 0.78]
N_RANGE = [10, 15, 20, 30]
MARGIN_RANGE = [0.005, 0.01, 0.015, 0.02, 0.025]
DWELL_RANGE = [5, 10, 15, 20, 30]
WARMUP_MULT = [0.5, 1.0, 2.0]


def simulate_2tier(segments, params, default_tier="LOW"):
    """2-tier: default_tier ↔ HIGH."""
    alpha = params["alpha"]
    t_high = params["t_high"]
    margin = params["margin"]
    min_dwell = params["min_dwell"]
    warmup = params["warmup"]

    tiers = []
    ema = 0.0
    current_tier = default_tier
    segs_since_change = min_dwell
    b_established = False
    segs_since_b = 0
    tier_changes = []

    for i, seg in enumerate(segments):
        sim_b, sim_a = seg.get("sim_b"), seg.get("sim_a")

        if not b_established:
            if sim_b is not None and sim_b > 0:
                b_established = True
                ema = min(sim_a, sim_b)
                segs_since_b = 0
            tiers.append(current_tier)
            continue

        ema = alpha * min(sim_a, sim_b) + (1 - alpha) * ema
        segs_since_b += 1

        if segs_since_b < warmup:
            tiers.append(current_tier)
            continue

        segs_since_change += 1
        new_tier = current_tier

        if current_tier != "HIGH":
            if ema > t_high + margin:
                new_tier = "HIGH"
        else:
            if ema < t_high - margin:
                new_tier = default_tier

        if new_tier != current_tier and segs_since_change >= min_dwell:
            tier_changes.append((i, current_tier, new_tier, ema))
            current_tier = new_tier
            segs_since_change = 0

        tiers.append(current_tier)

    return tiers, {"tier_changes": tier_changes}


def apply_dual(segments, baseline_dec, tier_assignments, tier_dual_t):
    """Apply dual-assignment with per-segment tier."""
    labels = []
    consec_dual = 0
    for seg, primary, tier in zip(segments, baseline_dec, tier_assignments):
        dual_t = tier_dual_t[tier]
        sim_a, sim_b = seg["sim_a"], seg["sim_b"]
        other = "B" if primary == "A" else "A"
        both = (sim_a is not None and sim_b is not None
                and sim_a >= dual_t and sim_b >= dual_t)
        is_dual = both and consec_dual < DUAL_CAP
        if is_dual:
            consec_dual += 1
            labels.append({primary, other})
        else:
            if not both: consec_dual = 0
            labels.append({primary})
    return labels


def metrics_from_labels(segments, baseline_dec, labels):
    speech_a = speech_b = overlap = 0
    for seg, primary, lset in zip(segments, baseline_dec, labels):
        dur = seg["end_ms"] - seg["start_ms"]
        if "A" in lset: speech_a += dur
        if "B" in lset: speech_b += dur
        if len(lset) > 1: overlap += dur
    total = speech_a + speech_b
    return {"speech_a": speech_a, "speech_b": speech_b, "overlap": overlap,
            "a_share": 100 * speech_a / total if total > 0 else 0}


def fixed_overlap(segments, baseline_dec, dual_t):
    """Fixed dual_t for all segments."""
    speech_a = speech_b = overlap = 0
    consec_dual = 0
    for seg, primary in zip(segments, baseline_dec):
        dur = seg["end_ms"] - seg["start_ms"]
        sim_a, sim_b = seg["sim_a"], seg["sim_b"]
        other = "B" if primary == "A" else "A"
        both = (sim_a is not None and sim_b is not None
                and sim_a >= dual_t and sim_b >= dual_t)
        is_dual = both and consec_dual < DUAL_CAP
        if primary == "A": speech_a += dur
        else: speech_b += dur
        if is_dual:
            consec_dual += 1
            if other == "A": speech_a += dur
            else: speech_b += dur
            overlap += dur
        else:
            if not both: consec_dual = 0
    total = speech_a + speech_b
    return {"speech_a": speech_a, "speech_b": speech_b, "overlap": overlap,
            "a_share": 100 * speech_a / total if total > 0 else 0}


def evaluate(segments, tiers, zones, snr_to_tier):
    correct = total = 0
    for seg, tier in zip(segments, tiers):
        if "expected_tier" not in seg: continue
        total += 1
        if tier == seg["expected_tier"]: correct += 1
    expected_tr = sum(1 for i in range(len(zones)-1)
                      if snr_to_tier[zones[i][2]] != snr_to_tier[zones[i+1][2]])
    actual_ch = sum(1 for i in range(1, len(tiers)) if tiers[i] != tiers[i-1])
    extra = max(0, actual_ch - expected_tr)
    return {"accuracy": 100.0*correct/total if total else 0,
            "correct": correct, "total": total,
            "actual_changes": actual_ch, "expected_changes": expected_tr,
            "extra": extra, "stability": 1.0/(1.0+extra)}


def sweep(all_data, default_tier, default_dual_t, snr_to_tier):
    """Sweep params, return best."""
    tier_dual_t = {default_tier: default_dual_t, "HIGH": 1.0}
    n_sc = len(all_data)
    best_score = -1
    best_p = None
    all_results = []

    combos = list(itertools.product(
        T_HIGH_RANGE, N_RANGE, MARGIN_RANGE, DWELL_RANGE, WARMUP_MULT))

    for t_high, n_val, margin, dwell, w_mult in combos:
        alpha = 2.0 / (n_val + 1)
        warmup = int(n_val * w_mult)
        params = {"alpha": alpha, "t_high": t_high, "margin": margin,
                  "min_dwell": dwell, "warmup": warmup}

        tot_correct = tot_segs = 0
        tot_stab = tot_lag = 0.0

        for sd in all_data:
            tiers, _ = simulate_2tier(sd["segments"], params, default_tier)
            ev = evaluate(sd["segments"], tiers, sd["zones"], snr_to_tier)
            tot_correct += ev["correct"]
            tot_segs += ev["total"]
            tot_stab += ev["stability"]

        acc = 100.0 * tot_correct / tot_segs if tot_segs else 0
        stab = tot_stab / n_sc
        score = acc * stab

        all_results.append({
            "t_high": t_high, "n": n_val, "margin": margin,
            "dwell": dwell, "warmup": warmup,
            "accuracy": acc, "stability": stab, "score": score})

        if score > best_score:
            best_score = score
            best_p = {"t_high": t_high, "n": n_val, "alpha": alpha,
                      "margin": margin, "min_dwell": dwell, "warmup": warmup}

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return best_p, all_results


def main():
    print("=" * 120)
    print("  2-TIER COMPARISON: LOW=0.82 vs LOW=0.84 as default")
    print("=" * 120)

    embedder = WeSpeakerEmbedder()
    clean_audio = {}
    for cn in [1, 2]:
        p = CLIPS.get(cn)
        if p and os.path.exists(p):
            clean_audio[cn] = load_audio(p)
            print(f"  Clip {cn}: {len(clean_audio[cn])/SAMPLE_RATE:.1f}s")

    # ── Run pipelines for all scenarios ──
    print(f"\n  Running pipelines...")
    all_data = []
    for sc_name in ["A", "B", "C", "D", "E"]:
        sc = SCENARIOS[sc_name]
        for cn in [1, 2]:
            if cn not in sc or cn not in clean_audio: continue
            zones = sc[cn]
            mixed = create_mixed_noise_clip(clean_audio[cn], zones)
            zone_desc = "→".join("Cl" if s is None else f"{s}" for _,_,s in zones)
            print(f"    {sc_name}/{cn} ({zone_desc})", end="", flush=True)
            t0 = time.time()
            segments = run_pipeline_segments(mixed, embedder)
            print(f" — {len(segments)} segs, {time.time()-t0:.1f}s")

            for seg in segments:
                mid_s = (seg["start_ms"] + seg["end_ms"]) / 2000.0
                for idx, (zs, ze, snr) in enumerate(zones):
                    if zs <= mid_s < ze:
                        seg["zone_snr"] = snr
                        seg["zone_idx"] = idx
                        break
            baseline_dec = apply_strategy(segments, {"type": "baseline"})
            all_data.append({"sc": sc_name, "clip": cn, "segments": segments,
                             "zones": zones, "baseline_dec": baseline_dec})

    # Also run clean clips (no noise) for baseline comparison
    print(f"\n  Running clean clips...")
    clean_data = {}
    for cn in [1, 2]:
        if cn not in clean_audio: continue
        print(f"    Clean/{cn}", end="", flush=True)
        t0 = time.time()
        segments = run_pipeline_segments(clean_audio[cn], embedder)
        print(f" — {len(segments)} segs, {time.time()-t0:.1f}s")
        baseline_dec = apply_strategy(segments, {"type": "baseline"})
        clean_data[cn] = {"segments": segments, "baseline_dec": baseline_dec}

    # ── Sweep for both defaults ──
    configs = [
        ("LOW=0.82", "LOW", 0.82,
         {None: "LOW", 35: "LOW", 30: "HIGH"}),
        ("LOW=0.84", "LOW84", 0.84,
         {None: "LOW84", 35: "LOW84", 30: "HIGH"}),
    ]

    best_params = {}
    for label, default_tier, dual_t, snr_to_tier in configs:
        print(f"\n{'='*120}")
        print(f"  SWEEP: {label} (default={dual_t}, HIGH=1.0)")
        print(f"{'='*120}")

        # Tag segments with tier expectations for this config
        for sd in all_data:
            for seg in sd["segments"]:
                seg["expected_tier"] = snr_to_tier[seg["zone_snr"]]

        bp, results = sweep(all_data, default_tier, dual_t, snr_to_tier)
        best_params[label] = (bp, default_tier, dual_t)

        print(f"\n  Top 10:")
        print(f"  {'T_high':>6s} {'N':>4s} {'Margin':>7s} {'Dwell':>6s} {'Warmup':>7s} "
              f"{'Accuracy':>9s} {'Stability':>10s} {'Score':>8s}")
        print(f"  {'─'*58}")
        for r in results[:10]:
            is_best = (r["t_high"] == bp["t_high"] and r["n"] == bp["n"]
                       and r["margin"] == bp["margin"]
                       and r["dwell"] == bp["min_dwell"]
                       and r["warmup"] == bp["warmup"])
            m = " ◀" if is_best else ""
            print(f"  {r['t_high']:>6.2f} {r['n']:>4d} {r['margin']:>7.3f} {r['dwell']:>6d} "
                  f"{r['warmup']:>7d} {r['accuracy']:>8.1f}% {r['stability']:>10.3f} "
                  f"{r['score']:>8.1f}{m}")

        print(f"\n  Best: t_high={bp['t_high']}, N={bp['n']}, "
              f"margin={bp['margin']}, dwell={bp['min_dwell']}, warmup={bp['warmup']}")

    # ── Side-by-side end-to-end ──
    print(f"\n{'='*120}")
    print(f"  END-TO-END COMPARISON")
    print(f"{'='*120}")

    # First: clean clips
    print(f"\n  ── CLEAN CLIPS (no noise — overlap accuracy vs ground truth) ──")
    for cn in [1, 2]:
        if cn not in clean_data: continue
        segs = clean_data[cn]["segments"]
        bdec = clean_data[cn]["baseline_dec"]
        gt = GT[cn]

        print(f"\n  Clip {cn} (GT: A={format_ms(gt['speech_a'])}, "
              f"B={format_ms(gt['speech_b'])}, Ov={format_ms(gt['overlap'])})")

        row_labels = ["GT", "Fixed 0.82", "Fixed 0.84", "No overlap"]
        row_data = [
            gt,
            fixed_overlap(segs, bdec, 0.82),
            fixed_overlap(segs, bdec, 0.84),
            fixed_overlap(segs, bdec, 1.0),
        ]
        # Add adaptive configs (on clean, they should stay in default tier)
        for label, (bp, default_tier, dual_t) in best_params.items():
            tier_dual_t = {default_tier: dual_t, "HIGH": 1.0}
            tiers, diag = simulate_2tier(segs, bp, default_tier)
            labs = apply_dual(segs, bdec, tiers, tier_dual_t)
            m = metrics_from_labels(segs, bdec, labs)
            n_changes = len(diag["tier_changes"])
            row_labels.append(f"Adapt {label} ({n_changes}ch)")
            row_data.append(m)

        cw = 14
        lw = 22
        sep = "  " + "─" * (lw + cw * 4)
        print(f"  {'':>{lw}s}{'Speech A':>{cw}s}{'Speech B':>{cw}s}{'Overlap':>{cw}s}{'A share':>{cw}s}")
        print(sep)
        for rl, rd in zip(row_labels, row_data):
            sa = rd.get("speech_a", rd.get("speech_a", 0))
            sb = rd.get("speech_b", rd.get("speech_b", 0))
            ov = rd.get("overlap", rd.get("overlap_time", 0))
            ash = rd.get("a_share", 0)
            if rl == "GT":
                total = gt["speech_a"] + gt["speech_b"]
                ash = 100 * gt["speech_a"] / total
            print(f"  {rl:>{lw}s}{format_ms(sa):>{cw}s}{format_ms(sb):>{cw}s}"
                  f"{format_ms(ov):>{cw}s}{ash:>{cw-1}.1f}%")
        print(sep)

    # Noisy scenarios
    print(f"\n  ── NOISY SCENARIOS (noise from start) ──")
    print(f"\n  {'Scenario':>12s} {'Clip':>5s}", end="")
    col_labels = ["Fixed 0.82", "Fixed 0.84", "No ovlp"]
    for label in best_params:
        col_labels.append(f"Ad {label}")
    for cl in col_labels:
        print(f" {cl:>12s}", end="")
    print(f"  (overlap in seconds)")
    print(f"  {'─' * (18 + 13 * len(col_labels))}")

    for sd in all_data:
        sc_name, cn = sd["sc"], sd["clip"]
        segs, zones, bdec = sd["segments"], sd["zones"], sd["baseline_dec"]
        desc = SCENARIOS[sc_name]["desc"]

        vals = []
        # Fixed
        vals.append(fixed_overlap(segs, bdec, 0.82)["overlap"] / 1000)
        vals.append(fixed_overlap(segs, bdec, 0.84)["overlap"] / 1000)
        vals.append(fixed_overlap(segs, bdec, 1.0)["overlap"] / 1000)

        # Adaptive
        for label, (bp, default_tier, dual_t) in best_params.items():
            tier_dual_t = {default_tier: dual_t, "HIGH": 1.0}
            tiers, diag = simulate_2tier(segs, bp, default_tier)
            labs = apply_dual(segs, bdec, tiers, tier_dual_t)
            m = metrics_from_labels(segs, bdec, labs)
            n_ch = len(diag["tier_changes"])
            vals.append((m["overlap"] / 1000, n_ch))

        print(f"  {desc:>12s} {cn:>5d}", end="")
        for v in vals:
            if isinstance(v, tuple):
                ov, ch = v
                print(f" {ov:>8.1f}({ch}ch)", end="")
            else:
                print(f" {v:>12.1f}", end="")
        print()

    print(f"  {'─' * (18 + 13 * len(col_labels))}")

    # Print best params summary
    print(f"\n  BEST PARAMETERS:")
    for label, (bp, default_tier, dual_t) in best_params.items():
        print(f"    {label}: t_high={bp['t_high']}, N={bp['n']}, "
              f"margin={bp['margin']}, dwell={bp['min_dwell']}, warmup={bp['warmup']}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
