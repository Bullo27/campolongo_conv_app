#!/usr/bin/env python3
"""2-tier adaptive noise detection: MODERATE (default) vs HIGH (disable overlap).

Simpler than 3-tier: just detect when noise is severe enough to disable overlap.
Uses the same EMA of min(sim_a,sim_b) but with a single threshold.

Reuses pipeline data from noise_adaptive_test.py scenarios.
"""
import os
import sys
import time
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS, FRAME_SIZE, SAMPLE_RATE, SPEECH_SEGMENT_FRAMES, MIN_FLUSH_FRAMES,
    PipelineSimulator, SpeakerIdentifier,
    WeSpeakerEmbedder, create_silero_vad, format_ms, load_audio,
)

from strategy_comparison import run_pipeline_segments, apply_strategy
from noise_robustness import add_gaussian_noise
from noise_adaptive_test import create_mixed_noise_clip, tag_segments_with_zones

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

# 2-tier: MODERATE (dual_t=0.84) or HIGH (dual_t=1.0, disabled)
TIER_DUAL_T = {"MODERATE": 0.84, "HIGH": 1.0}
DUAL_CAP = 3

# For the 2-tier system, 30dB expects HIGH, everything else expects MODERATE
SNR_TO_TIER_2 = {None: "MODERATE", 35: "MODERATE", 30: "HIGH"}

# Same scenarios as noise_adaptive_test.py
SCENARIOS = {
    "A": {
        "desc": "30dB → Clean",
        1: [(0, 100, 30), (100, 202, None)],
        2: [(0, 375, 30), (375, 751, None)],
    },
    "B": {
        "desc": "35dB → Clean",
        1: [(0, 100, 35), (100, 202, None)],
        2: [(0, 375, 35), (375, 751, None)],
    },
    "C": {
        "desc": "30dB → 35dB",
        1: [(0, 100, 30), (100, 202, 35)],
        2: [(0, 375, 30), (375, 751, 35)],
    },
    "D": {
        "desc": "35dB → 30dB",
        1: [(0, 100, 35), (100, 202, 30)],
        2: [(0, 375, 35), (375, 751, 30)],
    },
    "E": {
        "desc": "30dB → 35dB → Clean",
        2: [(0, 250, 30), (250, 500, 35), (500, 751, None)],
    },
}

# Sweep ranges
T_HIGH_RANGE = [0.73, 0.74, 0.75, 0.76, 0.77, 0.78]
N_RANGE = [10, 15, 20, 30]
MARGIN_RANGE = [0.005, 0.01, 0.015, 0.02, 0.025]
DWELL_RANGE = [5, 10, 15, 20, 30]
WARMUP_MULT = [0.5, 1.0, 2.0]


def simulate_2tier(segments, params):
    """2-tier adaptive: MODERATE (default) or HIGH (disable overlap)."""
    alpha = params["alpha"]
    t_high = params["t_high"]
    margin = params["margin"]
    min_dwell = params["min_dwell"]
    warmup = params["warmup"]

    tiers = []
    ema = 0.0
    current_tier = "MODERATE"
    segs_since_change = min_dwell
    b_established = False
    segs_since_b = 0
    ema_values = []
    tier_changes = []

    for i, seg in enumerate(segments):
        sim_b = seg.get("sim_b")
        sim_a = seg.get("sim_a")

        if not b_established:
            if sim_b is not None and sim_b > 0:
                b_established = True
                ema = min(sim_a, sim_b)
                segs_since_b = 0
            tiers.append(current_tier)
            ema_values.append(ema)
            continue

        min_sim = min(sim_a, sim_b)
        ema = alpha * min_sim + (1 - alpha) * ema
        ema_values.append(ema)
        segs_since_b += 1

        if segs_since_b < warmup:
            tiers.append(current_tier)
            continue

        segs_since_change += 1
        new_tier = current_tier

        if current_tier == "MODERATE":
            if ema > t_high + margin:
                new_tier = "HIGH"
        elif current_tier == "HIGH":
            if ema < t_high - margin:
                new_tier = "MODERATE"

        if new_tier != current_tier and segs_since_change >= min_dwell:
            tier_changes.append((i, current_tier, new_tier, ema))
            current_tier = new_tier
            segs_since_change = 0

        tiers.append(current_tier)

    return tiers, {"ema_values": ema_values, "tier_changes": tier_changes}


def evaluate_2tier(segments, tier_assignments, zones):
    """Evaluate 2-tier accuracy."""
    correct = total = 0
    for seg, tier in zip(segments, tier_assignments):
        if "expected_tier" not in seg:
            continue
        total += 1
        if tier == seg["expected_tier"]:
            correct += 1

    expected_transitions = sum(
        1 for i in range(len(zones) - 1)
        if SNR_TO_TIER_2[zones[i][2]] != SNR_TO_TIER_2[zones[i + 1][2]]
    )

    actual_changes = sum(
        1 for i in range(1, len(tier_assignments))
        if tier_assignments[i] != tier_assignments[i - 1]
    )
    extra = max(0, actual_changes - expected_transitions)

    # Transition lag
    lags = []
    for z_idx in range(len(zones) - 1):
        t1 = SNR_TO_TIER_2[zones[z_idx][2]]
        t2 = SNR_TO_TIER_2[zones[z_idx + 1][2]]
        if t1 == t2:
            continue  # no transition expected
        boundary_ms = zones[z_idx + 1][0] * 1000
        found = False
        for i, (seg, tier) in enumerate(zip(segments, tier_assignments)):
            if seg["start_ms"] >= boundary_ms:
                lag = 0
                for j in range(i, len(segments)):
                    if tier_assignments[j] == t2:
                        lags.append(lag)
                        found = True
                        break
                    lag += 1
                if not found:
                    zone_end_ms = zones[z_idx + 1][1] * 1000
                    remaining = sum(1 for s in segments
                                    if boundary_ms <= s["start_ms"] < zone_end_ms)
                    lags.append(remaining)
                break

    return {
        "accuracy": 100.0 * correct / total if total > 0 else 0,
        "correct": correct, "total": total,
        "actual_changes": actual_changes,
        "expected_changes": expected_transitions,
        "extra_switches": extra,
        "stability": 1.0 / (1.0 + extra),
        "avg_lag": float(np.mean(lags)) if lags else 0,
        "max_lag": max(lags) if lags else 0,
    }


def apply_dual(segments, baseline_dec, tier_assignments):
    """Apply dual-assignment with per-segment tier."""
    labels = []
    consec_dual = 0
    for seg, primary, tier in zip(segments, baseline_dec, tier_assignments):
        dual_t = TIER_DUAL_T[tier]
        sim_a, sim_b = seg["sim_a"], seg["sim_b"]
        other = "B" if primary == "A" else "A"
        both = (sim_a is not None and sim_b is not None
                and sim_a >= dual_t and sim_b >= dual_t)
        is_dual = both and consec_dual < DUAL_CAP
        if is_dual:
            consec_dual += 1
            labels.append({primary, other})
        else:
            if not both:
                consec_dual = 0
            labels.append({primary})
    return labels


def metrics_from_labels(segments, baseline_dec, labels):
    """Compute overlap, speech times from label sets."""
    speech_a = speech_b = overlap = 0
    for seg, primary, lset in zip(segments, baseline_dec, labels):
        dur = seg["end_ms"] - seg["start_ms"]
        if "A" in lset: speech_a += dur
        if "B" in lset: speech_b += dur
        if len(lset) > 1: overlap += dur
    total = speech_a + speech_b
    return {
        "speech_a": speech_a, "speech_b": speech_b, "overlap": overlap,
        "a_share": 100 * speech_a / total if total > 0 else 0,
    }


def main():
    print("=" * 110)
    print("  2-TIER ADAPTIVE NOISE DETECTION")
    print("  MODERATE (0.84, default) vs HIGH (1.0, overlap disabled)")
    print("=" * 110)

    embedder = WeSpeakerEmbedder()
    clean_audio = {}
    for cn in [1, 2]:
        p = CLIPS.get(cn)
        if p and os.path.exists(p):
            clean_audio[cn] = load_audio(p)
            print(f"  Clip {cn}: {len(clean_audio[cn]) / SAMPLE_RATE:.1f}s")

    # ── Run all scenarios ──
    print(f"\n{'=' * 110}")
    print(f"  RUNNING PIPELINES")
    print(f"{'=' * 110}")

    all_data = []
    for sc_name in ["A", "B", "C", "D", "E"]:
        sc = SCENARIOS[sc_name]
        for cn in [1, 2]:
            if cn not in sc or cn not in clean_audio:
                continue
            zones = sc[cn]
            mixed = create_mixed_noise_clip(clean_audio[cn], zones)

            zone_desc = " → ".join("Clean" if s is None else f"{s}dB" for _, _, s in zones)
            print(f"\n  {sc_name}/{cn}: {zone_desc}", end="", flush=True)

            t0 = time.time()
            segments = run_pipeline_segments(mixed, embedder)
            print(f" — {len(segments)} segs, {time.time()-t0:.1f}s")

            # Tag with 2-tier expectations
            for seg in segments:
                mid_s = (seg["start_ms"] + seg["end_ms"]) / 2000.0
                for idx, (zs, ze, snr) in enumerate(zones):
                    if zs <= mid_s < ze:
                        seg["zone_snr"] = snr
                        seg["expected_tier"] = SNR_TO_TIER_2[snr]
                        seg["zone_idx"] = idx
                        break

            baseline_dec = apply_strategy(segments, {"type": "baseline"})

            # Print distributions
            for z_idx, (zs, ze, snr) in enumerate(zones):
                mins = [min(s["sim_a"], s["sim_b"]) for s in segments
                        if s["zone_idx"] == z_idx and s.get("sim_b") and s["sim_b"] > 0]
                if mins:
                    arr = np.array(mins)
                    label = "Clean" if snr is None else f"{snr}dB"
                    print(f"    Z{z_idx}({label}): n={len(mins)}, "
                          f"mean={arr.mean():.4f}, std={arr.std():.4f}")

            all_data.append({
                "sc": sc_name, "clip": cn, "segments": segments,
                "zones": zones, "baseline_dec": baseline_dec,
            })

    n = len(all_data)

    # ── Sweep ──
    print(f"\n{'=' * 110}")
    print(f"  PARAMETER SWEEP ({n} scenarios)")
    print(f"{'=' * 110}")

    best_score = -1
    best_params = None
    results = []

    combos = list(itertools.product(
        T_HIGH_RANGE, N_RANGE, MARGIN_RANGE, DWELL_RANGE, WARMUP_MULT))
    print(f"  {len(combos)} combinations...")

    for t_high, n_val, margin, dwell, w_mult in combos:
        alpha = 2.0 / (n_val + 1)
        warmup = int(n_val * w_mult)
        params = {"alpha": alpha, "t_high": t_high, "margin": margin,
                  "min_dwell": dwell, "warmup": warmup}

        tot_correct = tot_segs = 0
        tot_stability = tot_lag = 0.0

        for sd in all_data:
            tiers, _ = simulate_2tier(sd["segments"], params)
            ev = evaluate_2tier(sd["segments"], tiers, sd["zones"])
            tot_correct += ev["correct"]
            tot_segs += ev["total"]
            tot_stability += ev["stability"]
            tot_lag += ev["avg_lag"]

        acc = 100.0 * tot_correct / tot_segs if tot_segs > 0 else 0
        stab = tot_stability / n
        lag = tot_lag / n
        lag_pen = 1.0 / (1.0 + lag / 10.0)
        score = acc * stab * lag_pen

        results.append({
            "t_high": t_high, "n": n_val, "margin": margin,
            "dwell": dwell, "warmup": warmup,
            "accuracy": acc, "stability": stab, "avg_lag": lag, "score": score,
        })
        if score > best_score:
            best_score = score
            best_params = {
                "t_high": t_high, "n": n_val, "alpha": alpha,
                "margin": margin, "min_dwell": dwell, "warmup": warmup,
            }

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n  Top 20 configs:")
    print(f"  {'T_high':>6s} {'N':>4s} {'Margin':>7s} {'Dwell':>6s} {'Warmup':>7s} "
          f"{'Accuracy':>9s} {'Stability':>10s} {'AvgLag':>7s} {'Score':>8s}")
    print(f"  {'─' * 68}")
    for r in results[:20]:
        is_best = (r["t_high"] == best_params["t_high"]
                   and r["n"] == best_params["n"]
                   and r["margin"] == best_params["margin"]
                   and r["dwell"] == best_params["min_dwell"]
                   and r["warmup"] == best_params["warmup"])
        m = " ◀" if is_best else ""
        print(f"  {r['t_high']:>6.2f} {r['n']:>4d} {r['margin']:>7.3f} {r['dwell']:>6d} "
              f"{r['warmup']:>7d} {r['accuracy']:>8.1f}% {r['stability']:>10.3f} "
              f"{r['avg_lag']:>7.1f} {r['score']:>8.1f}{m}")

    print(f"\n  Best: t_high={best_params['t_high']}, N={best_params['n']}, "
          f"margin={best_params['margin']}, dwell={best_params['min_dwell']}, "
          f"warmup={best_params['warmup']}")

    # ── End-to-end ──
    print(f"\n{'=' * 110}")
    print(f"  END-TO-END VALIDATION")
    print(f"{'=' * 110}")

    for sd in all_data:
        sc_name, cn = sd["sc"], sd["clip"]
        segments, zones, baseline_dec = sd["segments"], sd["zones"], sd["baseline_dec"]

        zone_desc = " → ".join("Clean" if s is None else f"{s}dB" for _, _, s in zones)
        print(f"\n  ── {sc_name}/{cn}: {zone_desc} ──")

        # Adaptive 2-tier
        adapt_tiers, diag = simulate_2tier(segments, best_params)
        adapt_ev = evaluate_2tier(segments, adapt_tiers, zones)
        adapt_labels = apply_dual(segments, baseline_dec, adapt_tiers)
        adapt_m = metrics_from_labels(segments, baseline_dec, adapt_labels)

        # Oracle 2-tier
        oracle_tiers = [seg["expected_tier"] for seg in segments]
        oracle_labels = apply_dual(segments, baseline_dec, oracle_tiers)
        oracle_m = metrics_from_labels(segments, baseline_dec, oracle_labels)

        # Fixed configs
        f82_labels = apply_dual(segments, baseline_dec, ["MODERATE"] * len(segments))
        # Wait, fixed 0.82 means LOW tier... but we only have MODERATE and HIGH in 2-tier
        # Let me add a "Fixed 0.82" by manually using dual_t=0.82
        f82_m = compute_fixed_overlap(segments, baseline_dec, 0.82)
        f84_m = metrics_from_labels(segments, baseline_dec,
                                     apply_dual(segments, baseline_dec, ["MODERATE"] * len(segments)))
        f10_m = metrics_from_labels(segments, baseline_dec,
                                     apply_dual(segments, baseline_dec, ["HIGH"] * len(segments)))

        configs = [
            ("Adaptive", adapt_m),
            ("Oracle-2T", oracle_m),
            ("Fixed 0.82", f82_m),
            ("Fixed 0.84", f84_m),
            ("No overlap", f10_m),
        ]
        cw = 12
        lw = 14
        sep = "  " + "─" * (lw + cw * len(configs))
        print(sep)
        print(f"  {'':>{lw}s}" + "".join(f"{c[0]:>{cw}s}" for c in configs))
        print(sep)
        print(f"  {'Speech A':>{lw}s}" + "".join(f"{format_ms(c[1]['speech_a']):>{cw}s}" for c in configs))
        print(f"  {'Speech B':>{lw}s}" + "".join(f"{format_ms(c[1]['speech_b']):>{cw}s}" for c in configs))
        print(f"  {'Overlap':>{lw}s}" + "".join(f"{format_ms(c[1]['overlap']):>{cw}s}" for c in configs))
        print(f"  {'A share':>{lw}s}" + "".join(f"{c[1]['a_share']:>{cw-1}.1f}%" for c in configs))
        print(sep)

        print(f"  Accuracy: {adapt_ev['accuracy']:.1f}%  "
              f"Changes: {adapt_ev['actual_changes']} "
              f"(exp {adapt_ev['expected_changes']}, +{adapt_ev['extra_switches']})")

        if diag["tier_changes"]:
            for si, old, new, ema_val in diag["tier_changes"]:
                seg = segments[si]
                print(f"    {format_ms(seg['start_ms'])}: {old}→{new} (EMA={ema_val:.4f})")

    # ── Summary ──
    print(f"\n{'=' * 110}")
    print(f"  FINAL 2-TIER PARAMETERS")
    print(f"{'=' * 110}")
    print(f"  T_high = {best_params['t_high']}  (MODERATE ↔ HIGH threshold)")
    print(f"  N      = {best_params['n']}  (EMA window, alpha={best_params['alpha']:.4f})")
    print(f"  Margin = {best_params['margin']}  (hysteresis)")
    print(f"  Dwell  = {best_params['min_dwell']}  (min segments between changes)")
    print(f"  Warmup = {best_params['warmup']}  (segments after B)")
    print(f"  Default = MODERATE (dual_t=0.84)")
    print(f"  HIGH   = overlap disabled (dual_t=1.0)")
    print(f"\nDone.")


def compute_fixed_overlap(segments, baseline_dec, dual_t):
    """Compute metrics with a fixed dual_t (for comparison)."""
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
    return {
        "speech_a": speech_a, "speech_b": speech_b, "overlap": overlap,
        "a_share": 100 * speech_a / total if total > 0 else 0,
    }


if __name__ == "__main__":
    main()
