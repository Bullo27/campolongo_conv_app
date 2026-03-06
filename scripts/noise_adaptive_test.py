#!/usr/bin/env python3
"""Adaptive noise detection: calibrate EMA-based tier selection.

REVISED: Uses noise-from-start scenarios where references are established
from noisy audio (the dangerous case where adaptive detection matters).

Scenarios:
  A: 30dB → Clean    (severe noise clearing)
  B: 35dB → Clean    (moderate noise clearing)
  C: 30dB → 35dB     (noise improvement, still noisy)
  D: 35dB → 30dB     (noise worsening)
  E: 30dB → 35dB → Clean  (multi-zone, clip 2 only)

Steps:
  1. Create noisy-start clips, run pipeline for each scenario
  2. Report per-zone min(sim_a,sim_b) distributions
  3. Sweep T_mod × T_high thresholds across all scenarios
  4. Sweep EMA params (N, margin, dwell, warmup) with best thresholds
  5. End-to-end: adaptive vs oracle vs fixed on overlap/speech metrics
"""
import os
import sys
import time
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_audio import (
    FRAME_MS, FRAME_SIZE, SAMPLE_RATE, SPEECH_SEGMENT_FRAMES, MIN_FLUSH_FRAMES,
    PipelineSimulator, SpeakerIdentifier, SpeechBrainDiarizer,
    WeSpeakerEmbedder, compare_timelines, create_silero_vad,
    extract_vad_segments, format_ms, load_audio,
)

from strategy_comparison import (
    run_pipeline_segments, apply_strategy, build_timeline, compute_metrics,
    compute_dual_metrics, relaxed_agreement,
)

from noise_robustness import add_gaussian_noise

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

# Tier mappings
SNR_TO_TIER = {
    None: "LOW",     # Clean → LOW (dual_t=0.82)
    35: "MODERATE",  # 35 dB → MODERATE (dual_t=0.84)
    30: "HIGH",      # 30 dB → HIGH (dual_t=1.0, overlap disabled)
}

TIER_DUAL_T = {
    "LOW": 0.82,
    "MODERATE": 0.84,
    "HIGH": 1.0,
}

DUAL_CAP = 3

# Scenarios: each is a dict with clip-specific zones
# Clip 1 (202s): split at 100s
# Clip 2 (751s): split at 375s (or 250s/500s for multi-zone)
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
        "desc": "30dB → 35dB → Clean (clip 2 only)",
        2: [(0, 250, 30), (250, 500, 35), (500, 751, None)],
    },
}

# Sweep ranges
T_MOD_RANGE = [0.70, 0.71, 0.72, 0.73, 0.74]
T_HIGH_RANGE = [0.73, 0.74, 0.75, 0.76, 0.77]
N_RANGE = [10, 15, 20, 30]
MARGIN_RANGE = [0.005, 0.01, 0.015, 0.02]
DWELL_RANGE = [5, 10, 15, 20]
WARMUP_MULT = [0.5, 1.0, 2.0]


# ── Noise creation ──

def create_mixed_noise_clip(audio_int16, zones, seed=42):
    """Apply different noise levels to different time ranges."""
    result = audio_int16.copy()
    rng = np.random.default_rng(seed)
    audio_f64 = audio_int16.astype(np.float64)
    signal_power = np.mean(audio_f64 ** 2)

    for start_s, end_s, snr in zones:
        if snr is None:
            continue
        start_idx = int(start_s * SAMPLE_RATE)
        end_idx = min(int(end_s * SAMPLE_RATE), len(audio_int16))
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        noise = rng.normal(0, np.sqrt(noise_power), end_idx - start_idx)
        result[start_idx:end_idx] = np.clip(
            audio_f64[start_idx:end_idx] + noise, -32768, 32767
        ).astype(np.int16)

    return result


def tag_segments_with_zones(segments, zones):
    """Add zone_snr, expected_tier, zone_idx to each segment."""
    for seg in segments:
        mid_s = (seg["start_ms"] + seg["end_ms"]) / 2000.0
        for idx, (zs, ze, snr) in enumerate(zones):
            if zs <= mid_s < ze:
                seg["zone_snr"] = snr
                seg["expected_tier"] = SNR_TO_TIER[snr]
                seg["zone_idx"] = idx
                break
        else:
            seg["zone_snr"] = zones[-1][2]
            seg["expected_tier"] = SNR_TO_TIER[zones[-1][2]]
            seg["zone_idx"] = len(zones) - 1


# ── Adaptive simulation ──

def simulate_adaptive(segments, params):
    """Simulate EMA-based adaptive noise tier detector.

    Returns (tier_list, diagnostics_dict).
    """
    alpha = params["alpha"]
    t_mod = params["t_mod"]
    t_high = params["t_high"]
    margin = params["margin"]
    min_dwell = params["min_dwell"]
    warmup = params["warmup"]

    tiers = []
    ema = 0.0
    current_tier = "MODERATE"
    segments_since_change = min_dwell
    b_established = False
    segments_since_b = 0
    ema_values = []
    tier_changes = []

    for i, seg in enumerate(segments):
        sim_b = seg.get("sim_b")
        sim_a = seg.get("sim_a")

        if not b_established:
            if sim_b is not None and sim_b > 0:
                b_established = True
                ema = min(sim_a, sim_b)
                segments_since_b = 0
            tiers.append(current_tier)
            ema_values.append(ema)
            continue

        min_sim = min(sim_a, sim_b)
        ema = alpha * min_sim + (1 - alpha) * ema
        ema_values.append(ema)
        segments_since_b += 1

        if segments_since_b < warmup:
            tiers.append(current_tier)
            continue

        segments_since_change += 1
        new_tier = current_tier

        if current_tier == "LOW":
            if ema > t_high + margin:
                new_tier = "HIGH"
            elif ema > t_mod + margin:
                new_tier = "MODERATE"
        elif current_tier == "MODERATE":
            if ema > t_high + margin:
                new_tier = "HIGH"
            elif ema < t_mod - margin:
                new_tier = "LOW"
        elif current_tier == "HIGH":
            if ema < t_mod - margin:
                new_tier = "LOW"
            elif ema < t_high - margin:
                new_tier = "MODERATE"

        if new_tier != current_tier and segments_since_change >= min_dwell:
            tier_changes.append((i, current_tier, new_tier, ema))
            current_tier = new_tier
            segments_since_change = 0

        tiers.append(current_tier)

    return tiers, {
        "ema_values": ema_values,
        "tier_changes": tier_changes,
        "b_established_at": next(
            (i for i, s in enumerate(segments)
             if s.get("sim_b") is not None and s["sim_b"] > 0), None),
    }


def evaluate_adaptive(segments, tier_assignments, zones):
    """Compute zone accuracy, transition lag, oscillation count."""
    expected_transitions = len(zones) - 1

    correct = total = 0
    for seg, tier in zip(segments, tier_assignments):
        if "expected_tier" not in seg:
            continue
        total += 1
        if tier == seg["expected_tier"]:
            correct += 1

    zone_accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Transition lag per zone boundary
    transition_lags = []
    for z_idx in range(len(zones) - 1):
        target_tier = SNR_TO_TIER[zones[z_idx + 1][2]]
        boundary_ms = zones[z_idx + 1][0] * 1000

        found = False
        for i, (seg, tier) in enumerate(zip(segments, tier_assignments)):
            if seg["start_ms"] >= boundary_ms:
                lag = 0
                for j in range(i, len(segments)):
                    if tier_assignments[j] == target_tier:
                        transition_lags.append(lag)
                        found = True
                        break
                    lag += 1
                if not found:
                    zone_end_ms = zones[z_idx + 1][1] * 1000
                    remaining = sum(
                        1 for s in segments
                        if boundary_ms <= s["start_ms"] < zone_end_ms
                    )
                    transition_lags.append(remaining)
                break

    actual_changes = sum(
        1 for i in range(1, len(tier_assignments))
        if tier_assignments[i] != tier_assignments[i - 1]
    )
    extra_switches = max(0, actual_changes - expected_transitions)

    return {
        "zone_accuracy": zone_accuracy,
        "avg_lag": float(np.mean(transition_lags)) if transition_lags else 0,
        "max_lag": max(transition_lags) if transition_lags else 0,
        "actual_changes": actual_changes,
        "expected_changes": expected_transitions,
        "extra_switches": extra_switches,
        "stability_score": 1.0 / (1.0 + extra_switches),
        "correct": correct,
        "total": total,
    }


# ── Dual-assignment ──

def apply_adaptive_dual(segments, baseline_decisions, tier_assignments):
    """Apply per-segment dual_t based on tier. Returns list of label sets."""
    segment_labels = []
    consec_dual = 0

    for seg, primary, tier in zip(segments, baseline_decisions, tier_assignments):
        dual_t = TIER_DUAL_T[tier]
        sim_a = seg["sim_a"]
        sim_b = seg["sim_b"]
        other = "B" if primary == "A" else "A"

        both_above = (
            sim_a is not None and sim_b is not None
            and sim_a >= dual_t and sim_b >= dual_t
        )
        is_dual = both_above and consec_dual < DUAL_CAP

        if is_dual:
            consec_dual += 1
            segment_labels.append({primary, other})
        else:
            if not both_above:
                consec_dual = 0
            segment_labels.append({primary})

    return segment_labels


def compute_dual_metrics_from_labels(segments, baseline_decisions, segment_labels):
    """Compute speech_a, speech_b, overlap from label sets."""
    speech_a = speech_b = overlap = 0
    changes = 0
    prev_primary = None

    for seg, primary, labels in zip(segments, baseline_decisions, segment_labels):
        dur = seg["end_ms"] - seg["start_ms"]
        if "A" in labels:
            speech_a += dur
        if "B" in labels:
            speech_b += dur
        if len(labels) > 1:
            overlap += dur
        if prev_primary is not None and primary != prev_primary:
            changes += 1
        prev_primary = primary

    total = speech_a + speech_b
    return {
        "speech_a": speech_a,
        "speech_b": speech_b,
        "overlap": overlap,
        "a_share": 100 * speech_a / total if total > 0 else 0,
        "changes": changes,
    }


# ── Main ──

def main():
    print("=" * 110)
    print("  ADAPTIVE NOISE DETECTION — Noise-From-Start Calibration")
    print("  All scenarios start noisy (refs established from noisy audio)")
    print("=" * 110)

    embedder = WeSpeakerEmbedder()

    # Load clean audio once per clip
    clean_audio = {}
    for clip_num in [1, 2]:
        path = CLIPS.get(clip_num)
        if path and os.path.exists(path):
            clean_audio[clip_num] = load_audio(path)
            dur = len(clean_audio[clip_num]) / SAMPLE_RATE
            print(f"  Clip {clip_num}: {dur:.1f}s loaded")

    # ── Step 1 & 2: Run pipeline for each scenario ──
    print(f"\n{'=' * 110}")
    print(f"  Steps 1-2: CREATE CLIPS & RUN PIPELINE")
    print(f"{'=' * 110}")

    all_scenario_data = []  # list of (scenario_name, clip_num, segments, zones, baseline_dec)

    for sc_name in ["A", "B", "C", "D", "E"]:
        sc = SCENARIOS[sc_name]
        desc = sc["desc"]

        for clip_num in [1, 2]:
            if clip_num not in sc or clip_num not in clean_audio:
                continue

            zones = sc[clip_num]
            audio = clean_audio[clip_num]
            duration_s = len(audio) / SAMPLE_RATE

            zone_desc = " → ".join(
                "Clean" if snr is None else f"{snr}dB"
                for _, _, snr in zones
            )
            print(f"\n  Scenario {sc_name} / Clip {clip_num}: {zone_desc}")

            mixed = create_mixed_noise_clip(audio, zones)

            t0 = time.time()
            segments = run_pipeline_segments(mixed, embedder)
            elapsed = time.time() - t0
            print(f"    {len(segments)} segments in {elapsed:.1f}s")

            tag_segments_with_zones(segments, zones)
            baseline_dec = apply_strategy(segments, {"type": "baseline"})

            # Per-zone min(sim_a,sim_b) distributions
            for z_idx, (zs, ze, snr) in enumerate(zones):
                zone_mins = []
                for seg in segments:
                    if (seg["zone_idx"] == z_idx
                            and seg.get("sim_b") is not None
                            and seg["sim_b"] > 0):
                        zone_mins.append(min(seg["sim_a"], seg["sim_b"]))
                if zone_mins:
                    arr = np.array(zone_mins)
                    label = "Clean" if snr is None else f"{snr}dB"
                    print(f"    Zone {z_idx} ({label}, {zs}-{ze}s): "
                          f"n={len(zone_mins)}, mean={arr.mean():.4f}, "
                          f"std={arr.std():.4f}, "
                          f"p25={np.percentile(arr, 25):.4f}, "
                          f"p75={np.percentile(arr, 75):.4f}")

            all_scenario_data.append({
                "scenario": sc_name,
                "clip": clip_num,
                "segments": segments,
                "zones": zones,
                "baseline_dec": baseline_dec,
            })

    n_scenarios = len(all_scenario_data)
    print(f"\n  Total: {n_scenarios} scenario/clip combinations")

    # ── Step 3: Threshold sweep ──
    print(f"\n{'=' * 110}")
    print(f"  Step 3: THRESHOLD CALIBRATION SWEEP")
    print(f"  Across all {n_scenarios} scenarios")
    print(f"{'=' * 110}")

    fixed_n = 20
    fixed_alpha = 2.0 / (fixed_n + 1)
    fixed_margin = 0.01
    fixed_dwell = 10
    fixed_warmup = fixed_n

    best_score = -1
    best_thresholds = None
    threshold_results = []

    for t_mod in T_MOD_RANGE:
        for t_high in T_HIGH_RANGE:
            if t_high <= t_mod + 0.01:
                continue

            params = {
                "alpha": fixed_alpha, "t_mod": t_mod, "t_high": t_high,
                "margin": fixed_margin, "min_dwell": fixed_dwell,
                "warmup": fixed_warmup,
            }

            total_correct = total_segs = 0
            total_stability = 0.0

            for sd in all_scenario_data:
                tiers, _ = simulate_adaptive(sd["segments"], params)
                ev = evaluate_adaptive(sd["segments"], tiers, sd["zones"])
                total_correct += ev["correct"]
                total_segs += ev["total"]
                total_stability += ev["stability_score"]

            accuracy = 100.0 * total_correct / total_segs if total_segs > 0 else 0
            avg_stability = total_stability / n_scenarios
            score = accuracy * avg_stability

            threshold_results.append({
                "t_mod": t_mod, "t_high": t_high,
                "accuracy": accuracy, "stability": avg_stability, "score": score,
            })
            if score > best_score:
                best_score = score
                best_thresholds = (t_mod, t_high)

    threshold_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n  Top 10 threshold combinations:")
    print(f"  {'T_mod':>6s} {'T_high':>7s} {'Accuracy':>9s} {'Stability':>10s} {'Score':>8s}")
    print(f"  {'─' * 42}")
    for r in threshold_results[:10]:
        m = " ◀" if (r["t_mod"], r["t_high"]) == best_thresholds else ""
        print(f"  {r['t_mod']:>6.2f} {r['t_high']:>7.2f} "
              f"{r['accuracy']:>8.1f}% {r['stability']:>10.3f} {r['score']:>8.1f}{m}")

    best_t_mod, best_t_high = best_thresholds
    print(f"\n  Best: T_mod={best_t_mod}, T_high={best_t_high}")

    # ── Step 4: EMA param sweep ──
    print(f"\n{'=' * 110}")
    print(f"  Step 4: EMA PARAMETER SWEEP")
    print(f"  T_mod={best_t_mod}, T_high={best_t_high}")
    print(f"{'=' * 110}")

    best_ema_score = -1
    best_ema = None
    ema_results = []
    combos = list(itertools.product(N_RANGE, MARGIN_RANGE, DWELL_RANGE, WARMUP_MULT))
    print(f"  {len(combos)} combinations...")

    for n_val, margin, dwell, w_mult in combos:
        alpha = 2.0 / (n_val + 1)
        warmup = int(n_val * w_mult)

        params = {
            "alpha": alpha, "t_mod": best_t_mod, "t_high": best_t_high,
            "margin": margin, "min_dwell": dwell, "warmup": warmup,
        }

        total_correct = total_segs = 0
        total_stability = total_lag = 0.0

        for sd in all_scenario_data:
            tiers, _ = simulate_adaptive(sd["segments"], params)
            ev = evaluate_adaptive(sd["segments"], tiers, sd["zones"])
            total_correct += ev["correct"]
            total_segs += ev["total"]
            total_stability += ev["stability_score"]
            total_lag += ev["avg_lag"]

        accuracy = 100.0 * total_correct / total_segs if total_segs > 0 else 0
        avg_stability = total_stability / n_scenarios
        avg_lag = total_lag / n_scenarios
        lag_penalty = 1.0 / (1.0 + avg_lag / 10.0)
        score = accuracy * avg_stability * lag_penalty

        ema_results.append({
            "n": n_val, "margin": margin, "dwell": dwell,
            "warmup": warmup, "w_mult": w_mult,
            "accuracy": accuracy, "stability": avg_stability,
            "avg_lag": avg_lag, "score": score,
        })
        if score > best_ema_score:
            best_ema_score = score
            best_ema = {"n": n_val, "alpha": alpha, "margin": margin,
                        "dwell": dwell, "warmup": warmup}

    ema_results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n  Top 15 EMA configs:")
    print(f"  {'N':>4s} {'Margin':>7s} {'Dwell':>6s} {'Warmup':>7s} "
          f"{'Accuracy':>9s} {'Stability':>10s} {'AvgLag':>7s} {'Score':>8s}")
    print(f"  {'─' * 60}")
    for r in ema_results[:15]:
        is_best = (r["n"] == best_ema["n"] and r["margin"] == best_ema["margin"]
                   and r["dwell"] == best_ema["dwell"]
                   and r["warmup"] == best_ema["warmup"])
        m = " ◀" if is_best else ""
        print(f"  {r['n']:>4d} {r['margin']:>7.3f} {r['dwell']:>6d} {r['warmup']:>7d} "
              f"{r['accuracy']:>8.1f}% {r['stability']:>10.3f} "
              f"{r['avg_lag']:>7.1f} {r['score']:>8.1f}{m}")

    print(f"\n  Best EMA: N={best_ema['n']}, alpha={best_ema['alpha']:.4f}, "
          f"margin={best_ema['margin']}, dwell={best_ema['dwell']}, "
          f"warmup={best_ema['warmup']}")

    # ── Step 5: End-to-end validation ──
    print(f"\n{'=' * 110}")
    print(f"  Step 5: END-TO-END VALIDATION")
    print(f"  Adaptive: T_mod={best_t_mod}, T_high={best_t_high}, "
          f"N={best_ema['n']}, margin={best_ema['margin']}, "
          f"dwell={best_ema['dwell']}, warmup={best_ema['warmup']}")
    print(f"{'=' * 110}")

    final_params = {
        "alpha": best_ema["alpha"], "t_mod": best_t_mod, "t_high": best_t_high,
        "margin": best_ema["margin"], "min_dwell": best_ema["dwell"],
        "warmup": best_ema["warmup"],
    }

    for sd in all_scenario_data:
        sc_name = sd["scenario"]
        clip_num = sd["clip"]
        segments = sd["segments"]
        zones = sd["zones"]
        baseline_dec = sd["baseline_dec"]

        zone_desc = " → ".join(
            "Clean" if snr is None else f"{snr}dB" for _, _, snr in zones)
        print(f"\n  ── Scenario {sc_name} / Clip {clip_num}: {zone_desc} ──")

        # Adaptive
        adaptive_tiers, diag = simulate_adaptive(segments, final_params)
        adaptive_ev = evaluate_adaptive(segments, adaptive_tiers, zones)
        adaptive_labels = apply_adaptive_dual(segments, baseline_dec, adaptive_tiers)
        adaptive_m = compute_dual_metrics_from_labels(segments, baseline_dec, adaptive_labels)

        # Oracle
        oracle_tiers = [seg["expected_tier"] for seg in segments]
        oracle_labels = apply_adaptive_dual(segments, baseline_dec, oracle_tiers)
        oracle_m = compute_dual_metrics_from_labels(segments, baseline_dec, oracle_labels)

        # Fixed 0.82 (LOW everywhere)
        f82_labels = apply_adaptive_dual(segments, baseline_dec, ["LOW"] * len(segments))
        f82_m = compute_dual_metrics_from_labels(segments, baseline_dec, f82_labels)

        # Fixed 0.84 (MODERATE everywhere)
        f84_labels = apply_adaptive_dual(segments, baseline_dec, ["MODERATE"] * len(segments))
        f84_m = compute_dual_metrics_from_labels(segments, baseline_dec, f84_labels)

        # Table
        configs = [
            ("Adaptive", adaptive_m),
            ("Oracle", oracle_m),
            ("Fixed 0.82", f82_m),
            ("Fixed 0.84", f84_m),
        ]
        cw = 13
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

        # Diagnostics
        print(f"  Zone accuracy: {adaptive_ev['zone_accuracy']:.1f}%  "
              f"Lag: {adaptive_ev['avg_lag']:.1f} avg / {adaptive_ev['max_lag']} max  "
              f"Changes: {adaptive_ev['actual_changes']} "
              f"(exp {adaptive_ev['expected_changes']}, "
              f"+{adaptive_ev['extra_switches']} extra)")

        if diag["tier_changes"]:
            for seg_idx, old, new, ema_val in diag["tier_changes"]:
                seg = segments[seg_idx]
                print(f"    {format_ms(seg['start_ms'])}: {old} → {new} (EMA={ema_val:.4f})")

        # Per-zone overlap comparison
        for z_idx, (zs, ze, snr) in enumerate(zones):
            z_segs = [(seg, dec, tier) for seg, dec, tier
                      in zip(segments, baseline_dec, adaptive_tiers)
                      if seg["zone_idx"] == z_idx]
            if not z_segs:
                continue
            z_s, z_d, z_t = zip(*z_segs)
            a_lab = apply_adaptive_dual(z_s, z_d, z_t)
            a_met = compute_dual_metrics_from_labels(z_s, z_d, a_lab)
            o_lab = apply_adaptive_dual(z_s, z_d, [seg["expected_tier"] for seg in z_s])
            o_met = compute_dual_metrics_from_labels(z_s, z_d, o_lab)
            snr_label = "Clean" if snr is None else f"{snr}dB"
            print(f"    Zone {z_idx} ({snr_label}): "
                  f"adapt ov={format_ms(a_met['overlap'])}, "
                  f"oracle ov={format_ms(o_met['overlap'])}, "
                  f"n={len(z_s)}")

    # ── Summary ──
    print(f"\n{'=' * 110}")
    print(f"  FINAL ADAPTIVE PARAMETERS")
    print(f"{'=' * 110}")
    print(f"  T_mod  = {best_t_mod}  (LOW → MODERATE boundary)")
    print(f"  T_high = {best_t_high}  (MODERATE → HIGH boundary)")
    print(f"  N      = {best_ema['n']}  (EMA window, alpha={best_ema['alpha']:.4f})")
    print(f"  Margin = {best_ema['margin']}  (hysteresis)")
    print(f"  Dwell  = {best_ema['dwell']}  (min segments between changes)")
    print(f"  Warmup = {best_ema['warmup']}  (segments after B)")
    print(f"  Default = MODERATE during warmup")
    print(f"\n  Tiers: LOW→0.82, MODERATE→0.84, HIGH→1.0 (disabled)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
