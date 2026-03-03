#!/usr/bin/env python3
"""Generate synthetic two-speaker conversation WAV files with known ground truth.

Creates a WAV file where Speaker A and Speaker B alternate, separated by silence.
Uses multi-harmonic signals (voice-like) so that MFCC-based speaker ID can
distinguish them — unlike pure sine waves, which Phase 1 showed are too similar
in MFCC space.

Ground truth is printed as JSON so validate_audio.py can compare against it.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000


def make_voice(duration_s: float, f0: float, formants: list[tuple[float, float]],
               num_harmonics: int = 30, noise_ratio: float = 0.02,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthesize a voice-like signal with harmonics shaped by formant resonances.

    Formant synthesis: generate a harmonic series at f0, then apply
    Gaussian-shaped formant filters so the spectral envelope differs
    between speakers — producing different MFCCs.

    Args:
        duration_s: Duration in seconds.
        f0: Fundamental frequency (Hz). ~120 for male, ~220 for female.
        formants: List of (center_freq_hz, bandwidth_hz) pairs. Typical:
                  Male:   [(700, 130), (1220, 70), (2600, 160)]
                  Female: [(310, 60), (2790, 90), (3310, 140)]
        num_harmonics: Number of harmonics to generate.
        noise_ratio: Amplitude of additive noise.
        rng: Random generator for noise.

    Returns:
        Float64 array normalised to [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE

    signal = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, num_harmonics + 1):
        freq = f0 * k
        if freq >= SAMPLE_RATE / 2:
            break
        # Compute formant gain: product of Gaussian resonances
        gain = 0.0
        for fc, bw in formants:
            gain += np.exp(-0.5 * ((freq - fc) / bw) ** 2)
        gain = max(gain, 0.01)  # floor to avoid total silence at any harmonic
        signal += gain * np.sin(2 * np.pi * freq * t)

    # Normalize harmonics to peak=1
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    # Add slight noise for realism
    signal += noise_ratio * rng.standard_normal(n_samples)

    # Final normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    return signal


# Formant presets — designed to be maximally different in MFCC space
# Male voice: low F1, low F2 — energy concentrated in low frequencies
FORMANTS_A = [(700, 130), (1220, 70), (2600, 160)]
# Female voice: high F1, high F2 — energy concentrated in high frequencies
FORMANTS_B = [(310, 60), (2790, 90), (3310, 140)]


def make_silence(duration_s: float) -> np.ndarray:
    """Pure silence (zeros)."""
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float64)


def build_conversation(segments: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Build a WAV from a list of segment descriptions.

    Each segment: {"speaker": "A"|"B"|"silence", "duration": seconds}

    Returns (audio_float64, ground_truth_segments) where ground truth has
    start_ms / end_ms added.
    """
    parts = []
    ground_truth = []
    rng_a = np.random.default_rng(100)
    rng_b = np.random.default_rng(200)
    offset_samples = 0

    for seg in segments:
        dur = seg["duration"]
        speaker = seg["speaker"]

        if speaker == "A":
            audio = make_voice(dur, f0=120.0, formants=FORMANTS_A, rng=rng_a)
        elif speaker == "B":
            audio = make_voice(dur, f0=220.0, formants=FORMANTS_B, rng=rng_b)
        else:
            audio = make_silence(dur)

        start_ms = round(offset_samples / SAMPLE_RATE * 1000)
        end_ms = round((offset_samples + len(audio)) / SAMPLE_RATE * 1000)

        ground_truth.append({
            "speaker": speaker,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
        })

        parts.append(audio)
        offset_samples += len(audio)

    audio = np.concatenate(parts)
    return audio, ground_truth


# --- Predefined test scenarios ---

SCENARIOS = {
    "simple": {
        "description": "Simple A-silence-B-silence-A conversation",
        "segments": [
            {"speaker": "silence", "duration": 0.5},
            {"speaker": "A", "duration": 3.0},
            {"speaker": "silence", "duration": 1.0},
            {"speaker": "B", "duration": 4.0},
            {"speaker": "silence", "duration": 0.5},
            {"speaker": "A", "duration": 2.0},
            {"speaker": "silence", "duration": 1.0},
        ],
    },
    "rapid": {
        "description": "Rapid turn-taking with short silences",
        "segments": [
            {"speaker": "A", "duration": 1.5},
            {"speaker": "silence", "duration": 0.3},
            {"speaker": "B", "duration": 1.0},
            {"speaker": "silence", "duration": 0.2},
            {"speaker": "A", "duration": 2.0},
            {"speaker": "silence", "duration": 0.3},
            {"speaker": "B", "duration": 1.5},
            {"speaker": "silence", "duration": 0.5},
        ],
    },
    "monologue": {
        "description": "Speaker A talks with self-pauses (STA test)",
        "segments": [
            {"speaker": "A", "duration": 3.0},
            {"speaker": "silence", "duration": 1.0},
            {"speaker": "A", "duration": 4.0},
            {"speaker": "silence", "duration": 0.5},
            {"speaker": "A", "duration": 2.0},
        ],
    },
    "long": {
        "description": "Longer conversation (~30s) with varied turn lengths",
        "segments": [
            {"speaker": "silence", "duration": 1.0},
            {"speaker": "A", "duration": 5.0},
            {"speaker": "silence", "duration": 1.5},
            {"speaker": "B", "duration": 3.0},
            {"speaker": "silence", "duration": 0.5},
            {"speaker": "A", "duration": 2.0},
            {"speaker": "silence", "duration": 1.0},
            {"speaker": "B", "duration": 6.0},
            {"speaker": "silence", "duration": 0.8},
            {"speaker": "A", "duration": 3.0},
            {"speaker": "silence", "duration": 0.5},
            {"speaker": "B", "duration": 2.0},
            {"speaker": "silence", "duration": 1.2},
        ],
    },
}


def compute_expected_metrics(ground_truth: list[dict]) -> dict:
    """Compute the metrics the app *should* produce for this ground truth.

    Uses the same state machine logic as the Kotlin app — initial silence
    before first speech is BFST, silence between same speaker is STA/STB,
    silence between different speakers is STM, trailing silence is BFST.
    """
    trt = sum(s["duration_ms"] for s in ground_truth)
    wta = 0
    wtb = 0
    sta = 0
    stb = 0
    stm = 0

    # Walk segments to compute metrics
    last_speaker = None  # last speaking entity (A or B)
    first_speech_seen = False
    initial_silence = 0

    for seg in ground_truth:
        sp = seg["speaker"]
        dur = seg["duration_ms"]

        if sp == "silence":
            if not first_speech_seen:
                initial_silence += dur
            # Silence after speech — will be resolved when next speaker is seen
            # We handle this by looking ahead (or in a second pass)
            continue

        # It's a speech segment
        if not first_speech_seen:
            first_speech_seen = True

        if sp == "A":
            wta += dur
        else:
            wtb += dur

        # Resolve any preceding silence (after the first speech)
        # Find preceding silence segments between this and the last speech
        # (already handled below in the loop approach)

    # Second pass — resolve silences
    last_speaker = None
    first_speech_seen = False
    pending_silence = 0

    for seg in ground_truth:
        sp = seg["speaker"]
        dur = seg["duration_ms"]

        if sp == "silence":
            if first_speech_seen:
                pending_silence += dur
            # Pre-first-speech silence contributes to BFST via trt - tct
            continue

        # Speech segment
        if not first_speech_seen:
            first_speech_seen = True
            last_speaker = sp
            pending_silence = 0
            continue

        # Resolve pending silence
        if pending_silence > 0 and last_speaker is not None:
            if sp == last_speaker:
                if sp == "A":
                    sta += pending_silence
                else:
                    stb += pending_silence
            else:
                stm += pending_silence
        pending_silence = 0
        last_speaker = sp

    # Any trailing pending_silence is unresolved → falls into BFST (trt - tct)

    tct = (wta + sta) + stm + (wtb + stb)
    bfst = trt - tct

    return {
        "trt": trt,
        "wta": wta,
        "wtb": wtb,
        "sta": sta,
        "stb": stb,
        "stm": stm,
        "cta": wta + sta,
        "ctb": wtb + stb,
        "tct": tct,
        "tst": sta + stb + stm,
        "bfst": bfst,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate test conversation WAV files")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()) + ["all"],
                        default="simple",
                        help="Which scenario to generate (default: simple)")
    parser.add_argument("--output-dir", default="test_audio",
                        help="Output directory (default: test_audio/)")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, info in SCENARIOS.items():
            total_dur = sum(s["duration"] for s in info["segments"])
            print(f"  {name:12s} — {info['description']} ({total_dur:.1f}s)")
        return

    scenarios = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}

    os.makedirs(args.output_dir, exist_ok=True)

    for name, info in scenarios.items():
        audio, ground_truth = build_conversation(info["segments"])
        expected = compute_expected_metrics(ground_truth)

        # Save WAV (16-bit PCM)
        wav_path = os.path.join(args.output_dir, f"{name}.wav")
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)

        # Save ground truth JSON
        gt_path = os.path.join(args.output_dir, f"{name}_ground_truth.json")
        gt_data = {
            "scenario": name,
            "description": info["description"],
            "sample_rate": SAMPLE_RATE,
            "duration_ms": sum(s["duration_ms"] for s in ground_truth),
            "segments": ground_truth,
            "expected_metrics": expected,
        }
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2)

        total_dur = sum(s["duration_ms"] for s in ground_truth) / 1000
        print(f"  {name:12s}: {wav_path} ({total_dur:.1f}s) + {gt_path}")

        # Print expected metrics
        print(f"               Expected: TRT={expected['trt']}ms  WTA={expected['wta']}ms  "
              f"WTB={expected['wtb']}ms  STA={expected['sta']}ms  STB={expected['stb']}ms  "
              f"STM={expected['stm']}ms  BFST={expected['bfst']}ms")

    print(f"\nGenerated {len(scenarios)} scenario(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
