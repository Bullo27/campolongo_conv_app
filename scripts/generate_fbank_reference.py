#!/usr/bin/env python3
"""Generate Fbank reference data using torchaudio for Kotlin cross-validation.

Produces JSON file with known Fbank output for synthetic audio signals,
matching the exact parameters used in the Conversation Timer pipeline.
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torchaudio


# Fbank parameters (must match FbankExtractor.kt and validate_audio.py)
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
NUM_MEL_BINS = 80


def generate_sine_int16(frequency_hz: float, num_samples: int, amplitude: float = 16000.0) -> np.ndarray:
    """Generate int16 sine wave, matching Kotlin's sineAudio() helper."""
    t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE
    samples = np.sin(2.0 * math.pi * frequency_hz * t) * amplitude
    return samples.astype(np.int16)


def compute_fbank(audio_int16: np.ndarray) -> np.ndarray:
    """Compute Fbank features using torchaudio with pipeline-matching parameters."""
    audio_f32 = torch.FloatTensor(audio_int16.astype(np.float32) / 32768.0).unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        audio_f32,
        sample_frequency=SAMPLE_RATE,
        frame_length=FRAME_LENGTH_MS,
        frame_shift=FRAME_SHIFT_MS,
        num_mel_bins=NUM_MEL_BINS,
        dither=0.0,
        energy_floor=1.0,
        window_type="hamming",
    )
    return fbank.numpy()  # [T, 80]


def main():
    # Generate 440 Hz sine wave, 1 second
    frequency_hz = 440.0
    num_samples = 16000
    amplitude = 16000.0

    audio = generate_sine_int16(frequency_hz, num_samples, amplitude)
    fbank = compute_fbank(audio)

    num_frames, num_bins = fbank.shape
    print(f"Audio: {frequency_hz} Hz sine, {num_samples} samples")
    print(f"Fbank shape: [{num_frames}, {num_bins}]")
    print(f"Value range: [{fbank.min():.4f}, {fbank.max():.4f}]")
    print(f"Mean: {fbank.mean():.4f}")

    # Save as JSON
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "src", "test", "resources", "fbank_reference_440hz.json"
    )

    # Use script dir as fallback
    if not os.path.isdir(os.path.dirname(output_path)):
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "app", "src", "test", "resources", "fbank_reference_440hz.json"
        )

    reference = {
        "frequency_hz": frequency_hz,
        "sample_rate": SAMPLE_RATE,
        "num_samples": num_samples,
        "amplitude": amplitude,
        "num_frames": int(num_frames),
        "num_mel_bins": int(num_bins),
        "fbank": [round(float(v), 6) for v in fbank.flatten().tolist()],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reference, f)

    print(f"Saved reference to: {output_path}")
    print(f"Total values: {len(reference['fbank'])}")


if __name__ == "__main__":
    main()
