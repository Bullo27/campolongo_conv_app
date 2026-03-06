#!/usr/bin/env python3
"""Save noisy versions of test clips as WAV files for manual listening."""
import os
import sys
import numpy as np
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_audio import load_audio, SAMPLE_RATE

CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test clips")
OUTPUT_DIR = os.path.join(CLIP_DIR, "noisy")

CLIPS = {
    1: os.path.join(CLIP_DIR, "clip1_theater_conversation.mp3"),
    2: os.path.join(CLIP_DIR, "clip2_tv_study_interview.mp3"),
}

SNR_LEVELS = [40, 30, 20, 15, 10, 5]


def add_gaussian_noise(audio_int16, snr_db, seed=42):
    audio_f64 = audio_int16.astype(np.float64)
    signal_power = np.mean(audio_f64 ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.default_rng(seed).normal(0, np.sqrt(noise_power), len(audio_f64))
    return np.clip(audio_f64 + noise, -32768, 32767).astype(np.int16)


def save_wav(audio_int16, path):
    """Save int16 audio as 16kHz mono WAV."""
    import struct
    n = len(audio_int16)
    data = audio_int16.tobytes()
    with open(path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(data)))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # mono
        f.write(struct.pack('<I', SAMPLE_RATE))
        f.write(struct.pack('<I', SAMPLE_RATE * 2))  # byte rate
        f.write(struct.pack('<H', 2))   # block align
        f.write(struct.pack('<H', 16))  # bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', len(data)))
        f.write(data)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for clip_num in [1, 2]:
        clip_path = CLIPS[clip_num]
        if not os.path.exists(clip_path):
            print(f"Clip {clip_num} not found, skipping")
            continue

        print(f"Loading clip {clip_num}...")
        audio = load_audio(clip_path)

        for snr in SNR_LEVELS:
            noisy = add_gaussian_noise(audio, snr)
            out_path = os.path.join(OUTPUT_DIR, f"clip{clip_num}_snr{snr}dB.wav")
            save_wav(noisy, out_path)
            print(f"  Saved: {out_path}")

    print(f"\nAll files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
