#!/usr/bin/env python3
"""Validate the Conversation Timer app's audio pipeline using real or synthetic audio.

Mirrors the Kotlin app's algorithm in Python:
  Audio frames → Silero VAD → MFCC extraction → Speaker identification → State machine → Metrics

For real audio, also runs Whisper transcription as independent ground truth.

Usage:
  python3 scripts/validate_audio.py test_audio/simple.wav
  python3 scripts/validate_audio.py test_audio/simple.wav --ground-truth test_audio/simple_ground_truth.json
  python3 scripts/validate_audio.py real_conversation.wav  # runs Whisper + pipeline
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

# ──────────────────────────────────────────────────────────────────
# Constants matching the Kotlin app
# ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_SIZE = 512          # samples per frame (~32ms)
FRAME_MS = FRAME_SIZE / SAMPLE_RATE * 1000  # 32ms
FFT_SIZE = 512
NUM_MEL_FILTERS = 26
NUM_MFCC_COEFFS = 13
PRE_EMPHASIS = 0.97

SPEECH_SEGMENT_FRAMES = 5   # buffer 5 speech frames before speaker ID
MIN_FLUSH_FRAMES = 2        # flush partial buffer if >= 2 frames

SPEAKER_SIM_THRESHOLD = 0.80
SPEAKER_AMBIGUITY_MARGIN = 0.10
SPEAKER_EMA_ALPHA = 0.1
DROP_C0 = True  # Exclude C0 (log energy) from MFCC — prevents it dominating cosine sim


# ──────────────────────────────────────────────────────────────────
# MFCC Extraction (matching MfccExtractor.kt exactly)
# ──────────────────────────────────────────────────────────────────

class MfccExtractor:
    def __init__(self, sample_rate=SAMPLE_RATE, num_coeffs=NUM_MFCC_COEFFS,
                 num_filters=NUM_MEL_FILTERS, fft_size=FFT_SIZE):
        self.sample_rate = sample_rate
        self.num_coeffs = num_coeffs
        self.num_filters = num_filters
        self.fft_size = fft_size
        self.spec_size = fft_size // 2 + 1

        # Pre-compute Hamming window
        self.hamming = np.array([
            0.54 - 0.46 * math.cos(2.0 * math.pi * i / (fft_size - 1))
            for i in range(fft_size)
        ], dtype=np.float32)

        # Pre-compute mel filterbank (matching Kotlin's createMelFilterbank)
        self.filterbank = self._create_mel_filterbank()

        # Pre-compute DCT basis (matching Kotlin's dctBasis)
        self.dct_basis = np.array([
            [math.cos(math.pi * i * (j + 0.5) / num_filters)
             for j in range(num_filters)]
            for i in range(num_coeffs)
        ], dtype=np.float32)

    def _hz_to_mel(self, hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self):
        spec_size = self.spec_size
        low_mel = self._hz_to_mel(0.0)
        high_mel = self._hz_to_mel(self.sample_rate / 2.0)

        mel_points = np.array([
            low_mel + i * (high_mel - low_mel) / (self.num_filters + 1)
            for i in range(self.num_filters + 2)
        ])

        bin_points = np.array([
            int(math.floor(self._mel_to_hz(m) * (self.fft_size + 1) / self.sample_rate))
            for m in mel_points
        ], dtype=np.int32)

        filterbank = np.zeros((self.num_filters, spec_size), dtype=np.float32)
        for i in range(self.num_filters):
            for j in range(bin_points[i], bin_points[i + 1]):
                if j < spec_size:
                    denom = max(1, bin_points[i + 1] - bin_points[i])
                    filterbank[i, j] = (j - bin_points[i]) / denom
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                if j < spec_size:
                    denom = max(1, bin_points[i + 2] - bin_points[i + 1])
                    filterbank[i, j] = (bin_points[i + 2] - j) / denom

        return filterbank

    def extract(self, samples_int16: np.ndarray) -> np.ndarray:
        """Extract MFCCs from a frame of int16 samples. Matches MfccExtractor.kt."""
        n = min(len(samples_int16), self.fft_size)
        samples = samples_int16.astype(np.float32)

        # Pre-emphasis + Hamming window
        frame = np.zeros(self.fft_size, dtype=np.float32)
        frame[0] = samples[0] * self.hamming[0]
        for i in range(1, n):
            frame[i] = (samples[i] - PRE_EMPHASIS * samples[i - 1]) * self.hamming[i]

        # FFT → power spectrum
        fft_result = np.fft.fft(frame)
        power = (np.abs(fft_result[:self.spec_size]) ** 2) / self.fft_size

        # Mel filterbank → log energies
        mel_energies = self.filterbank @ power
        log_mel = np.where(mel_energies > 1e-10,
                           np.log(mel_energies),
                           -23.0).astype(np.float32)

        # DCT
        mfcc = self.dct_basis @ log_mel
        return mfcc


# ──────────────────────────────────────────────────────────────────
# Speaker Identification (matching SpeakerIdentifier.kt exactly)
# ──────────────────────────────────────────────────────────────────

class SpeakerIdentifier:
    def __init__(self, threshold=SPEAKER_SIM_THRESHOLD,
                 margin=SPEAKER_AMBIGUITY_MARGIN,
                 ema_alpha=SPEAKER_EMA_ALPHA):
        self.threshold = threshold
        self.margin = margin
        self.ema_alpha = ema_alpha
        self.ref_a = None
        self.ref_b = None
        self.last_speaker = None

    def reset(self):
        self.ref_a = None
        self.ref_b = None
        self.last_speaker = None

    @staticmethod
    def _cosine_sim(a, b):
        dot = np.dot(a, b)
        norm_a = np.sqrt(np.dot(a, a))
        norm_b = np.sqrt(np.dot(b, b))
        denom = norm_a * norm_b
        return float(dot / denom) if denom > 0 else 0.0

    def _update_ref(self, speaker, mfcc):
        ref = self.ref_a if speaker == "A" else self.ref_b
        if ref is not None:
            ref[:] = (1 - self.ema_alpha) * ref + self.ema_alpha * mfcc

    def identify(self, mfcc: np.ndarray) -> str:
        """Returns 'A' or 'B'."""
        if self.ref_a is None:
            self.ref_a = mfcc.copy()
            self.last_speaker = "A"
            return "A"

        sim_a = self._cosine_sim(mfcc, self.ref_a)

        if self.ref_b is None:
            if sim_a >= self.threshold:
                self._update_ref("A", mfcc)
                self.last_speaker = "A"
                return "A"
            else:
                self.ref_b = mfcc.copy()
                self.last_speaker = "B"
                return "B"

        sim_b = self._cosine_sim(mfcc, self.ref_b)

        if sim_a > sim_b + self.margin:
            speaker = "A"
        elif sim_b > sim_a + self.margin:
            speaker = "B"
        else:
            speaker = self.last_speaker or "A"

        self._update_ref(speaker, mfcc)
        self.last_speaker = speaker
        return speaker


# ──────────────────────────────────────────────────────────────────
# Conversation State Machine (matching ConversationStateMachine.kt)
# ──────────────────────────────────────────────────────────────────

class ConversationStateMachine:
    IDLE = "IDLE"
    INITIAL_SILENCE = "INITIAL_SILENCE"
    SPEAKER_A = "SPEAKER_A_TALKING"
    SPEAKER_B = "SPEAKER_B_TALKING"
    PENDING_SILENCE = "PENDING_SILENCE"
    STOPPED = "STOPPED"

    def __init__(self):
        self.state = self.IDLE
        self.last_active_speaker = None
        self._pending_ms = 0
        self._pending_last_speaker = None
        # Raw accumulators
        self.trt = 0
        self.wta = 0
        self.wtb = 0
        self.sta = 0
        self.stb = 0
        self.stm = 0

    def on_record(self):
        self.state = self.INITIAL_SILENCE
        self.last_active_speaker = None
        self._pending_ms = 0
        self._pending_last_speaker = None
        self.trt = self.wta = self.wtb = self.sta = self.stb = self.stm = 0

    def on_stop(self):
        self._pending_ms = 0
        self._pending_last_speaker = None
        self.state = self.STOPPED

    def on_speech(self, speaker: str):
        if self.state == self.INITIAL_SILENCE:
            self.state = self.SPEAKER_A if speaker == "A" else self.SPEAKER_B
            self.last_active_speaker = speaker

        elif self.state == self.SPEAKER_A:
            if speaker == "B":
                self.state = self.SPEAKER_B
                self.last_active_speaker = "B"

        elif self.state == self.SPEAKER_B:
            if speaker == "A":
                self.state = self.SPEAKER_A
                self.last_active_speaker = "A"

        elif self.state == self.PENDING_SILENCE:
            self._resolve_pending(speaker)
            self.state = self.SPEAKER_A if speaker == "A" else self.SPEAKER_B
            self.last_active_speaker = speaker

    def on_silence(self):
        if self.state in (self.SPEAKER_A, self.SPEAKER_B):
            self._pending_last_speaker = self.last_active_speaker
            self._pending_ms = 0
            self.state = self.PENDING_SILENCE

    def on_tick(self, dt_ms: float):
        if self.state in (self.IDLE, self.STOPPED):
            return
        self.trt += dt_ms

        if self.state == self.SPEAKER_A:
            self.wta += dt_ms
        elif self.state == self.SPEAKER_B:
            self.wtb += dt_ms
        elif self.state == self.PENDING_SILENCE:
            self._pending_ms += dt_ms
        # INITIAL_SILENCE → just TRT accumulates (BFST = TRT - TCT)

    def _resolve_pending(self, next_speaker: str):
        prev = self._pending_last_speaker
        ms = self._pending_ms
        if prev is None:
            return
        if next_speaker == prev:
            if prev == "A":
                self.sta += ms
            else:
                self.stb += ms
        else:
            self.stm += ms
        self._pending_ms = 0
        self._pending_last_speaker = None

    def metrics(self) -> dict:
        cta = self.wta + self.sta
        ctb = self.wtb + self.stb
        tct = cta + self.stm + ctb
        return {
            "trt": round(self.trt),
            "wta": round(self.wta),
            "wtb": round(self.wtb),
            "sta": round(self.sta),
            "stb": round(self.stb),
            "stm": round(self.stm),
            "cta": round(cta),
            "ctb": round(ctb),
            "tct": round(tct),
            "tst": round(self.sta + self.stb + self.stm),
            "bfst": round(self.trt - tct),
        }


# ──────────────────────────────────────────────────────────────────
# Audio Pipeline Simulation (matching AudioPipeline.processFrame)
# ──────────────────────────────────────────────────────────────────

class GroundTruthSpeakerOracle:
    """Looks up the ground-truth speaker for a given timestamp.
    Used for synthetic audio where MFCC can't distinguish speakers."""

    def __init__(self, segments: list[dict]):
        self.segments = segments  # [{speaker, start_ms, end_ms}, ...]

    def speaker_at(self, ms: float) -> str | None:
        for seg in self.segments:
            if seg["speaker"] in ("A", "B") and seg["start_ms"] <= ms < seg["end_ms"]:
                return seg["speaker"]
        return None


class PipelineSimulator:
    """Simulates the app's AudioPipeline: VAD → MFCC buffer → Speaker ID → State Machine."""

    def __init__(self, vad_function, speaker_oracle: GroundTruthSpeakerOracle | None = None):
        """
        Args:
            vad_function: callable(frame_int16) -> bool  (True = speech)
            speaker_oracle: if provided, use ground-truth speaker labels instead of MFCC.
                            Useful for synthetic audio where MFCC can't distinguish speakers.
        """
        self.vad = vad_function
        self.mfcc = MfccExtractor()
        self.speaker_id = SpeakerIdentifier()
        self.speaker_oracle = speaker_oracle
        self.state_machine = ConversationStateMachine()
        self.speech_buffer = []
        self.speech_buffer_start_frame = 0  # track where buffered speech started
        self.timeline = []  # list of (start_ms, end_ms, label) for visualization

    def run(self, audio_int16: np.ndarray) -> dict:
        """Process the entire audio and return metrics + timeline."""
        self.state_machine.on_record()
        self.speaker_id.reset()
        self.speech_buffer = []
        self.speech_buffer_start_frame = 0
        self.timeline = []

        n_frames = len(audio_int16) // FRAME_SIZE

        for i in range(n_frames):
            start = i * FRAME_SIZE
            frame = audio_int16[start:start + FRAME_SIZE]

            is_speech = self.vad(frame)

            if is_speech:
                if not self.speech_buffer:
                    self.speech_buffer_start_frame = i
                self.speech_buffer.append(frame)
                if len(self.speech_buffer) >= SPEECH_SEGMENT_FRAMES:
                    speaker = self._identify_and_clear(i)
                    self.state_machine.on_speech(speaker)
                    self._update_timeline(i, speaker)
            else:
                if len(self.speech_buffer) >= MIN_FLUSH_FRAMES:
                    speaker = self._identify_and_clear(i)
                    self.state_machine.on_speech(speaker)
                    self._update_timeline(i, speaker)
                self.speech_buffer.clear()
                self.state_machine.on_silence()
                self._update_timeline(i, "silence")

            self.state_machine.on_tick(FRAME_MS)

        self.state_machine.on_stop()
        return self.state_machine.metrics()

    def _identify_and_clear(self, current_frame: int) -> str:
        # If we have a ground-truth oracle, use it instead of MFCC
        if self.speaker_oracle is not None:
            mid_frame = (self.speech_buffer_start_frame + current_frame) // 2
            ms = mid_frame * FRAME_MS
            oracle_speaker = self.speaker_oracle.speaker_at(ms)
            self.speech_buffer.clear()
            return oracle_speaker or "A"

        # Normal MFCC-based identification
        avg_mfcc = np.zeros(NUM_MFCC_COEFFS, dtype=np.float32)
        for frame in self.speech_buffer:
            avg_mfcc += self.mfcc.extract(frame)
        if len(self.speech_buffer) > 0:
            avg_mfcc /= len(self.speech_buffer)
        self.speech_buffer.clear()

        # Drop C0 (log energy) — it dominates cosine similarity
        if DROP_C0:
            avg_mfcc = avg_mfcc[1:]

        return self.speaker_id.identify(avg_mfcc)

    def _update_timeline(self, frame_idx, label):
        ms = round(frame_idx * FRAME_MS)
        if self.timeline and self.timeline[-1][2] == label:
            # Extend current segment
            self.timeline[-1] = (self.timeline[-1][0], ms + round(FRAME_MS), label)
        else:
            self.timeline.append((ms, ms + round(FRAME_MS), label))


# ──────────────────────────────────────────────────────────────────
# VAD options
# ──────────────────────────────────────────────────────────────────

def create_silero_vad():
    """Create a Silero VAD instance and return a frame-level callable.
    Best for real speech; doesn't work on synthetic harmonic signals."""
    from silero_vad import load_silero_vad
    import torch

    model = load_silero_vad(onnx=True)

    def is_speech(frame_int16: np.ndarray) -> bool:
        audio_f32 = frame_int16.astype(np.float32) / 32768.0
        tensor = torch.FloatTensor(audio_f32)
        prob = model(tensor, SAMPLE_RATE).item()
        return prob > 0.5

    return is_speech


def create_energy_vad(threshold_db: float = -40.0):
    """Simple energy-based VAD for synthetic test audio.
    Classifies a frame as speech if its RMS energy exceeds threshold_db (dBFS).
    Default -40 dBFS works well for silence=zeros vs harmonic signals."""

    def is_speech(frame_int16: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(frame_int16.astype(np.float64) ** 2))
        if rms < 1:
            return False
        db = 20.0 * math.log10(rms / 32768.0)
        return db > threshold_db

    return is_speech


# ──────────────────────────────────────────────────────────────────
# Whisper ground truth
# ──────────────────────────────────────────────────────────────────

def run_whisper(wav_path: str) -> dict:
    """Run faster-whisper on the audio file and return transcript + segments."""
    from faster_whisper import WhisperModel

    print("  Loading Whisper model (base, CPU)... ", end="", flush=True)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("done.")

    print("  Transcribing... ", end="", flush=True)
    segments, info = model.transcribe(wav_path, word_timestamps=True)
    segments = list(segments)
    print(f"done. ({info.duration:.1f}s audio, language={info.language})")

    result = {
        "language": info.language,
        "duration_s": info.duration,
        "segments": [],
    }
    for seg in segments:
        seg_data = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        }
        if seg.words:
            seg_data["words"] = [
                {"word": w.word.strip(), "start": w.start, "end": w.end}
                for w in seg.words
            ]
        result["segments"].append(seg_data)

    return result


# ──────────────────────────────────────────────────────────────────
# Audio loading / conversion
# ──────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio as 16kHz mono int16. Converts via ffmpeg if needed."""
    try:
        sr, data = wavfile.read(path)
        if sr == SAMPLE_RATE and data.dtype == np.int16 and data.ndim == 1:
            return data
    except Exception:
        pass

    # Convert via ffmpeg to 16kHz mono 16-bit WAV
    print(f"  Converting to 16kHz mono WAV via ffmpeg...")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
            tmp.name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sr, data = wavfile.read(tmp.name)
        return data.astype(np.int16)
    finally:
        os.unlink(tmp.name)


# ──────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────

def format_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def print_metrics_comparison(pipeline_metrics: dict, ground_truth_metrics: dict | None,
                             whisper_info: dict | None):
    """Print a formatted comparison table."""
    print("\n" + "=" * 65)
    print("  METRICS COMPARISON")
    print("=" * 65)

    labels = {
        "trt": "TRT  (Total Recording Time)",
        "wta": "WTA  (Word Time A)",
        "wtb": "WTB  (Word Time B)",
        "sta": "STA  (Silence Time A)",
        "stb": "STB  (Silence Time B)",
        "stm": "STM  (Silence Time Mixed)",
        "cta": "CTA  (Conversation Time A)",
        "ctb": "CTB  (Conversation Time B)",
        "tct": "TCT  (Total Conversation Time)",
        "tst": "TST  (Total Silence Time)",
        "bfst": "BFST (Before/Final Silence Time)",
    }

    header = f"{'Metric':<32s}  {'Pipeline':>9s}"
    if ground_truth_metrics:
        header += f"  {'Expected':>9s}  {'Diff':>8s}  {'%Err':>6s}"
    print(header)
    print("-" * len(header))

    for key in ["trt", "wta", "wtb", "sta", "stb", "stm", "cta", "ctb", "tct", "tst", "bfst"]:
        pval = pipeline_metrics.get(key, 0)
        row = f"{labels.get(key, key):<32s}  {format_ms(pval):>9s}"

        if ground_truth_metrics:
            gval = ground_truth_metrics.get(key, 0)
            diff = pval - gval
            pct = abs(diff) / gval * 100 if gval != 0 else 0
            row += f"  {format_ms(gval):>9s}  {diff:>+7.0f}ms  {pct:>5.1f}%"

        print(row)

    # TRT = TCT + BFST invariant check
    trt = pipeline_metrics["trt"]
    tct = pipeline_metrics["tct"]
    bfst = pipeline_metrics["bfst"]
    invariant_ok = abs(trt - (tct + bfst)) < 1  # rounding tolerance
    print(f"\n  TRT = TCT + BFST invariant: {trt} = {tct} + {bfst}  "
          f"{'PASS' if invariant_ok else 'FAIL'}")

    if whisper_info:
        print(f"\n{'=' * 65}")
        print("  WHISPER TRANSCRIPT")
        print("=" * 65)
        for seg in whisper_info["segments"]:
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")


def print_timeline(timeline: list, max_width: int = 60):
    """Print a visual timeline of speaker/silence segments."""
    if not timeline:
        return
    print(f"\n{'=' * 65}")
    print("  SPEAKER TIMELINE")
    print("=" * 65)
    total_ms = timeline[-1][1]
    for start, end, label in timeline:
        dur = end - start
        bar_len = max(1, round(dur / total_ms * max_width))
        if label == "silence":
            char = "\u00b7"
        elif label == "A":
            char = "A"
        else:
            char = "B"
        bar = char * bar_len
        print(f"  {start:6.0f}ms - {end:6.0f}ms  [{bar:<{max_width}s}] {label} ({dur:.0f}ms)")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Conversation Timer pipeline on audio files")
    parser.add_argument("audio", help="Path to WAV/audio file")
    parser.add_argument("--ground-truth", "-g",
                        help="Path to ground truth JSON (from generate_test_audio.py)")
    parser.add_argument("--no-whisper", action="store_true",
                        help="Skip Whisper transcription")
    parser.add_argument("--no-timeline", action="store_true",
                        help="Skip timeline visualization")
    parser.add_argument("--energy-vad", action="store_true",
                        help="Use energy-based VAD (for synthetic audio) instead of Silero")
    parser.add_argument("--oracle-speakers", action="store_true",
                        help="Use ground-truth speaker labels instead of MFCC speaker ID "
                             "(requires --ground-truth; for synthetic audio)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON instead of table")
    args = parser.parse_args()

    print(f"=== Audio Pipeline Validation ===\n")
    print(f"Input: {args.audio}")

    # Load audio
    audio = load_audio(args.audio)
    duration_s = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration_s:.1f}s ({len(audio)} samples, {len(audio) // FRAME_SIZE} frames)")

    # Load ground truth if provided
    gt_metrics = None
    if args.ground_truth:
        with open(args.ground_truth) as f:
            gt_data = json.load(f)
        gt_metrics = gt_data.get("expected_metrics")
        print(f"Ground truth: {args.ground_truth} ({gt_data.get('description', 'N/A')})")

    # Initialize VAD
    if args.energy_vad:
        print(f"\nUsing energy-based VAD (for synthetic audio)")
        vad_fn = create_energy_vad()
    else:
        print(f"\nInitializing Silero VAD...")
        vad_fn = create_silero_vad()

    # Set up speaker oracle if requested
    oracle = None
    if args.oracle_speakers:
        if not gt_metrics:
            print("ERROR: --oracle-speakers requires --ground-truth")
            sys.exit(1)
        oracle = GroundTruthSpeakerOracle(gt_data["segments"])
        print("Using ground-truth speaker oracle (bypassing MFCC speaker ID)")

    # Run pipeline simulation
    print(f"Running pipeline simulation ({len(audio) // FRAME_SIZE} frames)...")
    pipeline = PipelineSimulator(vad_fn, speaker_oracle=oracle)
    pipeline_metrics = pipeline.run(audio)

    # Run Whisper (if not skipped and no ground truth)
    whisper_info = None
    if not args.no_whisper:
        print(f"\nRunning Whisper transcription...")
        try:
            whisper_info = run_whisper(args.audio)
        except Exception as e:
            print(f"  Whisper failed: {e}")

    # Output
    if args.json:
        result = {
            "audio": args.audio,
            "duration_s": duration_s,
            "pipeline_metrics": pipeline_metrics,
        }
        if gt_metrics:
            result["ground_truth_metrics"] = gt_metrics
        if whisper_info:
            result["whisper"] = whisper_info
        print(json.dumps(result, indent=2))
    else:
        print_metrics_comparison(pipeline_metrics, gt_metrics, whisper_info)
        if not args.no_timeline:
            print_timeline(pipeline.timeline)

    print()


if __name__ == "__main__":
    main()
