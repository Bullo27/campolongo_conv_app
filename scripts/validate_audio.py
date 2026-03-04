#!/usr/bin/env python3
"""Validate the Conversation Timer app's audio pipeline using real or synthetic audio.

Mirrors the Kotlin app's algorithm in Python:
  Audio frames → Silero VAD → MFCC extraction → Speaker identification → State machine → Metrics

Modes:
  Basic:      python3 scripts/validate_audio.py clip.mp3
  Diarize:    python3 scripts/validate_audio.py clip.mp3 --diarize
  Synthetic:  python3 scripts/validate_audio.py test.wav --energy-vad --ground-truth test.json
  JSON out:   python3 scripts/validate_audio.py clip.mp3 --diarize --json
"""

import argparse
import collections
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

# OVT detection thresholds
OVT_BOTH_HIGH_THRESHOLD = 0.60   # both sims must be above this
OVT_DIFF_MARGIN = 0.15           # |sim_A - sim_B| must be below this

# VAD debounce settings (matching Kotlin android-vad library's hysteresis)
# VadEngine.kt: setSilenceDurationMs(300), setSpeechDurationMs(50)
DEFAULT_SILENCE_DEBOUNCE_MS = 300
DEFAULT_SPEECH_DEBOUNCE_MS = 50

# Speaker decision smoothing (majority-vote window over recent decisions)
DEFAULT_SMOOTHING_WINDOW = 1  # 1 = no smoothing (backward compatible)
DEFAULT_B_CONFIRM_FRAMES = 2  # consecutive below-threshold decisions to establish Speaker B

# Feature enrichment settings (Wave 2)
FEATURE_MODES = ("static", "delta", "delta+dd")  # 12, 24, 36 dims
DEFAULT_FEATURE_MODE = "static"
DEFAULT_CMVN_WINDOW = 0   # 0 = off (backward compatible)
N_DELTA = 2               # regression width for delta computation


# ──────────────────────────────────────────────────────────────────
# Cepstral Mean Variance Normalization (CMVN)
# ──────────────────────────────────────────────────────────────────

class CepstralNormalizer:
    """Running CMVN over a sliding window of recent MFCC vectors.
    Normalizes each dimension to zero mean and unit variance.
    """

    def __init__(self, window_size: int = 30, min_frames: int = 3):
        self._history = collections.deque(maxlen=window_size)
        self.min_frames = min_frames

    def normalize(self, mfcc: np.ndarray) -> np.ndarray:
        self._history.append(mfcc.copy())
        if len(self._history) < self.min_frames:
            return mfcc  # not enough history yet
        history = np.array(self._history)
        mean = history.mean(axis=0)
        std = np.maximum(history.std(axis=0), 1e-6)
        return (mfcc - mean) / std

    def reset(self):
        self._history.clear()


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
                 ema_alpha=SPEAKER_EMA_ALPHA,
                 b_confirm_frames: int = 2):
        self.threshold = threshold
        self.margin = margin
        self.ema_alpha = ema_alpha
        self.b_confirm_frames = max(1, b_confirm_frames)
        self.ref_a = None
        self.ref_b = None
        self.last_speaker = None
        # B confirmation state
        self._b_candidate_count = 0
        self._b_candidate_mfcc = None
        # Trace info from last identify() call
        self.last_sim_a = 0.0
        self.last_sim_b = 0.0
        self.last_ambiguous = False

    def reset(self):
        self.ref_a = None
        self.ref_b = None
        self.last_speaker = None
        self._b_candidate_count = 0
        self._b_candidate_mfcc = None
        self.last_sim_a = 0.0
        self.last_sim_b = 0.0
        self.last_ambiguous = False

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
        """Returns 'A' or 'B'. Also populates last_sim_a, last_sim_b, last_ambiguous."""
        if self.ref_a is None:
            self.ref_a = mfcc.copy()
            self.last_speaker = "A"
            self.last_sim_a = 1.0
            self.last_sim_b = 0.0
            self.last_ambiguous = False
            return "A"

        sim_a = self._cosine_sim(mfcc, self.ref_a)
        self.last_sim_a = sim_a

        if self.ref_b is None:
            self.last_sim_b = 0.0
            if sim_a >= self.threshold:
                # Back to A — reset B candidate streak
                self._b_candidate_count = 0
                self._b_candidate_mfcc = None
                self._update_ref("A", mfcc)
                self.last_speaker = "A"
                self.last_ambiguous = False
                return "A"
            else:
                # Below threshold — candidate B frame
                self._b_candidate_count += 1
                if self._b_candidate_mfcc is None:
                    self._b_candidate_mfcc = mfcc.copy()
                else:
                    self._b_candidate_mfcc += mfcc
                if self._b_candidate_count >= self.b_confirm_frames:
                    # Confirmed B — establish from averaged candidate MFCCs
                    self.ref_b = self._b_candidate_mfcc / self._b_candidate_count
                    self._b_candidate_count = 0
                    self._b_candidate_mfcc = None
                    self.last_speaker = "B"
                    self.last_sim_b = 1.0
                    self.last_ambiguous = False
                    return "B"
                # Not yet confirmed — return A as safe default
                # Don't EMA-update ref_a with B-candidate speech (would contaminate A's reference)
                self.last_speaker = "A"
                self.last_ambiguous = False
                return "A"

        sim_b = self._cosine_sim(mfcc, self.ref_b)
        self.last_sim_b = sim_b

        # Check ambiguity (potential overlap)
        both_high = (sim_a > OVT_BOTH_HIGH_THRESHOLD and sim_b > OVT_BOTH_HIGH_THRESHOLD)
        close = abs(sim_a - sim_b) < OVT_DIFF_MARGIN
        self.last_ambiguous = both_high and close

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

    def __init__(self, vad_function, speaker_oracle: GroundTruthSpeakerOracle | None = None,
                 smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
                 speech_segment_frames: int = SPEECH_SEGMENT_FRAMES,
                 b_confirm_frames: int = DEFAULT_B_CONFIRM_FRAMES,
                 feature_mode: str = DEFAULT_FEATURE_MODE,
                 cmvn_window: int = DEFAULT_CMVN_WINDOW):
        self.vad = vad_function
        self.mfcc = MfccExtractor()
        self.speaker_id = SpeakerIdentifier(b_confirm_frames=b_confirm_frames)
        self.speaker_oracle = speaker_oracle
        self.smoothing_window = max(1, smoothing_window)
        self.speech_segment_frames = max(1, speech_segment_frames)
        self.feature_mode = feature_mode
        self.cmvn = CepstralNormalizer(window_size=cmvn_window) if cmvn_window > 0 else None
        self.state_machine = ConversationStateMachine()
        self.speech_buffer = []
        self.speech_buffer_start_frame = 0
        self.timeline = []         # list of (start_ms, end_ms, label)
        self.similarity_trace = [] # list of (time_ms, sim_a, sim_b, decision, ambiguous)
        self.ovt_ms = 0.0         # Overlap/ambiguity time accumulator
        self._decision_buffer = collections.deque(maxlen=self.smoothing_window)
        self._last_smoothed_speaker = "A"  # tie-breaker: keep previous

    def run(self, audio_int16: np.ndarray) -> dict:
        """Process the entire audio and return metrics + timeline + OVT."""
        self.state_machine.on_record()
        self.speaker_id.reset()
        if self.cmvn is not None:
            self.cmvn.reset()
        self.speech_buffer = []
        self.speech_buffer_start_frame = 0
        self.timeline = []
        self.similarity_trace = []
        self.ovt_ms = 0.0
        self._decision_buffer.clear()
        self._last_smoothed_speaker = "A"

        n_frames = len(audio_int16) // FRAME_SIZE

        for i in range(n_frames):
            start = i * FRAME_SIZE
            frame = audio_int16[start:start + FRAME_SIZE]

            is_speech = self.vad(frame)

            if is_speech:
                if not self.speech_buffer:
                    self.speech_buffer_start_frame = i
                self.speech_buffer.append(frame)
                if len(self.speech_buffer) >= self.speech_segment_frames:
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

        metrics = self.state_machine.metrics()
        metrics["ovt"] = round(self.ovt_ms)
        return metrics

    def _identify_and_clear(self, current_frame: int) -> str:
        mid_frame = (self.speech_buffer_start_frame + current_frame) // 2
        time_ms = round(mid_frame * FRAME_MS)

        # If we have a ground-truth oracle, use it instead of MFCC
        if self.speaker_oracle is not None:
            ms = mid_frame * FRAME_MS
            oracle_speaker = self.speaker_oracle.speaker_at(ms)
            self.speech_buffer.clear()
            return oracle_speaker or "A"

        # Normal MFCC-based identification
        # 1. Extract per-frame MFCCs
        frame_mfccs = [self.mfcc.extract(f) for f in self.speech_buffer]
        if DROP_C0:
            frame_mfccs = [m[1:] for m in frame_mfccs]

        n_buffered = len(self.speech_buffer)
        self.speech_buffer.clear()

        # 2. Average static MFCCs
        avg_static = np.mean(frame_mfccs, axis=0)

        # 3. CMVN on avg_static (not on deltas — deltas are already relative)
        if self.cmvn is not None:
            avg_static = self.cmvn.normalize(avg_static)

        # 4. Compute deltas / delta-deltas if requested
        if self.feature_mode in ("delta", "delta+dd"):
            n = len(frame_mfccs)
            norm_factor = 2 * sum(k * k for k in range(1, N_DELTA + 1))  # = 10
            deltas = []
            for t in range(n):
                d = np.zeros_like(frame_mfccs[0])
                for k in range(1, N_DELTA + 1):
                    d += k * (frame_mfccs[min(t + k, n - 1)] - frame_mfccs[max(t - k, 0)])
                deltas.append(d / norm_factor)
            avg_delta = np.mean(deltas, axis=0)

            if self.feature_mode == "delta+dd":
                # Delta-deltas: same regression on delta sequence
                ddeltas = []
                for t in range(n):
                    dd = np.zeros_like(deltas[0])
                    for k in range(1, N_DELTA + 1):
                        dd += k * (deltas[min(t + k, n - 1)] - deltas[max(t - k, 0)])
                    ddeltas.append(dd / norm_factor)
                avg_ddelta = np.mean(ddeltas, axis=0)
                embedding = np.concatenate([avg_static, avg_delta, avg_ddelta])
            else:
                embedding = np.concatenate([avg_static, avg_delta])
        else:
            embedding = avg_static

        raw_speaker = self.speaker_id.identify(embedding)

        # Record similarity trace (raw values, before smoothing)
        sim_a = self.speaker_id.last_sim_a
        sim_b = self.speaker_id.last_sim_b
        ambiguous = self.speaker_id.last_ambiguous
        self.similarity_trace.append((time_ms, sim_a, sim_b, raw_speaker, ambiguous))

        # Accumulate OVT for ambiguous segments
        if ambiguous:
            self.ovt_ms += n_buffered * FRAME_MS

        # Majority-vote smoothing over recent decisions
        self._decision_buffer.append(raw_speaker)
        if self.smoothing_window <= 1:
            return raw_speaker
        count_a = sum(1 for d in self._decision_buffer if d == "A")
        count_b = len(self._decision_buffer) - count_a
        if count_a > count_b:
            smoothed = "A"
        elif count_b > count_a:
            smoothed = "B"
        else:
            smoothed = self._last_smoothed_speaker  # tie → keep previous
        self._last_smoothed_speaker = smoothed
        return smoothed

    def _update_timeline(self, frame_idx, label):
        ms = round(frame_idx * FRAME_MS)
        if self.timeline and self.timeline[-1][2] == label:
            self.timeline[-1] = (self.timeline[-1][0], ms + round(FRAME_MS), label)
        else:
            self.timeline.append((ms, ms + round(FRAME_MS), label))


# ──────────────────────────────────────────────────────────────────
# SpeechBrain Diarizer (ECAPA-TDNN second opinion)
# ──────────────────────────────────────────────────────────────────

def _patch_speechbrain_compat():
    """Monkey-patch torchaudio + huggingface_hub for speechbrain 1.0.3 compatibility.
    - torchaudio 2.10+ removed list_audio_backends()
    - huggingface_hub 1.5+ removed use_auth_token kwarg
    - speechbrain/spkrec-ecapa-voxceleb repo lacks custom.py
    """
    import functools
    import pathlib

    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]

    import huggingface_hub
    _orig_dl = huggingface_hub.hf_hub_download
    @functools.wraps(_orig_dl)
    def _patched_dl(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return _orig_dl(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched_dl

    if hasattr(huggingface_hub, "snapshot_download"):
        _orig_snap = huggingface_hub.snapshot_download
        @functools.wraps(_orig_snap)
        def _patched_snap(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return _orig_snap(*args, **kwargs)
        huggingface_hub.snapshot_download = _patched_snap

    # Create dummy custom.py so speechbrain doesn't try to download it
    savedir = pathlib.Path("/tmp/speechbrain_ecapa")
    savedir.mkdir(parents=True, exist_ok=True)
    dummy_custom = savedir / "custom.py"
    if not dummy_custom.exists():
        dummy_custom.write_text("# dummy\n")

    import speechbrain.utils.fetching as sb_fetch
    _orig_fetch = sb_fetch.fetch
    @functools.wraps(_orig_fetch)
    def _patched_fetch(*args, **kwargs):
        filename = kwargs.get("filename") or (args[0] if args else None)
        if filename and str(filename) == "custom.py":
            return dummy_custom
        return _orig_fetch(*args, **kwargs)
    sb_fetch.fetch = _patched_fetch


class SpeechBrainDiarizer:
    """Speaker diarization using speechbrain ECAPA-TDNN embeddings.
    Uses the same VAD segments as our pipeline for apples-to-apples comparison."""

    def __init__(self):
        _patch_speechbrain_compat()
        from speechbrain.inference.speaker import EncoderClassifier
        import torch
        self.torch = torch

        print("  Loading ECAPA-TDNN model... ", end="", flush=True)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_ecapa",
            run_opts={"device": "cpu"},
        )
        print("done.")

    def extract_embedding(self, audio_float32: np.ndarray) -> np.ndarray:
        """Extract a single embedding from an audio segment (float32, mono, 16kHz)."""
        waveform = self.torch.FloatTensor(audio_float32).unsqueeze(0)
        embedding = self.classifier.encode_batch(waveform)
        return embedding.squeeze().numpy()

    def diarize(self, audio_int16: np.ndarray, vad_segments: list) -> list:
        """Run diarization using fixed-window sliding approach over VAD-detected speech.

        Uses 1.5s windows with 0.75s shift. Only extracts embedding for windows where
        >50% of samples fall within VAD speech regions. This avoids dropping short
        segments and ensures full coverage of speech.

        Args:
            audio_int16: full audio as int16 array (16kHz mono)
            vad_segments: list of (start_ms, end_ms) tuples for speech regions

        Returns:
            timeline: list of (start_ms, end_ms, speaker_label)
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        if not vad_segments:
            return []

        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        total_samples = len(audio_f32)

        # Build sample-level VAD mask
        vad_mask = np.zeros(total_samples, dtype=bool)
        for start_ms, end_ms in vad_segments:
            s = int(start_ms * SAMPLE_RATE / 1000)
            e = min(int(end_ms * SAMPLE_RATE / 1000), total_samples)
            vad_mask[s:e] = True

        # Fixed-window sliding extraction
        window_samples = int(1.5 * SAMPLE_RATE)   # 1.5s = 24000 samples
        shift_samples = int(0.75 * SAMPLE_RATE)    # 0.75s = 12000 samples
        min_speech_ratio = 0.5

        embeddings = []
        window_ranges = []  # (start_ms, end_ms) for each window

        pos = 0
        while pos + window_samples <= total_samples:
            end_pos = pos + window_samples
            speech_ratio = vad_mask[pos:end_pos].mean()

            if speech_ratio >= min_speech_ratio:
                emb = self.extract_embedding(audio_f32[pos:end_pos])
                embeddings.append(emb)
                window_ranges.append((pos * 1000 / SAMPLE_RATE, end_pos * 1000 / SAMPLE_RATE))

            pos += shift_samples

        if len(embeddings) < 2:
            return [(s, e, "SB_0") for s, e in window_ranges]

        embeddings = np.array(embeddings)

        # Agglomerative clustering into 2 speakers
        distances = pdist(embeddings, metric="cosine")
        Z = linkage(distances, method="ward")
        labels = fcluster(Z, t=2, criterion="maxclust")

        # Build timeline: each window gets its speaker label
        # Overlapping windows → later window overwrites (shift < window, so natural)
        # Use sample-level labeling for precision
        speaker_map = np.full(total_samples, -1, dtype=np.int8)  # -1 = unlabeled
        for (start_ms, end_ms), label in zip(window_ranges, labels):
            s = int(start_ms * SAMPLE_RATE / 1000)
            e = min(int(end_ms * SAMPLE_RATE / 1000), total_samples)
            # Only label samples that are within VAD speech
            mask_slice = vad_mask[s:e]
            speaker_map[s:e][mask_slice] = label - 1  # 0 or 1

        # Convert sample-level labels to timeline segments
        timeline = []
        frame_samples = FRAME_SIZE  # ~32ms per frame
        n_frames = total_samples // frame_samples

        for i in range(n_frames):
            s = i * frame_samples
            e = s + frame_samples
            chunk = speaker_map[s:e]
            labeled = chunk[chunk >= 0]
            if len(labeled) == 0:
                continue
            # Majority vote within frame
            counts = np.bincount(labeled, minlength=2)
            winner = int(np.argmax(counts))
            speaker = f"SB_{winner}"
            start_ms = s * 1000 / SAMPLE_RATE
            end_ms = e * 1000 / SAMPLE_RATE

            if timeline and timeline[-1][2] == speaker:
                timeline[-1] = (timeline[-1][0], end_ms, speaker)
            else:
                timeline.append((start_ms, end_ms, speaker))

        return timeline


def extract_vad_segments(pipeline_timeline: list) -> list:
    """Extract speech-only segments from pipeline timeline for speechbrain.
    Returns list of (start_ms, end_ms) for speech regions."""
    segments = []
    for start_ms, end_ms, label in pipeline_timeline:
        if label in ("A", "B"):
            # Merge with previous if contiguous
            if segments and segments[-1][1] >= start_ms - FRAME_MS:
                segments[-1] = (segments[-1][0], end_ms)
            else:
                segments.append((start_ms, end_ms))
    return segments


# ──────────────────────────────────────────────────────────────────
# Timeline Comparison
# ──────────────────────────────────────────────────────────────────

def compare_timelines(our_timeline: list, sb_timeline: list, total_duration_ms: float) -> dict:
    """Compare our pipeline timeline with speechbrain's at frame-level resolution.

    Auto-maps speechbrain labels (SB_0, SB_1) to our (A, B) by majority overlap.

    Returns dict with agreement stats.
    """
    if not our_timeline or not sb_timeline:
        return {"agreement_pct": 0.0, "total_speech_frames": 0,
                "agreed_frames": 0, "label_map": {}}

    n_frames = int(total_duration_ms / FRAME_MS) + 1

    # Build frame-level label arrays
    our_labels = [None] * n_frames
    sb_labels = [None] * n_frames

    for start_ms, end_ms, label in our_timeline:
        if label == "silence":
            continue
        start_f = int(start_ms / FRAME_MS)
        end_f = int(end_ms / FRAME_MS)
        for f in range(max(0, start_f), min(n_frames, end_f)):
            our_labels[f] = label

    for start_ms, end_ms, label in sb_timeline:
        start_f = int(start_ms / FRAME_MS)
        end_f = int(end_ms / FRAME_MS)
        for f in range(max(0, start_f), min(n_frames, end_f)):
            sb_labels[f] = label

    # Find best label mapping: try both and pick the one with higher agreement
    sb_label_set = set(l for l in sb_labels if l is not None)
    if len(sb_label_set) < 2:
        # Only one speaker detected by speechbrain
        sb_list = sorted(sb_label_set)
        label_map = {sb_list[0]: "A"} if sb_list else {}
    else:
        sb_list = sorted(sb_label_set)
        # Try mapping 0→A, 1→B
        map_a = {sb_list[0]: "A", sb_list[1]: "B"}
        # Try mapping 0→B, 1→A
        map_b = {sb_list[0]: "B", sb_list[1]: "A"}

        count_a = count_b = 0
        for f in range(n_frames):
            if our_labels[f] is not None and sb_labels[f] is not None:
                if map_a.get(sb_labels[f]) == our_labels[f]:
                    count_a += 1
                if map_b.get(sb_labels[f]) == our_labels[f]:
                    count_b += 1

        label_map = map_a if count_a >= count_b else map_b

    # Compute agreement stats
    total_speech_frames = 0
    agreed_frames = 0
    our_only_frames = 0
    sb_only_frames = 0

    for f in range(n_frames):
        our = our_labels[f]
        sb = sb_labels[f]

        if our is not None and sb is not None:
            total_speech_frames += 1
            mapped_sb = label_map.get(sb, sb)
            if mapped_sb == our:
                agreed_frames += 1
        elif our is not None:
            our_only_frames += 1
        elif sb is not None:
            sb_only_frames += 1

    agreement_pct = (agreed_frames / total_speech_frames * 100) if total_speech_frames > 0 else 0.0

    # Compute per-speaker speech times from speechbrain
    sb_wta_ms = 0.0
    sb_wtb_ms = 0.0
    for start_ms, end_ms, label in sb_timeline:
        mapped = label_map.get(label, label)
        dur = end_ms - start_ms
        if mapped == "A":
            sb_wta_ms += dur
        elif mapped == "B":
            sb_wtb_ms += dur

    return {
        "agreement_pct": round(agreement_pct, 1),
        "total_speech_frames": total_speech_frames,
        "agreed_frames": agreed_frames,
        "disagreed_frames": total_speech_frames - agreed_frames,
        "our_only_frames": our_only_frames,
        "sb_only_frames": sb_only_frames,
        "label_map": label_map,
        "sb_wta_ms": round(sb_wta_ms),
        "sb_wtb_ms": round(sb_wtb_ms),
    }


# ──────────────────────────────────────────────────────────────────
# VAD options
# ──────────────────────────────────────────────────────────────────

def create_silero_vad(vad_threshold: float = 0.5,
                      silence_debounce_ms: float = DEFAULT_SILENCE_DEBOUNCE_MS,
                      speech_debounce_ms: float = DEFAULT_SPEECH_DEBOUNCE_MS):
    """Create a Silero VAD with hysteresis debouncing matching the Kotlin app.

    The android-vad library uses setSilenceDurationMs(300) / setSpeechDurationMs(50):
    isSpeech() holds its current state until the opposite raw signal persists for
    the debounce duration. This bridges over micro-pauses between words.
    """
    from silero_vad import load_silero_vad
    import torch

    model = load_silero_vad(onnx=True)

    state = {"confirmed_speech": False, "counter_ms": 0.0}

    def is_speech(frame_int16: np.ndarray) -> bool:
        audio_f32 = frame_int16.astype(np.float32) / 32768.0
        tensor = torch.FloatTensor(audio_f32)
        prob = model(tensor, SAMPLE_RATE).item()
        raw_speech = prob > vad_threshold

        if state["confirmed_speech"]:
            if raw_speech:
                state["counter_ms"] = 0.0
            else:
                state["counter_ms"] += FRAME_MS
                if state["counter_ms"] >= silence_debounce_ms:
                    state["confirmed_speech"] = False
                    state["counter_ms"] = 0.0
        else:
            if not raw_speech:
                state["counter_ms"] = 0.0
            else:
                state["counter_ms"] += FRAME_MS
                if state["counter_ms"] >= speech_debounce_ms:
                    state["confirmed_speech"] = True
                    state["counter_ms"] = 0.0

        return state["confirmed_speech"]

    return is_speech


def create_energy_vad(threshold_db: float = -40.0):
    """Simple energy-based VAD for synthetic test audio."""

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
                             whisper_info: dict | None, comparison: dict | None = None):
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
        "ovt": "OVT  (Overlap/Ambiguity Time)",
    }

    header = f"{'Metric':<32s}  {'Pipeline':>9s}"
    if ground_truth_metrics:
        header += f"  {'Expected':>9s}  {'Diff':>8s}  {'%Err':>6s}"
    print(header)
    print("-" * len(header))

    metric_keys = ["trt", "wta", "wtb", "sta", "stb", "stm", "cta", "ctb", "tct", "tst", "bfst"]
    if "ovt" in pipeline_metrics:
        metric_keys.append("ovt")

    for key in metric_keys:
        pval = pipeline_metrics.get(key, 0)
        row = f"{labels.get(key, key):<32s}  {format_ms(pval):>9s}"

        if ground_truth_metrics and key in ground_truth_metrics:
            gval = ground_truth_metrics.get(key, 0)
            diff = pval - gval
            pct = abs(diff) / gval * 100 if gval != 0 else 0
            row += f"  {format_ms(gval):>9s}  {diff:>+7.0f}ms  {pct:>5.1f}%"

        print(row)

    # TRT = TCT + BFST invariant check
    trt = pipeline_metrics["trt"]
    tct = pipeline_metrics["tct"]
    bfst = pipeline_metrics["bfst"]
    invariant_ok = abs(trt - (tct + bfst)) < 1
    print(f"\n  TRT = TCT + BFST invariant: {trt} = {tct} + {bfst}  "
          f"{'PASS' if invariant_ok else 'FAIL'}")

    # OVT as percentage of speech time
    if "ovt" in pipeline_metrics and tct > 0:
        ovt_pct = pipeline_metrics["ovt"] / tct * 100
        print(f"  OVT as % of TCT: {ovt_pct:.1f}%")

    # Speechbrain comparison
    if comparison:
        print(f"\n{'=' * 65}")
        print("  SPEECHBRAIN COMPARISON")
        print("=" * 65)
        print(f"  Frame-level agreement: {comparison['agreement_pct']:.1f}%"
              f" ({comparison['agreed_frames']}/{comparison['total_speech_frames']} speech frames)")
        if comparison.get('disagreed_frames'):
            print(f"  Disagreed frames: {comparison['disagreed_frames']}")
        print(f"  Label mapping: {comparison['label_map']}")
        if comparison.get('sb_wta_ms') is not None:
            print(f"  SpeechBrain WTA: {format_ms(comparison['sb_wta_ms'])}"
                  f"  (ours: {format_ms(pipeline_metrics.get('wta', 0))})")
            print(f"  SpeechBrain WTB: {format_ms(comparison['sb_wtb_ms'])}"
                  f"  (ours: {format_ms(pipeline_metrics.get('wtb', 0))})")

    # Whisper transcript
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


def print_similarity_trace_summary(trace: list):
    """Print a summary of the similarity trace."""
    if not trace:
        return
    print(f"\n{'=' * 65}")
    print("  SIMILARITY TRACE SUMMARY")
    print("=" * 65)

    sim_a_vals = [t[1] for t in trace]
    sim_b_vals = [t[2] for t in trace]
    ambiguous_count = sum(1 for t in trace if t[4])

    print(f"  Total speaker ID decisions: {len(trace)}")
    print(f"  Ambiguous decisions: {ambiguous_count} ({ambiguous_count / len(trace) * 100:.1f}%)")
    print(f"  sim_A range: [{min(sim_a_vals):.3f}, {max(sim_a_vals):.3f}]"
          f"  mean={np.mean(sim_a_vals):.3f}")
    print(f"  sim_B range: [{min(sim_b_vals):.3f}, {max(sim_b_vals):.3f}]"
          f"  mean={np.mean(sim_b_vals):.3f}")

    # Show first 10 decisions
    print(f"\n  First 10 decisions:")
    for i, (t_ms, sa, sb, dec, amb) in enumerate(trace[:10]):
        amb_flag = " [AMBIG]" if amb else ""
        print(f"    {t_ms:6.0f}ms  sim_A={sa:.3f}  sim_B={sb:.3f}  \u2192 {dec}{amb_flag}")
    if len(trace) > 10:
        print(f"    ... ({len(trace) - 10} more)")


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
    parser.add_argument("--diarize", action="store_true",
                        help="Run speechbrain ECAPA-TDNN diarization as second opinion")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="VAD probability threshold (default: 0.5, use 0.8 for noisy)")
    parser.add_argument("--silence-debounce-ms", type=float, default=DEFAULT_SILENCE_DEBOUNCE_MS,
                        help=f"Silence debounce in ms (default: {DEFAULT_SILENCE_DEBOUNCE_MS})")
    parser.add_argument("--speech-debounce-ms", type=float, default=DEFAULT_SPEECH_DEBOUNCE_MS,
                        help=f"Speech debounce in ms (default: {DEFAULT_SPEECH_DEBOUNCE_MS})")
    parser.add_argument("--no-debounce", action="store_true",
                        help="Disable VAD debouncing (raw per-frame decisions)")
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW,
                        help=f"Speaker decision smoothing window size (default: {DEFAULT_SMOOTHING_WINDOW}, "
                             "1=no smoothing, odd values recommended)")
    parser.add_argument("--speech-buffer", type=int, default=SPEECH_SEGMENT_FRAMES,
                        help=f"Speech frames to buffer before speaker ID (default: {SPEECH_SEGMENT_FRAMES})")
    parser.add_argument("--b-confirm-frames", type=int, default=DEFAULT_B_CONFIRM_FRAMES,
                        help=f"Consecutive below-threshold decisions to confirm Speaker B "
                             f"(default: {DEFAULT_B_CONFIRM_FRAMES})")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=DEFAULT_FEATURE_MODE,
                        help=f"Feature mode: static (12 dim), delta (24 dim), delta+dd (36 dim) "
                             f"(default: {DEFAULT_FEATURE_MODE})")
    parser.add_argument("--cmvn-window", type=int, default=DEFAULT_CMVN_WINDOW,
                        help=f"CMVN running window size (0=off, default: {DEFAULT_CMVN_WINDOW})")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON instead of table")
    args = parser.parse_args()

    print(f"=== Audio Pipeline Validation ===\n")
    print(f"Input: {args.audio}")

    # Load audio
    audio = load_audio(args.audio)
    duration_s = len(audio) / SAMPLE_RATE
    duration_ms = duration_s * 1000
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
        sil_db = 0 if args.no_debounce else args.silence_debounce_ms
        spe_db = 0 if args.no_debounce else args.speech_debounce_ms
        print(f"\nInitializing Silero VAD (threshold={args.vad_threshold}, "
              f"silence_debounce={sil_db}ms, speech_debounce={spe_db}ms)...")
        vad_fn = create_silero_vad(
            vad_threshold=args.vad_threshold,
            silence_debounce_ms=sil_db,
            speech_debounce_ms=spe_db,
        )

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
    pipeline = PipelineSimulator(vad_fn, speaker_oracle=oracle,
                                  smoothing_window=args.smoothing_window,
                                  speech_segment_frames=args.speech_buffer,
                                  b_confirm_frames=args.b_confirm_frames,
                                  feature_mode=args.feature_mode,
                                  cmvn_window=args.cmvn_window)
    pipeline_metrics = pipeline.run(audio)

    # Run speechbrain diarization if requested
    sb_timeline = None
    comparison = None
    if args.diarize:
        print(f"\nRunning speechbrain diarization...")
        try:
            diarizer = SpeechBrainDiarizer()
            vad_segments = extract_vad_segments(pipeline.timeline)
            print(f"  VAD segments for diarization: {len(vad_segments)}")
            sb_timeline = diarizer.diarize(audio, vad_segments)
            print(f"  SpeechBrain segments: {len(sb_timeline)}")
            comparison = compare_timelines(pipeline.timeline, sb_timeline, duration_ms)
            print(f"  Frame-level agreement: {comparison['agreement_pct']:.1f}%")
        except Exception as e:
            print(f"  SpeechBrain diarization failed: {e}")
            import traceback
            traceback.print_exc()

    # Run Whisper (if not skipped)
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
            "similarity_trace": [
                {"time_ms": t[0], "sim_a": round(t[1], 4), "sim_b": round(t[2], 4),
                 "decision": t[3], "ambiguous": t[4]}
                for t in pipeline.similarity_trace
            ],
            "timeline": [
                {"start_ms": s, "end_ms": e, "label": l}
                for s, e, l in pipeline.timeline
            ],
        }
        if gt_metrics:
            result["ground_truth_metrics"] = gt_metrics
        if comparison:
            result["comparison"] = comparison
        if sb_timeline:
            result["sb_timeline"] = [
                {"start_ms": s, "end_ms": e, "label": l}
                for s, e, l in sb_timeline
            ]
        if whisper_info:
            result["whisper"] = whisper_info
        print(json.dumps(result, indent=2))
    else:
        print_metrics_comparison(pipeline_metrics, gt_metrics, whisper_info, comparison)
        if not args.no_timeline:
            print_timeline(pipeline.timeline)
        print_similarity_trace_summary(pipeline.similarity_trace)

    print()


if __name__ == "__main__":
    main()
