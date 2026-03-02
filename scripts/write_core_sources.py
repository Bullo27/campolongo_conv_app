#!/usr/bin/env python3
"""
Write all core audio, state machine, and speaker identification Kotlin sources.
Phases 2-4 of the implementation plan.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PKG = PROJECT_ROOT / "app" / "src" / "main" / "java" / "com" / "campolongo" / "convtimer"


def write_file(rel_path, content):
    path = PKG / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Wrote: {rel_path}")


def main():
    print("Writing core sources (Phases 2-4)...\n")

    # =========================================================================
    # AUDIO PACKAGE
    # =========================================================================

    write_file("audio/AudioPipelineEvent.kt", """\
package com.campolongo.convtimer.audio

sealed class AudioPipelineEvent {
    data class SpeechDetected(val speaker: Speaker) : AudioPipelineEvent()
    data object SilenceDetected : AudioPipelineEvent()
    data class Error(val message: String) : AudioPipelineEvent()
}

enum class Speaker { A, B }
""")

    write_file("audio/AudioCaptureService.kt", """\
package com.campolongo.convtimer.audio

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.Closeable

class AudioCaptureService : Closeable {

    companion object {
        const val SAMPLE_RATE = 16000
        const val FRAME_SIZE = 512 // samples per frame (~32ms at 16kHz)
        private const val TAG = "AudioCapture"
    }

    private var audioRecord: AudioRecord? = null
    private val _frames = MutableSharedFlow<ShortArray>(extraBufferCapacity = 64)
    val frames: SharedFlow<ShortArray> = _frames

    @Volatile
    var isCapturing = false
        private set

    fun initialize(context: android.content.Context): Boolean {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "RECORD_AUDIO permission not granted")
            return false
        }

        val bufferSize = maxOf(
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            ),
            FRAME_SIZE * 4 * 2 // at least 4 frames worth
        )

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        return audioRecord?.state == AudioRecord.STATE_INITIALIZED
    }

    suspend fun startCapturing() = withContext(Dispatchers.IO) {
        val recorder = audioRecord ?: return@withContext
        recorder.startRecording()
        isCapturing = true
        Log.d(TAG, "Started capturing audio")

        val buffer = ShortArray(FRAME_SIZE)
        while (isActive && isCapturing) {
            val read = recorder.read(buffer, 0, FRAME_SIZE)
            if (read == FRAME_SIZE) {
                _frames.emit(buffer.copyOf())
            } else if (read < 0) {
                Log.e(TAG, "AudioRecord.read error: $read")
                break
            }
        }
    }

    fun pauseCapturing() {
        isCapturing = false
        audioRecord?.stop()
        Log.d(TAG, "Paused capturing")
    }

    fun resumeCapturing() {
        // Will be restarted via startCapturing() coroutine
    }

    override fun close() {
        isCapturing = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        Log.d(TAG, "AudioCaptureService closed")
    }
}
""")

    write_file("audio/VadEngine.kt", """\
package com.campolongo.convtimer.audio

import android.content.Context
import android.util.Log
import com.konovalov.vad.silero.Vad
import com.konovalov.vad.silero.VadSilero
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import com.konovalov.vad.silero.config.SampleRate
import java.io.Closeable

class VadEngine : Closeable {

    companion object {
        private const val TAG = "VadEngine"
    }

    private var vad: VadSilero? = null

    fun initialize(context: Context, mode: Mode = Mode.NORMAL) {
        vad = Vad.builder()
            .setContext(context)
            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
            .setFrameSize(FrameSize.FRAME_SIZE_512)
            .setMode(mode)
            .setSilenceDurationMs(300)
            .setSpeechDurationMs(50)
            .build()
        Log.d(TAG, "VAD initialized with mode: $mode")
    }

    fun setMode(mode: Mode) {
        // VadSilero doesn't support changing mode after creation,
        // but we can access the internal threshold. For now, we'll
        // just log it - a full re-initialization would be needed.
        Log.d(TAG, "Mode change requested: $mode (requires re-init)")
    }

    fun isSpeech(frame: ShortArray): Boolean {
        return vad?.isSpeech(frame) ?: false
    }

    override fun close() {
        vad?.close()
        vad = null
        Log.d(TAG, "VadEngine closed")
    }
}
""")

    write_file("audio/NoiseCalibrator.kt", """\
package com.campolongo.convtimer.audio

import android.util.Log
import com.konovalov.vad.silero.config.Mode
import kotlin.math.log10
import kotlin.math.sqrt

enum class NoiseLevel(val label: String, val vadMode: Mode) {
    QUIET("Quiet", Mode.NORMAL),
    MODERATE("Moderate", Mode.NORMAL),      // Use NORMAL for moderate too — AGGRESSIVE may be too strict
    NOISY("Noisy", Mode.VERY_AGGRESSIVE);
}

class NoiseCalibrator {

    companion object {
        private const val TAG = "NoiseCalibrator"
        private const val QUIET_THRESHOLD_DBFS = -40.0
        private const val NOISY_THRESHOLD_DBFS = -25.0
    }

    fun calibrate(frames: List<ShortArray>): NoiseLevel {
        if (frames.isEmpty()) return NoiseLevel.QUIET

        // Compute RMS energy across all frames
        var sumSquares = 0.0
        var count = 0
        for (frame in frames) {
            for (sample in frame) {
                sumSquares += sample.toDouble() * sample.toDouble()
                count++
            }
        }

        val rms = sqrt(sumSquares / count)
        // Convert to dBFS (16-bit audio, max value = 32767)
        val dbfs = if (rms > 0) 20.0 * log10(rms / 32767.0) else -96.0

        Log.d(TAG, "Calibration: RMS=${"%.1f".format(rms)}, dBFS=${"%.1f".format(dbfs)}")

        return when {
            dbfs < QUIET_THRESHOLD_DBFS -> NoiseLevel.QUIET
            dbfs < NOISY_THRESHOLD_DBFS -> NoiseLevel.MODERATE
            else -> NoiseLevel.NOISY
        }
    }
}
""")

    write_file("audio/MfccExtractor.kt", """\
package com.campolongo.convtimer.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin MFCC feature extraction.
 * Extracts 13 MFCC coefficients from a short audio segment.
 */
class MfccExtractor(
    private val sampleRate: Int = 16000,
    private val numCoeffs: Int = 13,
    private val numFilters: Int = 26,
    private val fftSize: Int = 512,
) {
    private val melFilterbank: Array<FloatArray> = createMelFilterbank()

    fun extract(samples: ShortArray): FloatArray {
        // Convert to float and apply pre-emphasis
        val signal = FloatArray(samples.size)
        signal[0] = samples[0].toFloat()
        for (i in 1 until samples.size) {
            signal[i] = samples[i].toFloat() - 0.97f * samples[i - 1].toFloat()
        }

        // Take center portion or pad to fftSize
        val frame = FloatArray(fftSize)
        val copyLen = minOf(signal.size, fftSize)
        val offset = maxOf(0, (signal.size - fftSize) / 2)
        for (i in 0 until copyLen) {
            frame[i] = signal[offset + i]
        }

        // Apply Hamming window
        for (i in frame.indices) {
            frame[i] *= (0.54f - 0.46f * cos(2.0 * PI * i / (fftSize - 1)).toFloat())
        }

        // Compute FFT and power spectrum
        val powerSpectrum = computePowerSpectrum(frame)

        // Apply Mel filterbank
        val melEnergies = FloatArray(numFilters)
        for (i in 0 until numFilters) {
            var sum = 0f
            for (j in powerSpectrum.indices) {
                sum += melFilterbank[i][j] * powerSpectrum[j]
            }
            melEnergies[i] = if (sum > 1e-10f) ln(sum.toDouble()).toFloat() else -23.0f // ln(1e-10)
        }

        // Apply DCT to get MFCCs
        val mfcc = FloatArray(numCoeffs)
        for (i in 0 until numCoeffs) {
            var sum = 0f
            for (j in 0 until numFilters) {
                sum += melEnergies[j] * cos(PI * i * (j + 0.5) / numFilters).toFloat()
            }
            mfcc[i] = sum
        }

        return mfcc
    }

    private fun computePowerSpectrum(frame: FloatArray): FloatArray {
        // Radix-2 Cooley-Tukey FFT
        val n = frame.size
        val real = frame.copyOf()
        val imag = FloatArray(n)

        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = real[i]; real[i] = real[j]; real[j] = tempR
                val tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // FFT butterfly
        var step = 1
        while (step < n) {
            val angleStep = -PI / step
            for (group in 0 until n step step * 2) {
                for (pair in 0 until step) {
                    val angle = angleStep * pair
                    val wr = cos(angle).toFloat()
                    val wi = sin(angle).toFloat()
                    val idx1 = group + pair
                    val idx2 = idx1 + step
                    val tr = wr * real[idx2] - wi * imag[idx2]
                    val ti = wr * imag[idx2] + wi * real[idx2]
                    real[idx2] = real[idx1] - tr
                    imag[idx2] = imag[idx1] - ti
                    real[idx1] += tr
                    imag[idx1] += ti
                }
            }
            step *= 2
        }

        // Power spectrum (only first half + 1)
        val specSize = n / 2 + 1
        val power = FloatArray(specSize)
        for (i in 0 until specSize) {
            power[i] = (real[i] * real[i] + imag[i] * imag[i]) / n
        }
        return power
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val specSize = fftSize / 2 + 1
        val lowMel = hzToMel(0f)
        val highMel = hzToMel(sampleRate / 2f)

        // Create equally spaced points in Mel scale
        val melPoints = FloatArray(numFilters + 2)
        for (i in melPoints.indices) {
            melPoints[i] = lowMel + i * (highMel - lowMel) / (numFilters + 1)
        }

        // Convert back to Hz and then to FFT bin indices
        val binPoints = IntArray(melPoints.size)
        for (i in melPoints.indices) {
            val hz = melToHz(melPoints[i])
            binPoints[i] = floor(hz * (fftSize + 1) / sampleRate).toInt()
        }

        // Create triangular filters
        val filterbank = Array(numFilters) { FloatArray(specSize) }
        for (i in 0 until numFilters) {
            for (j in binPoints[i] until binPoints[i + 1]) {
                if (j < specSize) {
                    filterbank[i][j] = (j - binPoints[i]).toFloat() /
                            maxOf(1, binPoints[i + 1] - binPoints[i])
                }
            }
            for (j in binPoints[i + 1] until binPoints[i + 2]) {
                if (j < specSize) {
                    filterbank[i][j] = (binPoints[i + 2] - j).toFloat() /
                            maxOf(1, binPoints[i + 2] - binPoints[i + 1])
                }
            }
        }
        return filterbank
    }

    private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)
}
""")

    write_file("audio/SpeakerIdentifier.kt", """\
package com.campolongo.convtimer.audio

import android.util.Log
import kotlin.math.sqrt

/**
 * Identifies speakers by comparing MFCC embeddings via cosine similarity.
 * First speaker detected = Speaker A; a distinct new voice = Speaker B.
 */
class SpeakerIdentifier(
    private val similarityThreshold: Float = 0.80f,
    private val ambiguityMargin: Float = 0.10f, // hysteresis zone: threshold +/- margin
    private val emaAlpha: Float = 0.1f, // exponential moving average update rate
) {
    companion object {
        private const val TAG = "SpeakerID"
    }

    private var speakerARef: FloatArray? = null
    private var speakerBRef: FloatArray? = null
    private var lastSpeaker: Speaker? = null

    fun identify(mfcc: FloatArray): Speaker {
        val refA = speakerARef
        if (refA == null) {
            // First speech ever — this is Speaker A
            speakerARef = mfcc.copyOf()
            lastSpeaker = Speaker.A
            Log.d(TAG, "Speaker A reference established")
            return Speaker.A
        }

        val simA = cosineSimilarity(mfcc, refA)
        val refB = speakerBRef

        if (refB == null) {
            // Speaker B not yet established
            if (simA >= similarityThreshold) {
                // Still Speaker A
                updateReference(Speaker.A, mfcc)
                lastSpeaker = Speaker.A
                return Speaker.A
            } else {
                // New speaker detected — this is Speaker B
                speakerBRef = mfcc.copyOf()
                lastSpeaker = Speaker.B
                Log.d(TAG, "Speaker B reference established (simA=%.3f)".format(simA))
                return Speaker.B
            }
        }

        // Both speakers established — classify
        val simB = cosineSimilarity(mfcc, refB)
        Log.v(TAG, "simA=%.3f, simB=%.3f".format(simA, simB))

        val speaker = when {
            // Clear Speaker A
            simA > simB + ambiguityMargin -> Speaker.A
            // Clear Speaker B
            simB > simA + ambiguityMargin -> Speaker.B
            // Ambiguous — keep previous speaker (hysteresis)
            else -> lastSpeaker ?: Speaker.A
        }

        updateReference(speaker, mfcc)
        lastSpeaker = speaker
        return speaker
    }

    fun reset() {
        speakerARef = null
        speakerBRef = null
        lastSpeaker = null
    }

    private fun updateReference(speaker: Speaker, mfcc: FloatArray) {
        val ref = if (speaker == Speaker.A) speakerARef else speakerBRef
        if (ref != null) {
            // EMA update
            for (i in ref.indices) {
                ref[i] = (1 - emaAlpha) * ref[i] + emaAlpha * mfcc[i]
            }
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom > 0f) dot / denom else 0f
    }
}
""")

    write_file("audio/AudioPipeline.kt", """\
package com.campolongo.convtimer.audio

import android.content.Context
import android.util.Log
import com.konovalov.vad.silero.config.Mode
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import java.io.Closeable

/**
 * Orchestrates the audio processing pipeline:
 * AudioCapture -> VAD -> MFCC -> SpeakerIdentifier -> Events
 */
class AudioPipeline(
    private val scope: CoroutineScope,
) : Closeable {

    companion object {
        private const val TAG = "AudioPipeline"
        private const val SPEECH_SEGMENT_FRAMES = 5 // ~160ms of speech before speaker ID
    }

    private val audioCapture = AudioCaptureService()
    private val vadEngine = VadEngine()
    private val mfccExtractor = MfccExtractor()
    private val speakerIdentifier = SpeakerIdentifier()
    private val noiseCalibrator = NoiseCalibrator()

    private val _events = MutableSharedFlow<AudioPipelineEvent>(extraBufferCapacity = 64)
    val events: SharedFlow<AudioPipelineEvent> = _events

    private var captureJob: Job? = null
    private var processingJob: Job? = null
    private val speechBuffer = mutableListOf<ShortArray>()

    var noiseLevel: NoiseLevel = NoiseLevel.QUIET
        private set

    fun initialize(context: Context): Boolean {
        if (!audioCapture.initialize(context)) {
            Log.e(TAG, "Failed to initialize audio capture")
            return false
        }
        vadEngine.initialize(context)
        return true
    }

    /**
     * Calibrate noise level by capturing ambient audio for ~1.5 seconds.
     * Call this before starting recording.
     */
    suspend fun calibrateNoise(context: Context): NoiseLevel {
        val calibrationCapture = AudioCaptureService()
        if (!calibrationCapture.initialize(context)) {
            return NoiseLevel.QUIET
        }

        val frames = mutableListOf<ShortArray>()
        val collectJob = scope.launch(Dispatchers.IO) {
            calibrationCapture.frames.collect { frame ->
                frames.add(frame)
            }
        }

        // Capture for ~1.5 seconds
        val captureJob = scope.launch(Dispatchers.IO) {
            calibrationCapture.startCapturing()
        }

        kotlinx.coroutines.delay(1500)
        calibrationCapture.close()
        captureJob.cancel()
        collectJob.cancel()

        noiseLevel = noiseCalibrator.calibrate(frames)
        Log.d(TAG, "Noise calibration result: ${noiseLevel.label}")

        // Re-initialize VAD with appropriate mode
        vadEngine.close()
        vadEngine.initialize(context, noiseLevel.vadMode)

        return noiseLevel
    }

    fun start() {
        speechBuffer.clear()
        speakerIdentifier.reset()

        captureJob = scope.launch(Dispatchers.IO) {
            audioCapture.startCapturing()
        }

        processingJob = scope.launch(Dispatchers.IO) {
            audioCapture.frames.collect { frame ->
                processFrame(frame)
            }
        }

        Log.d(TAG, "Pipeline started")
    }

    fun pause() {
        captureJob?.cancel()
        processingJob?.cancel()
        audioCapture.pauseCapturing()
        speechBuffer.clear()
        Log.d(TAG, "Pipeline paused")
    }

    fun resume(context: Context) {
        speechBuffer.clear()

        captureJob = scope.launch(Dispatchers.IO) {
            audioCapture.startCapturing()
        }

        processingJob = scope.launch(Dispatchers.IO) {
            audioCapture.frames.collect { frame ->
                processFrame(frame)
            }
        }

        Log.d(TAG, "Pipeline resumed")
    }

    fun stop() {
        captureJob?.cancel()
        processingJob?.cancel()
        audioCapture.pauseCapturing()
        speechBuffer.clear()
        Log.d(TAG, "Pipeline stopped")
    }

    private suspend fun processFrame(frame: ShortArray) {
        val isSpeech = vadEngine.isSpeech(frame)

        if (isSpeech) {
            speechBuffer.add(frame)
            if (speechBuffer.size >= SPEECH_SEGMENT_FRAMES) {
                identifyAndEmit()
            }
        } else {
            // Flush any accumulated speech
            if (speechBuffer.size >= 2) {
                identifyAndEmit()
            }
            speechBuffer.clear()
            _events.emit(AudioPipelineEvent.SilenceDetected)
        }
    }

    private suspend fun identifyAndEmit() {
        // Concatenate buffered frames
        val totalSamples = speechBuffer.sumOf { it.size }
        val segment = ShortArray(totalSamples)
        var offset = 0
        for (frame in speechBuffer) {
            frame.copyInto(segment, offset)
            offset += frame.size
        }
        speechBuffer.clear()

        // Extract MFCC and identify speaker
        val mfcc = mfccExtractor.extract(segment)
        val speaker = speakerIdentifier.identify(mfcc)
        _events.emit(AudioPipelineEvent.SpeechDetected(speaker))
    }

    override fun close() {
        stop()
        audioCapture.close()
        vadEngine.close()
        Log.d(TAG, "AudioPipeline closed")
    }
}
""")

    # =========================================================================
    # STATE PACKAGE
    # =========================================================================

    write_file("state/MetricsAccumulator.kt", """\
package com.campolongo.convtimer.state

import java.util.concurrent.atomic.AtomicLong

data class MetricsSnapshot(
    val trt: Long = 0L,
    val wta: Long = 0L,
    val wtb: Long = 0L,
    val sta: Long = 0L,
    val stb: Long = 0L,
    val stm: Long = 0L,
) {
    val cta: Long get() = wta + sta
    val ctb: Long get() = wtb + stb
    val tct: Long get() = cta + stm + ctb
    val tst: Long get() = sta + stb + stm
    val bfst: Long get() = trt - tct
}

class MetricsAccumulator {
    private val _trt = AtomicLong(0L)
    private val _wta = AtomicLong(0L)
    private val _wtb = AtomicLong(0L)
    private val _sta = AtomicLong(0L)
    private val _stb = AtomicLong(0L)
    private val _stm = AtomicLong(0L)

    fun addTrt(dt: Long) { _trt.addAndGet(dt) }
    fun addWta(dt: Long) { _wta.addAndGet(dt) }
    fun addWtb(dt: Long) { _wtb.addAndGet(dt) }
    fun addSta(dt: Long) { _sta.addAndGet(dt) }
    fun addStb(dt: Long) { _stb.addAndGet(dt) }
    fun addStm(dt: Long) { _stm.addAndGet(dt) }

    fun snapshot(): MetricsSnapshot = MetricsSnapshot(
        trt = _trt.get(),
        wta = _wta.get(),
        wtb = _wtb.get(),
        sta = _sta.get(),
        stb = _stb.get(),
        stm = _stm.get(),
    )

    fun reset() {
        _trt.set(0L)
        _wta.set(0L)
        _wtb.set(0L)
        _sta.set(0L)
        _stb.set(0L)
        _stm.set(0L)
    }
}
""")

    write_file("state/ConversationState.kt", """\
package com.campolongo.convtimer.state

import com.campolongo.convtimer.audio.Speaker

enum class ConvState {
    IDLE,
    INITIAL_SILENCE,
    SPEAKER_A_TALKING,
    SPEAKER_B_TALKING,
    PENDING_SILENCE,
    PAUSED,
    STOPPED,
}

/**
 * Finite state machine for conversation tracking.
 * Drives metrics accumulation based on audio pipeline events.
 */
class ConversationStateMachine {

    var state: ConvState = ConvState.IDLE
        private set

    var lastActiveSpeaker: Speaker? = null
        private set

    // The state we were in before pausing (to restore on resume)
    private var stateBeforePause: ConvState = ConvState.IDLE

    // Pending silence accumulation
    private var pendingSilenceMs: Long = 0L
    private var pendingSilenceLastSpeaker: Speaker? = null

    private val metrics = MetricsAccumulator()

    fun snapshot(): MetricsSnapshot = metrics.snapshot()

    fun reset() {
        state = ConvState.IDLE
        lastActiveSpeaker = null
        stateBeforePause = ConvState.IDLE
        pendingSilenceMs = 0L
        pendingSilenceLastSpeaker = null
        metrics.reset()
    }

    // --- User actions ---

    fun onRecord() {
        if (state == ConvState.IDLE || state == ConvState.STOPPED) {
            reset()
            state = ConvState.INITIAL_SILENCE
        }
    }

    fun onPause() {
        if (state != ConvState.IDLE && state != ConvState.STOPPED && state != ConvState.PAUSED) {
            stateBeforePause = state
            state = ConvState.PAUSED
        }
    }

    fun onResume() {
        if (state == ConvState.PAUSED) {
            state = stateBeforePause
        }
    }

    fun onStop() {
        if (state != ConvState.IDLE && state != ConvState.STOPPED) {
            // Resolve any pending silence as BFST (final silence)
            // BFST is derived as TRT - TCT, so pending silence just stays unresolved
            // (it's already counted in TRT but not in any conversation metric)
            pendingSilenceMs = 0L
            pendingSilenceLastSpeaker = null
            state = ConvState.STOPPED
        }
    }

    // --- Audio pipeline events ---

    fun onSpeechDetected(speaker: Speaker) {
        when (state) {
            ConvState.INITIAL_SILENCE -> {
                // First speech — transition to speaking state
                state = if (speaker == Speaker.A) ConvState.SPEAKER_A_TALKING else ConvState.SPEAKER_B_TALKING
                lastActiveSpeaker = speaker
            }
            ConvState.SPEAKER_A_TALKING -> {
                if (speaker == Speaker.B) {
                    // Direct speaker change (no silence gap)
                    state = ConvState.SPEAKER_B_TALKING
                    lastActiveSpeaker = Speaker.B
                }
                // If still Speaker A, stay in SPEAKER_A_TALKING
            }
            ConvState.SPEAKER_B_TALKING -> {
                if (speaker == Speaker.A) {
                    state = ConvState.SPEAKER_A_TALKING
                    lastActiveSpeaker = Speaker.A
                }
            }
            ConvState.PENDING_SILENCE -> {
                // Resolve pending silence
                resolvePendingSilence(speaker)
                state = if (speaker == Speaker.A) ConvState.SPEAKER_A_TALKING else ConvState.SPEAKER_B_TALKING
                lastActiveSpeaker = speaker
            }
            else -> {} // Ignore in IDLE, PAUSED, STOPPED
        }
    }

    fun onSilenceDetected() {
        when (state) {
            ConvState.SPEAKER_A_TALKING, ConvState.SPEAKER_B_TALKING -> {
                pendingSilenceLastSpeaker = lastActiveSpeaker
                pendingSilenceMs = 0L
                state = ConvState.PENDING_SILENCE
            }
            else -> {} // Stay in current state
        }
    }

    // --- Time tick (called periodically, e.g., every ~32ms) ---

    fun onTimeTick(dtMs: Long) {
        if (state == ConvState.PAUSED || state == ConvState.IDLE || state == ConvState.STOPPED) {
            return
        }

        metrics.addTrt(dtMs)

        when (state) {
            ConvState.INITIAL_SILENCE -> {
                // BFST is derived as TRT - TCT, no explicit accumulation needed
            }
            ConvState.SPEAKER_A_TALKING -> {
                metrics.addWta(dtMs)
            }
            ConvState.SPEAKER_B_TALKING -> {
                metrics.addWtb(dtMs)
            }
            ConvState.PENDING_SILENCE -> {
                pendingSilenceMs += dtMs
            }
            else -> {}
        }
    }

    private fun resolvePendingSilence(nextSpeaker: Speaker) {
        val prevSpeaker = pendingSilenceLastSpeaker ?: return
        val ms = pendingSilenceMs

        if (nextSpeaker == prevSpeaker) {
            // Within-speaker silence
            when (prevSpeaker) {
                Speaker.A -> metrics.addSta(ms)
                Speaker.B -> metrics.addStb(ms)
            }
        } else {
            // Between-speaker silence
            metrics.addStm(ms)
        }

        pendingSilenceMs = 0L
        pendingSilenceLastSpeaker = null
    }
}
""")

    # =========================================================================
    # UPDATED VIEWMODEL
    # =========================================================================

    write_file("viewmodel/RecordingViewModel.kt", """\
package com.campolongo.convtimer.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.campolongo.convtimer.audio.AudioPipeline
import com.campolongo.convtimer.audio.AudioPipelineEvent
import com.campolongo.convtimer.audio.NoiseLevel
import com.campolongo.convtimer.state.ConversationStateMachine
import com.campolongo.convtimer.state.MetricsSnapshot
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

enum class RecordingState {
    IDLE, CALIBRATING, RECORDING, PAUSED, STOPPED
}

data class UiState(
    val recordingState: RecordingState = RecordingState.IDLE,
    val metrics: MetricsSnapshot = MetricsSnapshot(),
    val noiseLevel: NoiseLevel = NoiseLevel.QUIET,
    val permissionNeeded: Boolean = false,
)

class RecordingViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "RecordingVM"
        private const val TICK_INTERVAL_MS = 32L // ~one audio frame period
        private const val UI_UPDATE_INTERVAL_MS = 100L
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val pipeline = AudioPipeline(viewModelScope)
    private val stateMachine = ConversationStateMachine()

    private var tickJob: Job? = null
    private var eventCollectorJob: Job? = null
    private var uiUpdateJob: Job? = null
    private var pipelineInitialized = false

    fun onPermissionGranted() {
        val context = getApplication<Application>()
        pipelineInitialized = pipeline.initialize(context)
        if (!pipelineInitialized) {
            Log.e(TAG, "Failed to initialize audio pipeline")
        }
        _uiState.value = _uiState.value.copy(permissionNeeded = false)
    }

    fun onPermissionDenied() {
        _uiState.value = _uiState.value.copy(permissionNeeded = false)
    }

    fun onRecord() {
        if (!pipelineInitialized) {
            _uiState.value = _uiState.value.copy(permissionNeeded = true)
            return
        }

        val context = getApplication<Application>()

        // First calibrate noise, then start recording
        _uiState.value = _uiState.value.copy(recordingState = RecordingState.CALIBRATING)

        viewModelScope.launch {
            val noise = pipeline.calibrateNoise(context)
            _uiState.value = _uiState.value.copy(noiseLevel = noise)

            // Now start recording
            stateMachine.onRecord()
            pipeline.start()
            startEventCollection()
            startTimeTick()
            startUiUpdates()
            _uiState.value = _uiState.value.copy(recordingState = RecordingState.RECORDING)
        }
    }

    fun onPause() {
        stateMachine.onPause()
        pipeline.pause()
        stopTimeTick()
        stopEventCollection()
        _uiState.value = _uiState.value.copy(
            recordingState = RecordingState.PAUSED,
            metrics = stateMachine.snapshot()
        )
    }

    fun onResume() {
        val context = getApplication<Application>()
        stateMachine.onResume()
        pipeline.resume(context)
        startEventCollection()
        startTimeTick()
        _uiState.value = _uiState.value.copy(recordingState = RecordingState.RECORDING)
    }

    fun onStop() {
        stateMachine.onStop()
        pipeline.stop()
        stopTimeTick()
        stopEventCollection()
        stopUiUpdates()
        _uiState.value = _uiState.value.copy(
            recordingState = RecordingState.STOPPED,
            metrics = stateMachine.snapshot()
        )
    }

    private fun startEventCollection() {
        eventCollectorJob = viewModelScope.launch {
            pipeline.events.collect { event ->
                when (event) {
                    is AudioPipelineEvent.SpeechDetected -> {
                        stateMachine.onSpeechDetected(event.speaker)
                    }
                    is AudioPipelineEvent.SilenceDetected -> {
                        stateMachine.onSilenceDetected()
                    }
                    is AudioPipelineEvent.Error -> {
                        Log.e(TAG, "Pipeline error: ${event.message}")
                    }
                }
            }
        }
    }

    private fun stopEventCollection() {
        eventCollectorJob?.cancel()
        eventCollectorJob = null
    }

    private fun startTimeTick() {
        tickJob = viewModelScope.launch {
            while (true) {
                delay(TICK_INTERVAL_MS)
                stateMachine.onTimeTick(TICK_INTERVAL_MS)
            }
        }
    }

    private fun stopTimeTick() {
        tickJob?.cancel()
        tickJob = null
    }

    private fun startUiUpdates() {
        uiUpdateJob = viewModelScope.launch {
            while (true) {
                delay(UI_UPDATE_INTERVAL_MS)
                _uiState.value = _uiState.value.copy(metrics = stateMachine.snapshot())
            }
        }
    }

    private fun stopUiUpdates() {
        uiUpdateJob?.cancel()
        uiUpdateJob = null
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
    }
}
""")

    # =========================================================================
    # UPDATED RECORDING SCREEN (with permission handling + noise level)
    # =========================================================================

    write_file("ui/screen/RecordingScreen.kt", """\
package com.campolongo.convtimer.ui.screen

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.campolongo.convtimer.audio.NoiseLevel
import com.campolongo.convtimer.ui.component.ControlButtons
import com.campolongo.convtimer.ui.component.MetricsGrid
import com.campolongo.convtimer.ui.component.MicrophoneIcon
import com.campolongo.convtimer.ui.component.RecordingTimeDisplay
import com.campolongo.convtimer.viewmodel.RecordingState
import com.campolongo.convtimer.viewmodel.RecordingViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordingScreen(viewModel: RecordingViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            viewModel.onPermissionGranted()
            viewModel.onRecord()
        } else {
            viewModel.onPermissionDenied()
        }
    }

    LaunchedEffect(uiState.permissionNeeded) {
        if (uiState.permissionNeeded) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Voice Recorder",
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                }
            )
        }
    ) { paddingValues ->
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            Spacer(modifier = Modifier.height(16.dp))

            MicrophoneIcon()

            Spacer(modifier = Modifier.height(8.dp))

            if (uiState.recordingState == RecordingState.CALIBRATING) {
                Text(
                    text = "Calibrating noise level...",
                    fontSize = 16.sp,
                    color = com.campolongo.convtimer.ui.theme.PauseOrange
                )
            } else {
                RecordingTimeDisplay(trtMs = uiState.metrics.trt)
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Noise level indicator
            if (uiState.recordingState != RecordingState.IDLE) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Noise:", fontSize = 12.sp)
                    NoiseLevel.entries.forEach { level ->
                        FilterChip(
                            selected = uiState.noiseLevel == level,
                            onClick = { /* TODO: allow user to change noise level */ },
                            label = { Text(level.label, fontSize = 11.sp) }
                        )
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            ControlButtons(
                recordingState = uiState.recordingState,
                onRecord = viewModel::onRecord,
                onPause = viewModel::onPause,
                onResume = viewModel::onResume,
                onStop = viewModel::onStop
            )

            Spacer(modifier = Modifier.height(24.dp))

            MetricsGrid(
                metrics = uiState.metrics,
                modifier = Modifier.weight(1f)
            )
        }
    }
}
""")

    # Update ControlButtons to handle CALIBRATING state
    write_file("ui/component/ControlButtons.kt", """\
package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.campolongo.convtimer.ui.theme.RecordGreen
import com.campolongo.convtimer.ui.theme.PauseOrange
import com.campolongo.convtimer.ui.theme.ResumeBlue
import com.campolongo.convtimer.ui.theme.StopRed
import com.campolongo.convtimer.viewmodel.RecordingState

@Composable
fun ControlButtons(
    recordingState: RecordingState,
    onRecord: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            ActionButton(
                text = "Record",
                color = RecordGreen,
                enabled = recordingState == RecordingState.IDLE || recordingState == RecordingState.STOPPED,
                onClick = onRecord
            )
            ActionButton(
                text = "Pause",
                color = PauseOrange,
                enabled = recordingState == RecordingState.RECORDING,
                onClick = onPause
            )
            ActionButton(
                text = "Resume",
                color = ResumeBlue,
                enabled = recordingState == RecordingState.PAUSED,
                onClick = onResume
            )
        }
        ActionButton(
            text = "Stop",
            color = StopRed,
            enabled = recordingState == RecordingState.RECORDING || recordingState == RecordingState.PAUSED,
            onClick = onStop
        )
    }
}

@Composable
private fun ActionButton(
    text: String,
    color: Color,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        colors = ButtonDefaults.buttonColors(containerColor = color),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.height(40.dp)
    ) {
        Text(text = text, fontSize = 14.sp, color = Color.White)
    }
}
""")

    # Update MainActivity to use AndroidViewModel
    write_file("MainActivity.kt", """\
package com.campolongo.convtimer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.campolongo.convtimer.ui.theme.ConversationTimerTheme
import com.campolongo.convtimer.ui.screen.RecordingScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ConversationTimerTheme {
                RecordingScreen()
            }
        }
    }
}
""")

    print("\nAll core sources written!")
    print("Run build_app.py to verify compilation.")


if __name__ == "__main__":
    main()
