package com.campolongo.convtimer.audio

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import java.io.Closeable
import kotlin.math.min

/**
 * Orchestrates the audio processing pipeline:
 * AudioCapture -> VAD -> Fbank+ONNX -> SpeakerIdentifier -> Dual-assignment -> Events
 *
 * v2 neural pipeline: WeSpeaker ECAPA-TDNN-512 ONNX embeddings (192-dim),
 * dual-assignment overlap detection, adaptive noise tier selection.
 */
class AudioPipeline(
    private val scope: CoroutineScope,
) : Closeable {

    companion object {
        private const val TAG = "AudioPipeline"

        // Pipeline buffering
        private const val SPEECH_SEGMENT_FRAMES = 47  // ~1.5s of speech before speaker ID
        private const val MIN_FLUSH_FRAMES = 2        // min buffer to flush on silence

        // Dual-assignment overlap
        private const val DUAL_CAP = 3                // max consecutive dual-assignment segments
        private const val DUAL_T_LOW_NOISE = 0.82f    // Low Noise mode threshold
        private const val DUAL_T_ADAPTIVE_DEFAULT = 0.84f  // Adaptive Noise default
        private const val DUAL_T_HIGH = 1.0f          // High noise = overlap disabled

        // Adaptive noise EMA constants (only active in ADAPTIVE_NOISE mode)
        private const val ADAPTIVE_ALPHA = 0.0645f    // = 2/(30+1), N=30
        private const val NOISE_T_HIGH = 0.78f        // EMA threshold for HIGH tier
        private const val NOISE_MARGIN = 0.025f       // hysteresis band
        private const val NOISE_MIN_DWELL = 5         // min segments between tier changes
        private const val NOISE_WARMUP = 15           // segments after B before tier logic
    }

    private val audioCapture = AudioCaptureService()
    private val vadEngine = VadEngine()
    private val speakerIdentifier = SpeakerIdentifier()
    private val noiseCalibrator = NoiseCalibrator()
    private var onnxExtractor: OnnxEmbeddingExtractor? = null

    private val _events = MutableSharedFlow<AudioPipelineEvent>(extraBufferCapacity = 64)
    val events: SharedFlow<AudioPipelineEvent> = _events

    private var captureJob: Job? = null
    private var processingJob: Job? = null
    private val speechBuffer = mutableListOf<ShortArray>()

    // Overlap mode (user-selectable)
    var overlapMode: OverlapMode = OverlapMode.LOW_NOISE
        private set

    // Dual-assignment state
    private var consecDual = 0

    // Adaptive noise state
    private var noiseEma = 0f
    private var segsSinceTierChange = NOISE_MIN_DWELL  // allow immediate first transition
    private var currentTier = AdaptiveTier.LOW
    private var bEstablished = false
    private var segsSinceB = 0

    var noiseLevel: NoiseLevel = NoiseLevel.QUIET
        private set

    fun initialize(context: Context): Boolean {
        if (!audioCapture.initialize(context)) {
            Log.e(TAG, "Failed to initialize audio capture")
            return false
        }
        vadEngine.initialize(context)
        onnxExtractor = OnnxEmbeddingExtractor(context)
        Log.d(TAG, "ONNX embedding extractor initialized")
        return true
    }

    /**
     * Calibrate noise level by capturing ambient audio for ~1.5 seconds.
     */
    suspend fun calibrateNoise(context: Context): NoiseLevel {
        val frames = mutableListOf<ShortArray>()
        val collectJob = scope.launch(Dispatchers.IO) {
            audioCapture.frames.collect { frame ->
                frames.add(frame)
            }
        }

        val capJob = scope.launch(Dispatchers.IO) {
            audioCapture.startCapturing()
        }

        delay(1500)
        audioCapture.pauseCapturing()
        capJob.join()
        collectJob.cancel()

        noiseLevel = noiseCalibrator.calibrate(frames)
        Log.d(TAG, "Noise calibration result: ${noiseLevel.label}")

        // Re-initialize VAD with appropriate mode
        vadEngine.close()
        vadEngine.initialize(context, noiseLevel.vadMode)

        return noiseLevel
    }

    fun setNoiseLevel(context: Context, level: NoiseLevel) {
        noiseLevel = level
        vadEngine.close()
        vadEngine.initialize(context, level.vadMode)
        Log.d(TAG, "Noise level manually set to: ${level.label}")
    }

    fun setOverlapMode(mode: OverlapMode) {
        overlapMode = mode
        resetAdaptiveState()
        Log.d(TAG, "Overlap mode set to: $mode")
    }

    fun start() {
        speakerIdentifier.reset()
        consecDual = 0
        resetAdaptiveState()
        launchPipelineJobs()
        Log.d(TAG, "Pipeline started")
    }

    fun pause() {
        stopPipelineJobs()
        Log.d(TAG, "Pipeline paused")
    }

    fun resume() {
        launchPipelineJobs()
        Log.d(TAG, "Pipeline resumed")
    }

    fun stop() {
        stopPipelineJobs()
        Log.d(TAG, "Pipeline stopped")
    }

    private fun resetAdaptiveState() {
        noiseEma = 0f
        segsSinceTierChange = NOISE_MIN_DWELL
        currentTier = AdaptiveTier.LOW
        bEstablished = false
        segsSinceB = 0
        consecDual = 0
    }

    private fun launchPipelineJobs() {
        speechBuffer.clear()

        captureJob = scope.launch(Dispatchers.IO) {
            audioCapture.startCapturing()
        }

        processingJob = scope.launch(Dispatchers.IO) {
            audioCapture.frames.collect { frame ->
                processFrame(frame)
            }
        }
    }

    private fun stopPipelineJobs() {
        captureJob?.cancel()
        processingJob?.cancel()
        audioCapture.pauseCapturing()
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
            if (speechBuffer.size >= MIN_FLUSH_FRAMES) {
                identifyAndEmit()
            }
            speechBuffer.clear()
            _events.emit(AudioPipelineEvent.SilenceDetected)
        }
    }

    private suspend fun identifyAndEmit() {
        val extractor = onnxExtractor ?: return

        // 1. Concatenate speech buffer into single audio segment
        val nFrames = speechBuffer.size
        val rawAudio = concatenateBuffer(speechBuffer)
        speechBuffer.clear()

        // 2. Extract neural embedding
        val embedding = extractor.extractEmbedding(rawAudio)

        // 3. Identify primary speaker
        val primarySpeaker = speakerIdentifier.identify(embedding, nFrames)

        // 4. Get similarities for dual-assignment & adaptive noise
        val simA = speakerIdentifier.lastSimA
        val simB = speakerIdentifier.lastSimB

        // 5. Adaptive noise tier update (only in ADAPTIVE_NOISE mode)
        if (overlapMode == OverlapMode.ADAPTIVE_NOISE) {
            updateNoiseTier(simA, simB)
        }

        // 6. Dual-assignment overlap check
        val dualT = when (overlapMode) {
            OverlapMode.LOW_NOISE -> DUAL_T_LOW_NOISE
            OverlapMode.ADAPTIVE_NOISE -> when (currentTier) {
                AdaptiveTier.HIGH -> DUAL_T_HIGH
                AdaptiveTier.LOW -> DUAL_T_ADAPTIVE_DEFAULT
            }
        }
        val bothAbove = simA >= dualT && simB >= dualT
        val isDual = bothAbove && consecDual < DUAL_CAP

        if (isDual) {
            consecDual++
            _events.emit(AudioPipelineEvent.SpeechDetected(Speaker.BOTH))
        } else {
            if (!bothAbove) consecDual = 0
            // No smoothing (window=1): emit primary directly
            _events.emit(AudioPipelineEvent.SpeechDetected(primarySpeaker))
        }
    }

    private fun updateNoiseTier(simA: Float, simB: Float) {
        // Only start after B established
        if (!bEstablished) {
            if (simB > 0f && speakerIdentifier.isBEstablished) {
                bEstablished = true
                noiseEma = min(simA, simB)
            }
            return
        }

        // EMA update
        noiseEma = ADAPTIVE_ALPHA * min(simA, simB) + (1f - ADAPTIVE_ALPHA) * noiseEma

        // Warmup: skip tier decisions
        segsSinceB++
        if (segsSinceB < NOISE_WARMUP) return

        // Tier decision with hysteresis + min dwell
        segsSinceTierChange++
        if (segsSinceTierChange < NOISE_MIN_DWELL) return

        val newTier = when {
            currentTier != AdaptiveTier.HIGH && noiseEma > NOISE_T_HIGH + NOISE_MARGIN ->
                AdaptiveTier.HIGH
            currentTier == AdaptiveTier.HIGH && noiseEma < NOISE_T_HIGH - NOISE_MARGIN ->
                AdaptiveTier.LOW
            else -> currentTier
        }
        if (newTier != currentTier) {
            Log.d(TAG, "Noise tier: $currentTier -> $newTier (ema=%.4f)".format(noiseEma))
            currentTier = newTier
            segsSinceTierChange = 0
        }
    }

    private fun concatenateBuffer(buffer: List<ShortArray>): ShortArray {
        val total = buffer.sumOf { it.size }
        val result = ShortArray(total)
        var offset = 0
        for (frame in buffer) {
            frame.copyInto(result, offset)
            offset += frame.size
        }
        return result
    }

    override fun close() {
        stop()
        audioCapture.close()
        vadEngine.close()
        onnxExtractor?.close()
        Log.d(TAG, "AudioPipeline closed")
    }

    /** Internal adaptive noise tier. */
    private enum class AdaptiveTier { LOW, HIGH }
}
