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
        private const val MFCC_COEFFS = 13
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
     * Reuses the main AudioCaptureService to avoid device-specific issues
     * with concurrent AudioRecord instances.
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

    fun start() {
        speakerIdentifier.reset()
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
        // Don't clear speechBuffer here — avoids race with IO thread.
        // It is cleared in launchPipelineJobs() before restarting.
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
        // Compute MFCCs per frame and average — uses all buffered audio
        val avgMfcc = FloatArray(MFCC_COEFFS)
        for (frame in speechBuffer) {
            val frameMfcc = mfccExtractor.extract(frame)
            for (i in avgMfcc.indices) {
                avgMfcc[i] += frameMfcc[i]
            }
        }
        val count = speechBuffer.size
        speechBuffer.clear()

        if (count > 0) {
            for (i in avgMfcc.indices) {
                avgMfcc[i] /= count
            }
        }

        // Drop C0 (log energy) — it dominates cosine similarity and
        // masks the spectral-shape coefficients that distinguish speakers.
        val embedding = avgMfcc.copyOfRange(1, MFCC_COEFFS)
        val speaker = speakerIdentifier.identify(embedding)
        _events.emit(AudioPipelineEvent.SpeechDetected(speaker))
    }

    override fun close() {
        stop()
        audioCapture.close()
        vadEngine.close()
        Log.d(TAG, "AudioPipeline closed")
    }
}
