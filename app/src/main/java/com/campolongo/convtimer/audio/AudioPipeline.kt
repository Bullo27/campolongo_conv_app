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

    fun setNoiseLevel(context: Context, level: NoiseLevel) {
        noiseLevel = level
        vadEngine.close()
        vadEngine.initialize(context, level.vadMode)
        Log.d(TAG, "Noise level manually set to: ${level.label}")
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
