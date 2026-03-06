package com.campolongo.convtimer.viewmodel

import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.campolongo.convtimer.audio.AudioPipeline
import com.campolongo.convtimer.audio.AudioPipelineEvent
import com.campolongo.convtimer.audio.NoiseLevel
import com.campolongo.convtimer.audio.OverlapMode
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
    val overlapMode: OverlapMode = OverlapMode.LOW_NOISE,
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
        stopUiUpdates()
        _uiState.value = _uiState.value.copy(
            recordingState = RecordingState.PAUSED,
            metrics = stateMachine.snapshot()
        )
    }

    fun onResume() {
        stateMachine.onResume()
        pipeline.resume()
        startEventCollection()
        startTimeTick()
        startUiUpdates()
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

    fun onNoiseLevel(level: NoiseLevel) {
        val context = getApplication<Application>()
        pipeline.setNoiseLevel(context, level)
        _uiState.value = _uiState.value.copy(noiseLevel = level)
    }

    fun onOverlapMode(mode: OverlapMode) {
        pipeline.setOverlapMode(mode)
        _uiState.value = _uiState.value.copy(overlapMode = mode)
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
            var lastTick = SystemClock.elapsedRealtime()
            while (true) {
                delay(TICK_INTERVAL_MS)
                val now = SystemClock.elapsedRealtime()
                val elapsed = now - lastTick
                lastTick = now
                stateMachine.onTimeTick(elapsed)
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
                val newMetrics = stateMachine.snapshot()
                val current = _uiState.value
                if (newMetrics != current.metrics) {
                    _uiState.value = current.copy(metrics = newMetrics)
                }
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
