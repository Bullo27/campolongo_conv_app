package com.campolongo.convtimer.audio

sealed class AudioPipelineEvent {
    data class SpeechDetected(val speaker: Speaker) : AudioPipelineEvent()
    data object SilenceDetected : AudioPipelineEvent()
    data class Error(val message: String) : AudioPipelineEvent()
}

enum class Speaker { A, B }
