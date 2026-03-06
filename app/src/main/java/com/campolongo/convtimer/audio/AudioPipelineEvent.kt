package com.campolongo.convtimer.audio

sealed class AudioPipelineEvent {
    data class SpeechDetected(val speaker: Speaker) : AudioPipelineEvent()
    data object SilenceDetected : AudioPipelineEvent()
    data class Error(val message: String) : AudioPipelineEvent()
}

enum class Speaker { A, B, BOTH }

/** User-selectable overlap detection mode. */
enum class OverlapMode {
    /** Fixed dual_t=0.82, best overlap accuracy, no adaptive noise logic. */
    LOW_NOISE,
    /** Default dual_t=0.84, EMA auto-escalates to 1.0 in severe noise. */
    ADAPTIVE_NOISE,
}
