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
            // Pending silence is left unresolved — it's already counted in TRT
            // but not in any conversation metric, so it falls into BFST (= TRT - TCT)
            // as final silence, which is the correct semantic.
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
