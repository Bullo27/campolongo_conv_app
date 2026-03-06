package com.campolongo.convtimer.state

import com.campolongo.convtimer.audio.Speaker

enum class ConvState {
    IDLE,
    INITIAL_SILENCE,
    SPEAKER_A_TALKING,
    SPEAKER_B_TALKING,
    BOTH_TALKING,
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
            pendingSilenceMs = 0L
            pendingSilenceLastSpeaker = null
            state = ConvState.STOPPED
        }
    }

    // --- Audio pipeline events ---

    fun onSpeechDetected(speaker: Speaker) {
        when (state) {
            ConvState.INITIAL_SILENCE,
            ConvState.SPEAKER_A_TALKING,
            ConvState.SPEAKER_B_TALKING,
            ConvState.BOTH_TALKING -> {
                val newState = speakerToState(speaker)
                if (newState != state) {
                    state = newState
                    lastActiveSpeaker = primarySpeaker(speaker)
                }
            }
            ConvState.PENDING_SILENCE -> {
                // For silence classification, treat BOTH as continuation of lastActiveSpeaker
                val resolveAs = if (speaker == Speaker.BOTH) {
                    lastActiveSpeaker ?: Speaker.A
                } else {
                    speaker
                }
                resolvePendingSilence(resolveAs)
                state = speakerToState(speaker)
                lastActiveSpeaker = primarySpeaker(speaker)
            }
            else -> {} // Ignore in IDLE, PAUSED, STOPPED
        }
    }

    fun onSilenceDetected() {
        when (state) {
            ConvState.SPEAKER_A_TALKING, ConvState.SPEAKER_B_TALKING, ConvState.BOTH_TALKING -> {
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
            ConvState.BOTH_TALKING -> {
                // During overlap, credit both speakers + track overlap separately
                metrics.addWta(dtMs)
                metrics.addWtb(dtMs)
                metrics.addOvt(dtMs)
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
                Speaker.BOTH -> metrics.addStm(ms) // shouldn't normally happen
            }
        } else {
            // Between-speaker silence
            metrics.addStm(ms)
        }

        pendingSilenceMs = 0L
        pendingSilenceLastSpeaker = null
    }

    private fun speakerToState(speaker: Speaker): ConvState = when (speaker) {
        Speaker.A -> ConvState.SPEAKER_A_TALKING
        Speaker.B -> ConvState.SPEAKER_B_TALKING
        Speaker.BOTH -> ConvState.BOTH_TALKING
    }

    /** For lastActiveSpeaker tracking, BOTH keeps the previous speaker. */
    private fun primarySpeaker(speaker: Speaker): Speaker = when (speaker) {
        Speaker.BOTH -> lastActiveSpeaker ?: Speaker.A
        else -> speaker
    }
}
