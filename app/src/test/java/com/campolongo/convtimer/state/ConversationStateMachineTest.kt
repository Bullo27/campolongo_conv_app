package com.campolongo.convtimer.state

import com.campolongo.convtimer.audio.Speaker
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class ConversationStateMachineTest {

    private lateinit var sm: ConversationStateMachine

    @Before
    fun setUp() {
        sm = ConversationStateMachine()
    }

    // --- Init / Reset ---

    @Test
    fun initialStateIsIdle() {
        assertEquals(ConvState.IDLE, sm.state)
        assertNull(sm.lastActiveSpeaker)
    }

    @Test
    fun resetClearsEverything() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(100)
        sm.reset()
        assertEquals(ConvState.IDLE, sm.state)
        assertNull(sm.lastActiveSpeaker)
        val snap = sm.snapshot()
        assertEquals(0L, snap.trt)
        assertEquals(0L, snap.wta)
    }

    // --- Recording lifecycle ---

    @Test
    fun onRecordFromIdleGoesToInitialSilence() {
        sm.onRecord()
        assertEquals(ConvState.INITIAL_SILENCE, sm.state)
    }

    @Test
    fun onRecordFromStoppedGoesToInitialSilence() {
        sm.onRecord()
        sm.onStop()
        assertEquals(ConvState.STOPPED, sm.state)
        sm.onRecord()
        assertEquals(ConvState.INITIAL_SILENCE, sm.state)
    }

    @Test
    fun onRecordFromStoppedResetsMetrics() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(500)
        sm.onStop()
        sm.onRecord()
        assertEquals(0L, sm.snapshot().trt)
        assertEquals(0L, sm.snapshot().wta)
    }

    @Test
    fun onRecordIgnoredFromActiveStates() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
        sm.onRecord() // should be ignored
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
    }

    @Test
    fun onRecordIgnoredFromPaused() {
        sm.onRecord()
        sm.onPause()
        assertEquals(ConvState.PAUSED, sm.state)
        sm.onRecord()
        assertEquals(ConvState.PAUSED, sm.state)
    }

    // --- First speech ---

    @Test
    fun firstSpeechSpeakerATransitionsCorrectly() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
        assertEquals(Speaker.A, sm.lastActiveSpeaker)
    }

    @Test
    fun firstSpeechSpeakerBTransitionsCorrectly() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.B)
        assertEquals(ConvState.SPEAKER_B_TALKING, sm.state)
        assertEquals(Speaker.B, sm.lastActiveSpeaker)
    }

    // --- Direct speaker change ---

    @Test
    fun directChangeFromAToB() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSpeechDetected(Speaker.B)
        assertEquals(ConvState.SPEAKER_B_TALKING, sm.state)
        assertEquals(Speaker.B, sm.lastActiveSpeaker)
    }

    @Test
    fun directChangeFromBToA() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.B)
        sm.onSpeechDetected(Speaker.A)
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
        assertEquals(Speaker.A, sm.lastActiveSpeaker)
    }

    @Test
    fun sameSpeakerStaysInSameState() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSpeechDetected(Speaker.A)
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
    }

    // --- Silence → pending ---

    @Test
    fun silenceFromSpeakerAGoesToPending() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSilenceDetected()
        assertEquals(ConvState.PENDING_SILENCE, sm.state)
    }

    @Test
    fun silenceFromSpeakerBGoesToPending() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.B)
        sm.onSilenceDetected()
        assertEquals(ConvState.PENDING_SILENCE, sm.state)
    }

    @Test
    fun silenceIgnoredFromInitialSilence() {
        sm.onRecord()
        sm.onSilenceDetected()
        assertEquals(ConvState.INITIAL_SILENCE, sm.state)
    }

    @Test
    fun silenceIgnoredFromPending() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSilenceDetected()
        sm.onSilenceDetected() // second silence should be ignored
        assertEquals(ConvState.PENDING_SILENCE, sm.state)
    }

    // --- Pending resolution ---

    @Test
    fun pendingResolvedSameSpeakerGoesToSta() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSilenceDetected()
        sm.onTimeTick(200) // 200ms of pending silence
        sm.onSpeechDetected(Speaker.A)
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
        assertEquals(200L, sm.snapshot().sta)
        assertEquals(0L, sm.snapshot().stm)
    }

    @Test
    fun pendingResolvedSameSpeakerGoesToStb() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.B)
        sm.onSilenceDetected()
        sm.onTimeTick(150)
        sm.onSpeechDetected(Speaker.B)
        assertEquals(ConvState.SPEAKER_B_TALKING, sm.state)
        assertEquals(150L, sm.snapshot().stb)
        assertEquals(0L, sm.snapshot().stm)
    }

    @Test
    fun pendingResolvedDifferentSpeakerGoesToStm() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSilenceDetected()
        sm.onTimeTick(300)
        sm.onSpeechDetected(Speaker.B)
        assertEquals(ConvState.SPEAKER_B_TALKING, sm.state)
        assertEquals(300L, sm.snapshot().stm)
        assertEquals(0L, sm.snapshot().sta)
    }

    // --- Pause / Resume ---

    @Test
    fun pauseFromSpeakerAThenResumeRestores() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onPause()
        assertEquals(ConvState.PAUSED, sm.state)
        sm.onResume()
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
    }

    @Test
    fun pauseFromPendingSilenceThenResumeRestores() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSilenceDetected()
        sm.onPause()
        assertEquals(ConvState.PAUSED, sm.state)
        sm.onResume()
        assertEquals(ConvState.PENDING_SILENCE, sm.state)
    }

    @Test
    fun pauseIgnoredFromIdle() {
        sm.onPause()
        assertEquals(ConvState.IDLE, sm.state)
    }

    @Test
    fun pauseIgnoredFromStopped() {
        sm.onRecord()
        sm.onStop()
        sm.onPause()
        assertEquals(ConvState.STOPPED, sm.state)
    }

    @Test
    fun resumeIgnoredFromNonPaused() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onResume() // not paused, should be ignored
        assertEquals(ConvState.SPEAKER_A_TALKING, sm.state)
    }

    // --- Stop ---

    @Test
    fun stopFromSpeakingState() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onStop()
        assertEquals(ConvState.STOPPED, sm.state)
    }

    @Test
    fun stopFromPendingSilenceDiscardsPending() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(100) // 100ms talking
        sm.onSilenceDetected()
        sm.onTimeTick(200) // 200ms pending silence
        sm.onStop()
        assertEquals(ConvState.STOPPED, sm.state)
        // Pending silence was NOT resolved, so sta/stb/stm = 0
        // The pending time is in TRT but not in TCT → it's BFST
        val snap = sm.snapshot()
        assertEquals(0L, snap.sta)
        assertEquals(0L, snap.stm)
        assertEquals(300L, snap.trt)  // 100 + 200
        assertEquals(100L, snap.wta)
        assertEquals(200L, snap.bfst) // 300 - 100 = 200
    }

    @Test
    fun stopFromPausedWorks() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onPause()
        sm.onStop()
        assertEquals(ConvState.STOPPED, sm.state)
    }

    @Test
    fun stopIgnoredFromIdle() {
        sm.onStop()
        assertEquals(ConvState.IDLE, sm.state)
    }

    // --- Time tick ---

    @Test
    fun timeTickAccumulatesTrtInAllActiveStates() {
        sm.onRecord()
        sm.onTimeTick(100) // INITIAL_SILENCE
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(200) // SPEAKER_A_TALKING
        sm.onSilenceDetected()
        sm.onTimeTick(50) // PENDING_SILENCE
        assertEquals(350L, sm.snapshot().trt)
    }

    @Test
    fun timeTickAccumulatesWtaInSpeakerATalking() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(100)
        sm.onTimeTick(50)
        assertEquals(150L, sm.snapshot().wta)
    }

    @Test
    fun timeTickAccumulatesWtbInSpeakerBTalking() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.B)
        sm.onTimeTick(200)
        assertEquals(200L, sm.snapshot().wtb)
    }

    @Test
    fun timeTickInPendingSilenceAccumulatesPendingMs() {
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onSilenceDetected()
        sm.onTimeTick(100)
        sm.onTimeTick(100)
        sm.onSpeechDetected(Speaker.A)
        assertEquals(200L, sm.snapshot().sta)
    }

    @Test
    fun timeTickNoOpInIdlePausedStopped() {
        // IDLE
        sm.onTimeTick(100)
        assertEquals(0L, sm.snapshot().trt)

        // Start and pause
        sm.onRecord()
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(50)
        sm.onPause()
        sm.onTimeTick(999) // should be ignored
        assertEquals(50L, sm.snapshot().trt)

        sm.onResume()
        sm.onStop()
        sm.onTimeTick(999) // should be ignored
        assertEquals(50L, sm.snapshot().trt)
    }

    @Test
    fun timeTickInInitialSilenceOnlyAddsTrt() {
        sm.onRecord()
        sm.onTimeTick(500)
        val snap = sm.snapshot()
        assertEquals(500L, snap.trt)
        assertEquals(0L, snap.wta)
        assertEquals(0L, snap.wtb)
        assertEquals(0L, snap.tct)
        assertEquals(500L, snap.bfst) // all time is BFST
    }

    // --- Full conversation scenario ---

    @Test
    fun fullConversationScenario() {
        sm.onRecord()

        // Initial silence: 500ms
        sm.onTimeTick(500)

        // Speaker A talks for 1000ms
        sm.onSpeechDetected(Speaker.A)
        sm.onTimeTick(1000)

        // Silence gap, then same speaker (STA): 200ms
        sm.onSilenceDetected()
        sm.onTimeTick(200)
        sm.onSpeechDetected(Speaker.A)

        // Speaker A continues: 500ms
        sm.onTimeTick(500)

        // Silence gap, then different speaker (STM): 300ms
        sm.onSilenceDetected()
        sm.onTimeTick(300)
        sm.onSpeechDetected(Speaker.B)

        // Speaker B talks: 800ms
        sm.onTimeTick(800)

        // Silence gap, same speaker B (STB): 150ms
        sm.onSilenceDetected()
        sm.onTimeTick(150)
        sm.onSpeechDetected(Speaker.B)

        // Speaker B continues: 400ms
        sm.onTimeTick(400)

        sm.onStop()

        val snap = sm.snapshot()

        // TRT = 500 + 1000 + 200 + 500 + 300 + 800 + 150 + 400 = 3850
        assertEquals(3850L, snap.trt)

        // WTA = 1000 + 500 = 1500
        assertEquals(1500L, snap.wta)

        // WTB = 800 + 400 = 1200
        assertEquals(1200L, snap.wtb)

        // STA = 200 (within-A silence)
        assertEquals(200L, snap.sta)

        // STB = 150 (within-B silence)
        assertEquals(150L, snap.stb)

        // STM = 300 (between-speaker silence)
        assertEquals(300L, snap.stm)

        // CTA = WTA + STA = 1700
        assertEquals(1700L, snap.cta)

        // CTB = WTB + STB = 1350
        assertEquals(1350L, snap.ctb)

        // TCT = CTA + STM + CTB = 1700 + 300 + 1350 = 3350
        assertEquals(3350L, snap.tct)

        // TST = STA + STB + STM = 650
        assertEquals(650L, snap.tst)

        // BFST = TRT - TCT = 3850 - 3350 = 500 (the initial silence)
        assertEquals(500L, snap.bfst)
    }
}
