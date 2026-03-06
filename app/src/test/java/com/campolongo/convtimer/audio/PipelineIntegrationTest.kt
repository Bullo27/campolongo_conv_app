package com.campolongo.convtimer.audio

import com.campolongo.convtimer.state.ConversationStateMachine
import com.campolongo.convtimer.state.ConvState
import com.campolongo.convtimer.state.MetricsSnapshot
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

/**
 * Integration tests that feed synthetic audio through scripted pipeline decisions
 * into the ConversationStateMachine.
 *
 * The neural pipeline (Fbank+ONNX) is NOT tested here (requires Android context/model).
 * Speaker decisions use a scripted sequence to test timing logic deterministically.
 *
 * Each frame = 512 samples @ 16kHz = 32ms.
 */
class PipelineIntegrationTest {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val FRAME_SIZE = 512
        private const val FRAME_MS = 32L
        private const val SPEECH_SEGMENT_FRAMES = 47 // v2 neural pipeline buffer
        private const val MIN_FLUSH_FRAMES = 2
        private const val AMPLITUDE = 16000.0
    }

    private lateinit var stateMachine: ConversationStateMachine

    @Before
    fun setUp() {
        stateMachine = ConversationStateMachine()
    }

    private fun sineFrame(frequencyHz: Double, frameIndex: Int = 0): ShortArray =
        ShortArray(FRAME_SIZE) { i ->
            val t = (frameIndex * FRAME_SIZE + i).toDouble() / SAMPLE_RATE
            (sin(2.0 * PI * frequencyHz * t) * AMPLITUDE).toInt().toShort()
        }

    private fun silenceFrame(): ShortArray = ShortArray(FRAME_SIZE) { 0 }

    /**
     * Replicates AudioPipeline.processFrame() timing logic with scripted speaker decisions.
     */
    private inner class PipelineSimulator(private val speakerSequence: Iterator<Speaker>) {
        private val speechBuffer = mutableListOf<ShortArray>()

        fun processFrame(frame: ShortArray, isSpeech: Boolean) {
            stateMachine.onTimeTick(FRAME_MS)

            if (isSpeech) {
                speechBuffer.add(frame)
                if (speechBuffer.size >= SPEECH_SEGMENT_FRAMES) {
                    identifyAndEmit()
                }
            } else {
                if (speechBuffer.size >= MIN_FLUSH_FRAMES) {
                    identifyAndEmit()
                }
                speechBuffer.clear()
                stateMachine.onSilenceDetected()
            }
        }

        private fun identifyAndEmit() {
            speechBuffer.clear()
            val speaker = speakerSequence.next()
            stateMachine.onSpeechDetected(speaker)
        }
    }

    data class FrameSpec(val frame: ShortArray, val isSpeech: Boolean)

    private fun speechFrames(freq: Double, count: Int, startIndex: Int = 0): List<FrameSpec> =
        (0 until count).map { i -> FrameSpec(sineFrame(freq, startIndex + i), true) }

    private fun silenceFrames(count: Int): List<FrameSpec> =
        (0 until count).map { FrameSpec(silenceFrame(), false) }

    private fun runScenario(frames: List<FrameSpec>, speakerDecisions: List<Speaker>): MetricsSnapshot {
        stateMachine.onRecord()
        val sim = PipelineSimulator(speakerDecisions.iterator())
        for (spec in frames) {
            sim.processFrame(spec.frame, spec.isSpeech)
        }
        stateMachine.onStop()
        return stateMachine.snapshot()
    }

    private fun assertInvariant(snap: MetricsSnapshot) {
        assertEquals(
            "TRT must equal TCT + BFST (TRT=${snap.trt}, TCT=${snap.tct}, BFST=${snap.bfst})",
            snap.trt, snap.tct + snap.bfst
        )
    }

    private fun assertNonNegative(snap: MetricsSnapshot, label: String) {
        assertTrue("$label: TRT >= 0", snap.trt >= 0)
        assertTrue("$label: WTA >= 0", snap.wta >= 0)
        assertTrue("$label: WTB >= 0", snap.wtb >= 0)
        assertTrue("$label: STA >= 0", snap.sta >= 0)
        assertTrue("$label: STB >= 0", snap.stb >= 0)
        assertTrue("$label: STM >= 0", snap.stm >= 0)
        assertTrue("$label: OVT >= 0", snap.ovt >= 0)
        assertTrue("$label: BFST >= 0", snap.bfst >= 0)
    }

    // ===== Test 1: Single speaker, all speech =====
    // 94 frames of speech = 47+47 = 2 emissions, both A.
    // Frames 1-47: tick in INITIAL_SILENCE. At 47: emit A → SPEAKER_A
    // Frames 48-94: tick in SPEAKER_A → WTA. At 94: emit A.
    // TRT = 94*32 = 3008. WTA = 47*32 = 1504. BFST = 1504.

    @Test
    fun singleSpeakerAllSpeech() {
        val frames = speechFrames(200.0, 94)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A))
        assertEquals("TRT", 94 * FRAME_MS, snap.trt)
        assertEquals("WTA", 47 * FRAME_MS, snap.wta)
        assertEquals("WTB", 0L, snap.wtb)
        assertEquals("BFST", 47 * FRAME_MS, snap.bfst)
        assertInvariant(snap)
    }

    // ===== Test 2: Two speakers, direct switch =====
    // 47 speech A → emit A, 47 speech A (ticks as A) → emit B (switch), 47 speech B → emit B
    // Total = 141 frames. Emits: A, B, B

    @Test
    fun twoSpeakersDirectSwitch() {
        val frames = speechFrames(200.0, 94) + speechFrames(2000.0, 47, startIndex = 94)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.B, Speaker.B))
        assertEquals("TRT", 141 * FRAME_MS, snap.trt)
        assertEquals("WTA", 47 * FRAME_MS, snap.wta) // frames 48-94
        assertEquals("WTB", 47 * FRAME_MS, snap.wtb) // frames 95-141
        assertEquals("STM", 0L, snap.stm)
        assertInvariant(snap)
    }

    // ===== Test 3: Speaker A → silence → Speaker A (STA) =====

    @Test
    fun withinSpeakerSilenceSta() {
        val frames = speechFrames(200.0, 94) +
                silenceFrames(10) +
                speechFrames(200.0, 94, startIndex = 104)
        // Emits: at frame 47 (A), at frame 94 (A), flush at silence (buffer empty after 94, no flush),
        // at frame 104+47=151 (A), at frame 104+94=198 (A)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A, Speaker.A, Speaker.A))
        assertTrue("STA > 0 for within-speaker silence", snap.sta > 0)
        assertEquals("STM", 0L, snap.stm)
        assertInvariant(snap)
    }

    // ===== Test 4: Speaker A → silence → Speaker B (STM) =====

    @Test
    fun betweenSpeakerSilenceStm() {
        val frames = speechFrames(200.0, 94) +
                silenceFrames(10) +
                speechFrames(2000.0, 94, startIndex = 104)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A, Speaker.B, Speaker.B))
        assertTrue("STM > 0 for between-speaker silence", snap.stm > 0)
        assertEquals("STA", 0L, snap.sta)
        assertInvariant(snap)
    }

    // ===== Test 5: Initial silence → speech =====

    @Test
    fun initialSilenceThenSpeech() {
        val frames = silenceFrames(20) + speechFrames(200.0, 94, startIndex = 20)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A))
        assertTrue("BFST includes initial silence", snap.bfst >= 20 * FRAME_MS)
        assertInvariant(snap)
    }

    // ===== Test 6: Overlap (BOTH speaker) handling =====

    @Test
    fun overlapBothSpeakerIncrementsBothMetrics() {
        // 47 speech → emit A (establish speaker A)
        // 47 speech → emit BOTH (overlap)
        // 47 speech → emit A
        val frames = speechFrames(200.0, 141)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.BOTH, Speaker.A))

        // During BOTH_TALKING, both WTA and WTB should be incremented
        assertTrue("WTA should include overlap time", snap.wta > 0)
        assertTrue("WTB should include overlap time from BOTH", snap.wtb > 0)
        assertTrue("OVT should be > 0", snap.ovt > 0)
        assertEquals("OVT equals WTB (only overlap contributed to B)", snap.ovt, snap.wtb)
        assertInvariant(snap)
    }

    // ===== Test 7: TRT invariant across scenarios =====

    @Test
    fun trtInvariantHoldsAcrossScenarios() {
        data class Scenario(val label: String, val frames: List<FrameSpec>, val speakers: List<Speaker>)

        val scenarios = listOf(
            Scenario("only silence", silenceFrames(20), emptyList()),
            Scenario("single speaker",
                speechFrames(200.0, 94),
                listOf(Speaker.A, Speaker.A)),
            Scenario("with overlap",
                speechFrames(200.0, 141),
                listOf(Speaker.A, Speaker.BOTH, Speaker.B)),
            Scenario("long silence + speech",
                silenceFrames(30) + speechFrames(200.0, 47, startIndex = 30),
                listOf(Speaker.A)),
        )

        for (scenario in scenarios) {
            stateMachine = ConversationStateMachine()
            val snap = runScenario(scenario.frames, scenario.speakers)
            assertInvariant(snap)
            assertNonNegative(snap, scenario.label)
        }
    }

    // ===== Test 8: Partial buffer flush =====

    @Test
    fun partialBufferFlushOnSilence() {
        // 3 speech frames + 2 silence. Buffer=3 >= MIN_FLUSH_FRAMES(2) → flush
        val frames = speechFrames(200.0, 3) + silenceFrames(2)
        val snap = runScenario(frames, listOf(Speaker.A))
        assertEquals("TRT", 5 * FRAME_MS, snap.trt)
        assertInvariant(snap)
    }

    // ===== Test 9: No flush when buffer too small =====

    @Test
    fun noFlushWhenBufferTooSmall() {
        // 1 speech frame + 1 silence. Buffer=1 < MIN_FLUSH_FRAMES → no flush
        val frames = speechFrames(200.0, 1) + silenceFrames(1)
        val snap = runScenario(frames, emptyList())
        assertEquals("TRT", 2 * FRAME_MS, snap.trt)
        assertEquals("WTA", 0L, snap.wta)
        assertInvariant(snap)
    }

    // ===== Test 10: Overlap followed by silence =====

    @Test
    fun overlapThenSilenceResolvesCorrectly() {
        // A talks, then BOTH talks, then silence, then B talks
        val frames = speechFrames(200.0, 94) +  // emits A, then BOTH
                silenceFrames(10) +
                speechFrames(200.0, 94, startIndex = 104)  // emits B, B
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.BOTH, Speaker.B, Speaker.B))

        assertTrue("STM > 0 (BOTH→B with silence between)", snap.stm > 0)
        assertTrue("OVT > 0", snap.ovt > 0)
        assertInvariant(snap)
    }
}
