package com.campolongo.convtimer.audio

import com.campolongo.convtimer.state.ConversationStateMachine
import com.campolongo.convtimer.state.ConvState
import com.campolongo.convtimer.state.MetricsSnapshot
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Integration tests that feed synthetic audio through the real pipeline components:
 * MfccExtractor → Speaker decision → ConversationStateMachine.
 *
 * VAD decisions and speaker assignments are scripted to test deterministically.
 * The real MfccExtractor processes every frame (validating no crashes/NaN),
 * while speaker decisions use a scripted sequence because synthetic audio cannot
 * produce MFCC vectors different enough to cross the SpeakerIdentifier threshold
 * (pure tones produce cosine similarity ~0.85-0.91 in MFCC space, above the 0.80 threshold).
 *
 * The real SpeakerIdentifier is validated separately in SpeakerIdentifierTest with
 * constructed MFCC vectors that mimic real speech variability.
 *
 * Each frame = 512 samples @ 16kHz = 32ms.
 */
class PipelineIntegrationTest {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val FRAME_SIZE = 512
        private const val FRAME_MS = 32L // 512 / 16000 = 32ms
        private const val SPEECH_SEGMENT_FRAMES = 5 // test uses 5 for predictable timing (app uses 8)
        private const val MFCC_COEFFS = 13
        private const val AMPLITUDE = 16000.0
    }

    private lateinit var mfccExtractor: MfccExtractor
    private lateinit var stateMachine: ConversationStateMachine

    @Before
    fun setUp() {
        mfccExtractor = MfccExtractor()
        stateMachine = ConversationStateMachine()
    }

    // --- Audio frame generators ---

    private fun sineFrame(frequencyHz: Double, frameIndex: Int = 0): ShortArray =
        ShortArray(FRAME_SIZE) { i ->
            val t = (frameIndex * FRAME_SIZE + i).toDouble() / SAMPLE_RATE
            (sin(2.0 * PI * frequencyHz * t) * AMPLITUDE).toInt().toShort()
        }

    private fun silenceFrame(): ShortArray = ShortArray(FRAME_SIZE) { 0 }

    // --- Pipeline Simulator ---

    /**
     * Replicates AudioPipeline.processFrame() logic (AudioPipeline.kt:135-173):
     * - Buffers speech frames; at SPEECH_SEGMENT_FRAMES, extracts MFCCs, averages, emits speaker
     * - Flushes partial buffer (>=2 frames) on silence
     * - Ticks stateMachine.onTimeTick(32) per frame (before events)
     *
     * Speaker decisions come from a scripted iterator rather than SpeakerIdentifier,
     * while MFCC extraction still runs on every frame to validate correctness.
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
                if (speechBuffer.size >= 2) {
                    identifyAndEmit()
                }
                speechBuffer.clear()
                stateMachine.onSilenceDetected()
            }
        }

        private fun identifyAndEmit() {
            // Run real MFCC extraction on all buffered frames (validates no NaN/Inf/crash)
            val avgMfcc = FloatArray(MFCC_COEFFS)
            for (buffered in speechBuffer) {
                val frameMfcc = mfccExtractor.extract(buffered)
                for (i in avgMfcc.indices) {
                    avgMfcc[i] += frameMfcc[i]
                }
                // Validate MFCC output
                for (i in frameMfcc.indices) {
                    assertFalse("MFCC coefficient $i is NaN", frameMfcc[i].isNaN())
                    assertFalse("MFCC coefficient $i is Inf", frameMfcc[i].isInfinite())
                }
            }
            speechBuffer.clear()

            // Use scripted speaker decision
            val speaker = speakerSequence.next()
            stateMachine.onSpeechDetected(speaker)
        }
    }

    // --- Frame spec and helpers ---

    data class FrameSpec(val frame: ShortArray, val isSpeech: Boolean)

    private fun speechFrames(freq: Double, count: Int, startIndex: Int = 0): List<FrameSpec> =
        (0 until count).map { i -> FrameSpec(sineFrame(freq, startIndex + i), true) }

    private fun silenceFrames(count: Int): List<FrameSpec> =
        (0 until count).map { FrameSpec(silenceFrame(), false) }

    /**
     * Runs a scenario: starts recording, processes all frames, stops.
     * @param speakerDecisions the speaker to return for each identifyAndEmit call
     */
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
        assertTrue("$label: BFST >= 0", snap.bfst >= 0)
    }

    // ===== Test 8: MFCC produces different vectors for different frequencies =====

    @Test
    fun mfccProducesDifferentVectorsForDifferentFrequencies() {
        val lowMfcc = mfccExtractor.extract(sineFrame(200.0))
        val highMfcc = mfccExtractor.extract(sineFrame(2000.0))

        // Vectors should differ (at least one coefficient different)
        var anyDifferent = false
        for (i in lowMfcc.indices) {
            if (lowMfcc[i] != highMfcc[i]) { anyDifferent = true; break }
        }
        assertTrue("200Hz and 2000Hz should produce different MFCC vectors", anyDifferent)

        // Note: cosine similarity is ~0.91 for these pure tones, above the 0.80 speaker ID
        // threshold. Real speech has richer spectral content that produces more separable MFCCs.
        // This validates that MfccExtractor IS sensitive to frequency, even though the difference
        // isn't large enough for the speaker ID threshold with synthetic signals.
        val similarity = cosineSimilarity(lowMfcc, highMfcc)
        assertTrue("Similarity should be < 1.0 (got $similarity)", similarity < 1.0f)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) { dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i] }
        val denom = sqrt(normA * normB)
        return if (denom > 0f) dot / denom else 0f
    }

    // ===== Test 1: Single speaker, all speech =====

    @Test
    fun singleSpeakerAllSpeech() {
        val frames = speechFrames(200.0, 15)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A, Speaker.A))

        assertEquals("TRT", 480L, snap.trt)
        assertEquals("WTA", 320L, snap.wta)
        assertEquals("WTB", 0L, snap.wtb)
        assertEquals("STA", 0L, snap.sta)
        assertEquals("STB", 0L, snap.stb)
        assertEquals("STM", 0L, snap.stm)
        assertEquals("BFST (initial buffer delay)", 160L, snap.bfst)
        assertInvariant(snap)
    }

    // ===== Test 2: Two speakers, direct switch =====

    @Test
    fun twoSpeakersDirectSwitch() {
        val frames = speechFrames(200.0, 10) + speechFrames(2000.0, 10, startIndex = 10)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A, Speaker.B, Speaker.B))

        assertEquals("TRT", 640L, snap.trt)
        assertEquals("WTA", 320L, snap.wta)
        assertEquals("WTB", 160L, snap.wtb)
        assertEquals("STA", 0L, snap.sta)
        assertEquals("STB", 0L, snap.stb)
        assertEquals("STM", 0L, snap.stm)
        assertEquals("BFST", 160L, snap.bfst)
        assertInvariant(snap)
    }

    // ===== Test 3: Within-speaker silence (STA) =====

    @Test
    fun withinSpeakerSilenceSta() {
        val frames = speechFrames(200.0, 10) +
                silenceFrames(5) +
                speechFrames(200.0, 10, startIndex = 15)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A, Speaker.A, Speaker.A))

        assertEquals("TRT", 800L, snap.trt)
        assertEquals("WTA", 352L, snap.wta)
        assertEquals("WTB", 0L, snap.wtb)
        assertEquals("STA", 288L, snap.sta)
        assertEquals("STB", 0L, snap.stb)
        assertEquals("STM", 0L, snap.stm)
        assertEquals("BFST", 160L, snap.bfst)
        assertInvariant(snap)
    }

    // ===== Test 4: Between-speaker silence (STM) =====

    @Test
    fun betweenSpeakerSilenceStm() {
        val frames = speechFrames(200.0, 10) +
                silenceFrames(5) +
                speechFrames(2000.0, 10, startIndex = 15)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A, Speaker.B, Speaker.B))

        assertEquals("TRT", 800L, snap.trt)
        assertEquals("WTA", 192L, snap.wta)
        assertEquals("WTB", 160L, snap.wtb)
        assertEquals("STA", 0L, snap.sta)
        assertEquals("STB", 0L, snap.stb)
        assertEquals("STM", 288L, snap.stm)
        assertEquals("BFST", 160L, snap.bfst)
        assertInvariant(snap)
    }

    // ===== Test 5: Initial silence then speech =====

    @Test
    fun initialSilenceThenSpeech() {
        val frames = silenceFrames(10) + speechFrames(200.0, 10, startIndex = 10)
        val snap = runScenario(frames, listOf(Speaker.A, Speaker.A))

        assertEquals("TRT", 640L, snap.trt)
        assertEquals("WTA", 160L, snap.wta)
        assertEquals("WTB", 0L, snap.wtb)
        assertEquals("STA", 0L, snap.sta)
        assertEquals("STM", 0L, snap.stm)
        assertEquals("BFST (10 silence + 5 buffer frames)", 480L, snap.bfst)
        assertInvariant(snap)
    }

    // ===== Test 6: Full conversation =====

    @Test
    fun fullConversation() {
        val frames = speechFrames(200.0, 10) +
                silenceFrames(5) +
                speechFrames(2000.0, 10, startIndex = 15) +
                silenceFrames(5) +
                speechFrames(200.0, 10, startIndex = 30)
        val snap = runScenario(frames, listOf(
            Speaker.A, Speaker.A,
            Speaker.B, Speaker.B,
            Speaker.A, Speaker.A,
        ))

        assertEquals("TRT", 1280L, snap.trt)
        assertEquals("WTA", 352L, snap.wta)
        assertEquals("WTB", 192L, snap.wtb)
        assertEquals("STA", 0L, snap.sta)
        assertEquals("STB", 0L, snap.stb)
        assertEquals("STM", 576L, snap.stm)
        assertEquals("BFST", 160L, snap.bfst)

        assertEquals("CTA = WTA + STA", 352L, snap.cta)
        assertEquals("CTB = WTB + STB", 192L, snap.ctb)
        assertEquals("TCT = CTA + STM + CTB", 1120L, snap.tct)
        assertEquals("TST = STA + STB + STM", 576L, snap.tst)
        assertInvariant(snap)
    }

    // ===== Test 7: TRT invariant across scenarios =====

    @Test
    fun trtInvariantHoldsAcrossScenarios() {
        data class Scenario(val label: String, val frames: List<FrameSpec>, val speakers: List<Speaker>)

        val scenarios = listOf(
            Scenario("only silence", silenceFrames(20), emptyList()),
            Scenario("single speaker",
                speechFrames(200.0, 10),
                listOf(Speaker.A, Speaker.A)),
            Scenario("alternating speech/silence",
                speechFrames(200.0, 5) + silenceFrames(3) + speechFrames(200.0, 5, startIndex = 8),
                listOf(Speaker.A, Speaker.A)),
            Scenario("long silence + speech",
                silenceFrames(15) + speechFrames(200.0, 5, startIndex = 15),
                listOf(Speaker.A)),
            Scenario("rapid switching",
                speechFrames(200.0, 5) + speechFrames(2000.0, 5, startIndex = 5) + speechFrames(200.0, 5, startIndex = 10),
                listOf(Speaker.A, Speaker.B, Speaker.A)),
        )

        for (scenario in scenarios) {
            mfccExtractor = MfccExtractor()
            stateMachine = ConversationStateMachine()

            val snap = runScenario(scenario.frames, scenario.speakers)
            assertInvariant(snap)
            assertNonNegative(snap, scenario.label)
        }
    }

    // ===== Test 9: Partial buffer flush =====

    @Test
    fun partialBufferFlushOnSilence() {
        val frames = speechFrames(200.0, 3) + silenceFrames(2)
        val snap = runScenario(frames, listOf(Speaker.A))

        assertEquals("TRT", 160L, snap.trt)
        assertEquals("WTA", 0L, snap.wta)
        assertInvariant(snap)
    }

    // ===== Test 10: Buffer too small =====

    @Test
    fun noFlushWhenBufferTooSmall() {
        val frames = speechFrames(200.0, 1) + silenceFrames(1)
        val snap = runScenario(frames, emptyList())

        assertEquals("TRT", 64L, snap.trt)
        assertEquals("WTA", 0L, snap.wta)
        assertEquals("BFST = TRT (nothing happened)", 64L, snap.bfst)
        assertInvariant(snap)
    }
}
