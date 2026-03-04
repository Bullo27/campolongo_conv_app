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
    //
    // 15 frames of speech, all Speaker A.
    //
    // Timeline (tick happens before event each frame):
    //   Frames 1-5:   tick in INITIAL_SILENCE (only TRT). At 5: emit A → SPEAKER_A
    //   Frames 6-10:  tick in SPEAKER_A → WTA += 32 each (=160). At 10: emit A.
    //   Frames 11-15: tick in SPEAKER_A → WTA += 32 each (=320). At 15: emit A.
    //
    // identifyAndEmit called 3 times → 3 speaker decisions, all A.

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

    // ===== Test 2: Two speakers, direct switch (no silence between) =====
    //
    // 10 frames A, then 10 frames B, continuous speech.
    //
    // Timeline:
    //   Frames 1-5:   INITIAL_SILENCE ticks. At 5: emit A → SPEAKER_A
    //   Frames 6-10:  SPEAKER_A ticks → WTA=160. At 10: emit A.
    //   Frames 11-15: SPEAKER_A ticks → WTA=320. At 15: emit B → SPEAKER_B (direct switch)
    //   Frames 16-20: SPEAKER_B ticks → WTB=160. At 20: emit B.
    //
    // Note: frames 11-15 tick as SPEAKER_A because the state doesn't change
    // until the buffer fills and speaker ID runs. This is the real app behavior.
    //
    // identifyAndEmit called 4 times: A, A, B, B.

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

    // ===== Test 3: Speaker A → silence → Speaker A (within-speaker silence = STA) =====
    //
    // 10 frames A speech, 5 frames silence, 10 frames A speech.
    //
    // Timeline:
    //   Frames 1-5:   INITIAL_SILENCE. At 5: emit A → SPEAKER_A.
    //   Frames 6-10:  SPEAKER_A → WTA=160. At 10: emit A.
    //   Frame 11 (silence): tick SPEAKER_A → WTA=192. onSilenceDetected → PENDING(prev=A).
    //   Frames 12-15 (silence): tick PENDING → pending=32,64,96,128.
    //   Frames 16-20 (speech A): tick PENDING → pending=160..288. At 20: emit A.
    //     resolvePendingSilence(A): prev=A, next=A → STA=288. → SPEAKER_A.
    //   Frames 21-25 (speech A): tick SPEAKER_A → WTA=224..352. At 25: emit A.
    //
    // identifyAndEmit called 4 times: A, A, A, A.

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

    // ===== Test 4: Speaker A → silence → Speaker B (between-speaker silence = STM) =====
    //
    // 10 frames A speech, 5 frames silence, 10 frames B speech.
    //
    // Timeline:
    //   Frames 1-5:   INITIAL_SILENCE. At 5: emit A → SPEAKER_A.
    //   Frames 6-10:  SPEAKER_A → WTA=160. At 10: emit A.
    //   Frame 11 (silence): tick SPEAKER_A → WTA=192. onSilenceDetected → PENDING(prev=A).
    //   Frames 12-15 (silence): PENDING → pending=32,64,96,128.
    //   Frames 16-20 (speech B): PENDING → pending=160..288. At 20: emit B.
    //     resolvePendingSilence(B): prev=A, next=B → STM=288. → SPEAKER_B.
    //   Frames 21-25 (speech B): SPEAKER_B → WTB=160. At 25: emit B.
    //
    // identifyAndEmit called 4 times: A, A, B, B.

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

    // ===== Test 5: Initial silence → speech (BFST calculation) =====
    //
    // 10 frames silence, then 10 frames A speech.
    //
    // Timeline:
    //   Frames 1-10 (silence): tick INITIAL_SILENCE (only TRT). onSilenceDetected (no-op).
    //   Frames 11-15 (speech A): tick INITIAL_SILENCE (only TRT). At 15: emit A → SPEAKER_A.
    //   Frames 16-20 (speech A): tick SPEAKER_A → WTA=160. At 20: emit A.
    //
    // identifyAndEmit called 2 times: A, A.

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
    //
    // A talks (10), silence (5), B talks (10), silence (5), A talks (10), stop.
    //
    // Timeline:
    //   Frames 1-5:   INITIAL_SILENCE. At 5: emit A → SPEAKER_A.
    //   Frames 6-10:  SPEAKER_A → WTA=160. At 10: emit A.
    //   Frame 11 (silence): tick SPEAKER_A → WTA=192. Silence → PENDING(prev=A).
    //   Frames 12-15 (silence): PENDING → pending=32,64,96,128.
    //   Frames 16-20 (speech B): PENDING → pending=160..288. At 20: emit B. STM=288. → SPEAKER_B.
    //   Frames 21-25 (speech B): SPEAKER_B → WTB=160. At 25: emit B.
    //   Frame 26 (silence): tick SPEAKER_B → WTB=192. Silence → PENDING(prev=B).
    //   Frames 27-30 (silence): PENDING → pending=32,64,96,128.
    //   Frames 31-35 (speech A): PENDING → pending=160..288. At 35: emit A.
    //     prev=B, next=A → STM+=288 (total STM=576). → SPEAKER_A.
    //   Frames 36-40 (speech A): SPEAKER_A → WTA=192+160=352. At 40: emit A.
    //
    // identifyAndEmit called 8 times: A, A, B, B, A, A — wait, let me count.
    // Frames 1-5 → emit (1). Frames 6-10 → emit (2). Frames 16-20 → emit (3).
    // Frames 21-25 → emit (4). Frames 31-35 → emit (5). Frames 36-40 → emit (6).
    // But also: frame 11 is silence, buffer was cleared at emit(2). Buffer empty, <2, no flush.
    // Frame 26 is silence, buffer was cleared at emit(4). Buffer empty, <2, no flush.
    // So 6 emits total.

    @Test
    fun fullConversation() {
        val frames = speechFrames(200.0, 10) +                           // A talks
                silenceFrames(5) +                                        // silence
                speechFrames(2000.0, 10, startIndex = 15) +              // B talks
                silenceFrames(5) +                                        // silence
                speechFrames(200.0, 10, startIndex = 30)                 // A talks
        val snap = runScenario(frames, listOf(
            Speaker.A, Speaker.A,   // first A block (frames 1-10)
            Speaker.B, Speaker.B,   // B block (frames 16-25)
            Speaker.A, Speaker.A,   // second A block (frames 31-40)
        ))

        assertEquals("TRT", 1280L, snap.trt)
        assertEquals("WTA", 352L, snap.wta)
        assertEquals("WTB", 192L, snap.wtb)
        assertEquals("STA", 0L, snap.sta)
        assertEquals("STB", 0L, snap.stb)
        assertEquals("STM", 576L, snap.stm)
        assertEquals("BFST", 160L, snap.bfst)

        // Derived metrics
        assertEquals("CTA = WTA + STA", 352L, snap.cta)
        assertEquals("CTB = WTB + STB", 192L, snap.ctb)
        assertEquals("TCT = CTA + STM + CTB", 1120L, snap.tct)
        assertEquals("TST = STA + STB + STM", 576L, snap.tst)
        assertInvariant(snap)
    }

    // ===== Test 7: TRT = TCT + BFST invariant across multiple scenarios =====

    @Test
    fun trtInvariantHoldsAcrossScenarios() {
        data class Scenario(val label: String, val frames: List<FrameSpec>, val speakers: List<Speaker>)

        val scenarios = listOf(
            // Only silence — no identifyAndEmit calls
            Scenario("only silence", silenceFrames(20), emptyList()),
            // Single speaker, speech only
            Scenario("single speaker",
                speechFrames(200.0, 10),
                listOf(Speaker.A, Speaker.A)),
            // Alternating speech/silence with partial buffer flush
            // 5 speech → emit(A). 3 silence (buffer empty, <2, no flush). 5 speech → emit(A). = 2 emits
            Scenario("alternating speech/silence",
                speechFrames(200.0, 5) + silenceFrames(3) + speechFrames(200.0, 5, startIndex = 8),
                listOf(Speaker.A, Speaker.A)),
            // Long initial silence + short speech
            Scenario("long silence + speech",
                silenceFrames(15) + speechFrames(200.0, 5, startIndex = 15),
                listOf(Speaker.A)),
            // Rapid speaker switching: A(5) B(5) A(5) = 3 emits
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

    // ===== Test 9: Partial buffer flush on silence =====
    //
    // 3 speech frames, then silence. Buffer has 3 frames (>= 2), so it flushes.
    //
    // Timeline:
    //   Frames 1-3: INITIAL_SILENCE ticks. Speech, buffer=1,2,3.
    //   Frame 4 (silence): tick INITIAL_SILENCE. Buffer=3 >= 2 → flush → emit A → SPEAKER_A.
    //     Then silence → PENDING(prev=A).
    //   Frame 5 (silence): tick PENDING → pending=32.
    //
    // identifyAndEmit called 1 time: A.

    @Test
    fun partialBufferFlushOnSilence() {
        val frames = speechFrames(200.0, 3) + silenceFrames(2)
        val snap = runScenario(frames, listOf(Speaker.A))

        assertEquals("TRT", 160L, snap.trt)
        // Frame 4 tick is in INITIAL_SILENCE (TRT only), then emit A → SPEAKER_A, then silence → PENDING.
        // Frame 5 tick is in PENDING → pending=32.
        // WTA = 0 (speech was detected at frame 4, but then immediately went to silence/PENDING)
        assertEquals("WTA", 0L, snap.wta)
        assertInvariant(snap)
    }

    // ===== Test 10: Buffer too small, no flush on silence =====
    //
    // 1 speech frame, then silence. Buffer has 1 frame (< 2), no flush.
    //
    // Timeline:
    //   Frame 1: INITIAL_SILENCE tick. Speech, buffer=1.
    //   Frame 2 (silence): INITIAL_SILENCE tick. Buffer=1 < 2 → no flush. Clear. onSilenceDetected (no-op).
    //
    // identifyAndEmit never called → 0 speaker decisions needed.

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
