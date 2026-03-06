package com.campolongo.convtimer.audio

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class SpeakerIdentifierTest {

    private lateinit var identifier: SpeakerIdentifier

    // Synthetic 192-dim vectors (matching ECAPA-TDNN embedding dimension)
    private val dim = 192
    private val vectorA = FloatArray(dim) { if (it % 2 == 0) 1.0f else 0.0f }
    private val vectorB = FloatArray(dim) { if (it % 2 == 0) 0.0f else 1.0f }
    private val vectorCloseToA = FloatArray(dim) { if (it % 2 == 0) 1.0f else 0.05f }
    private val vectorAmbiguous = FloatArray(dim) { 1.0f }  // equal sim to A and B

    @Before
    fun setUp() {
        identifier = SpeakerIdentifier()
    }

    @Test
    fun firstCallAlwaysReturnsSpeakerA() {
        assertEquals(Speaker.A, identifier.identify(vectorA))
        assertEquals(1.0f, identifier.lastSimA, 0.01f)
        assertEquals(0.0f, identifier.lastSimB, 0.01f)
    }

    @Test
    fun firstCallWithAnyVectorReturnsSpeakerA() {
        assertEquals(Speaker.A, identifier.identify(vectorB))
    }

    @Test
    fun similarVectorToAReturnsSpeakerA() {
        identifier.identify(vectorA) // establish A
        assertEquals(Speaker.A, identifier.identify(vectorCloseToA))
    }

    @Test
    fun differentVectorEstablishesSpeakerB() {
        identifier.identify(vectorA) // establish A
        // With bConfirmFrames=2, first below-threshold returns A (candidate)
        assertEquals(Speaker.A, identifier.identify(vectorB, nFrames = 47))
        // Second consecutive below-threshold confirms B
        assertEquals(Speaker.B, identifier.identify(vectorB, nFrames = 47))
    }

    @Test
    fun singleBelowThresholdDoesNotEstablishB() {
        identifier.identify(vectorA) // establish A
        identifier.identify(vectorB, nFrames = 47) // 1st below-threshold
        identifier.identify(vectorCloseToA) // back above threshold → resets candidate
        assertEquals(Speaker.A, identifier.identify(vectorB, nFrames = 47))
    }

    @Test
    fun flushSegmentsCannotTriggerB() {
        identifier.identify(vectorA) // establish A
        // Short segments (nFrames < 47) should NOT count toward B confirmation
        identifier.identify(vectorB, nFrames = 10) // flush, returns A
        identifier.identify(vectorB, nFrames = 10) // flush, returns A
        identifier.identify(vectorB, nFrames = 10) // flush, returns A
        assertNull(identifier.speakerBRef)
        assertEquals(Speaker.A, identifier.identify(vectorB, nFrames = 10))
    }

    @Test
    fun fullSegmentsCanTriggerB() {
        identifier.identify(vectorA) // establish A
        identifier.identify(vectorB, nFrames = 47) // full segment, 1st candidate
        identifier.identify(vectorB, nFrames = 47) // full segment, confirms B
        assertNotNull(identifier.speakerBRef)
    }

    @Test
    fun bothEstablishedClearA() {
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47) // confirm B
        assertEquals(Speaker.A, identifier.identify(vectorCloseToA))
    }

    @Test
    fun bothEstablishedClearB() {
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47) // confirm B
        val closeToB = FloatArray(dim) { if (it % 2 == 0) 0.05f else 1.0f }
        assertEquals(Speaker.B, identifier.identify(closeToB))
    }

    @Test
    fun ambiguousVectorKeepsLastSpeaker() {
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47) // confirm B, lastSpeaker = B
        // Ambiguous: equal similarity to both → keeps last (B)
        assertEquals(Speaker.B, identifier.identify(vectorAmbiguous))
    }

    @Test
    fun ambiguousAfterSpeakerAKeepsA() {
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorCloseToA) // switch to A
        assertEquals(Speaker.A, identifier.identify(vectorAmbiguous))
    }

    @Test
    fun referencesFreezeAfterBEstablished() {
        identifier.identify(vectorA) // establish A
        val refABefore = identifier.speakerARef!!.copyOf()
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47) // confirm B → refs freeze
        val refBAfter = identifier.speakerBRef!!.copyOf()

        // Send many vectors — refs should NOT change
        val driftVector = FloatArray(dim) { 0.5f }
        repeat(20) {
            identifier.identify(driftVector)
        }

        // Refs unchanged
        assertArrayEquals(refABefore, identifier.speakerARef, 1e-6f)
        assertArrayEquals(refBAfter, identifier.speakerBRef, 1e-6f)
    }

    @Test
    fun emaUpdatesBeforeBEstablished() {
        identifier.identify(vectorA) // establish A
        val refABefore = identifier.speakerARef!!.copyOf()
        // Send similar-to-A vectors → EMA should update ref_A
        val driftVector = FloatArray(dim) { if (it % 2 == 0) 0.95f else 0.1f }
        repeat(5) {
            identifier.identify(driftVector)
        }
        // Ref should have drifted
        var anyDifferent = false
        for (i in refABefore.indices) {
            if (refABefore[i] != identifier.speakerARef!![i]) { anyDifferent = true; break }
        }
        assertTrue("Reference A should drift before B is established", anyDifferent)
    }

    @Test
    fun marginZeroMeansNearestNeighbor() {
        // With margin=0 (default), the closer speaker wins even by tiny amount
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47)
        // Create a vector slightly closer to A than B
        val slightlyA = FloatArray(dim) { if (it % 2 == 0) 0.6f else 0.4f }
        // simA should be slightly > simB → returns A
        val result = identifier.identify(slightlyA)
        assertTrue("lastSimA > lastSimB", identifier.lastSimA > identifier.lastSimB)
        assertEquals(Speaker.A, result)
    }

    @Test
    fun lastSimExposedCorrectly() {
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47) // B established

        identifier.identify(vectorCloseToA)
        assertTrue("lastSimA should be high for vector close to A", identifier.lastSimA > 0.9f)
        assertTrue("lastSimB should be low for vector close to A", identifier.lastSimB < 0.5f)
    }

    @Test
    fun resetClearsAllState() {
        identifier.identify(vectorA)
        identifier.identify(vectorB, nFrames = 47)
        identifier.identify(vectorB, nFrames = 47)
        identifier.reset()

        assertNull(identifier.speakerARef)
        assertNull(identifier.speakerBRef)
        assertEquals(0f, identifier.lastSimA, 0.001f)
        assertEquals(0f, identifier.lastSimB, 0.001f)
        assertEquals(Speaker.A, identifier.identify(vectorB))
    }

    @Test
    fun zeroVectorHandledGracefully() {
        val zero = FloatArray(dim) { 0.0f }
        assertEquals(Speaker.A, identifier.identify(zero))
        val nonZero = FloatArray(dim) { 1.0f }
        assertEquals(Speaker.A, identifier.identify(nonZero, nFrames = 47))
        assertEquals(Speaker.B, identifier.identify(nonZero, nFrames = 47))
    }

    @Test
    fun customThresholds() {
        val strict = SpeakerIdentifier(
            similarityThreshold = 0.99f, ambiguityMargin = 0.01f,
            bConfirmFrames = 1, minFramesForB = 1
        )
        strict.identify(vectorA)
        val slightlyDifferent = FloatArray(dim) { if (it % 2 == 0) 1.0f else 0.2f }
        assertEquals(Speaker.B, strict.identify(slightlyDifferent, nFrames = 1))
    }
}
