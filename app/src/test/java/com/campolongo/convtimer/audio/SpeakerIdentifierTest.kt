package com.campolongo.convtimer.audio

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class SpeakerIdentifierTest {

    private lateinit var identifier: SpeakerIdentifier

    // Synthetic 13-dim vectors
    private val vectorA = FloatArray(13) { if (it % 2 == 0) 1.0f else 0.0f }
    private val vectorB = FloatArray(13) { if (it % 2 == 0) 0.0f else 1.0f }
    // A vector very similar to A (high cosine similarity)
    private val vectorCloseToA = FloatArray(13) { if (it % 2 == 0) 1.0f else 0.05f }
    // A vector midway between A and B (ambiguous)
    private val vectorAmbiguous = FloatArray(13) { 1.0f }

    @Before
    fun setUp() {
        identifier = SpeakerIdentifier()
    }

    @Test
    fun firstCallAlwaysReturnsSpeakerA() {
        assertEquals(Speaker.A, identifier.identify(vectorA))
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
        assertEquals(Speaker.B, identifier.identify(vectorB))
    }

    @Test
    fun bothEstablishedClearA() {
        identifier.identify(vectorA) // establish A
        identifier.identify(vectorB) // establish B
        assertEquals(Speaker.A, identifier.identify(vectorCloseToA))
    }

    @Test
    fun bothEstablishedClearB() {
        identifier.identify(vectorA) // establish A
        identifier.identify(vectorB) // establish B
        // A vector very similar to B
        val closeToB = FloatArray(13) { if (it % 2 == 0) 0.05f else 1.0f }
        assertEquals(Speaker.B, identifier.identify(closeToB))
    }

    @Test
    fun ambiguousVectorKeepsLastSpeaker() {
        identifier.identify(vectorA) // establish A, lastSpeaker = A
        identifier.identify(vectorB) // establish B, lastSpeaker = B

        // Ambiguous: equal similarity to both → hysteresis keeps last (B)
        val result = identifier.identify(vectorAmbiguous)
        assertEquals(Speaker.B, result)
    }

    @Test
    fun ambiguousAfterSpeakerAKeepsA() {
        identifier.identify(vectorA) // establish A, lastSpeaker = A
        identifier.identify(vectorB) // establish B, lastSpeaker = B
        identifier.identify(vectorCloseToA) // switch to A, lastSpeaker = A

        // Ambiguous → should keep A
        val result = identifier.identify(vectorAmbiguous)
        assertEquals(Speaker.A, result)
    }

    @Test
    fun emaReferenceDrift() {
        identifier.identify(vectorA) // establish A reference
        // Send many vectors that slowly drift A's reference
        val driftVector = FloatArray(13) { if (it % 2 == 0) 0.95f else 0.1f }
        repeat(20) {
            identifier.identify(driftVector)
        }
        // After drift, A's reference should have moved toward driftVector
        // A vector close to original A should still be A but with lower similarity
        val result = identifier.identify(vectorA)
        assertEquals(Speaker.A, result)
    }

    @Test
    fun resetClearsAllState() {
        identifier.identify(vectorA)
        identifier.identify(vectorB)
        identifier.reset()

        // After reset, first call should establish new speaker A again
        assertEquals(Speaker.A, identifier.identify(vectorB))
    }

    @Test
    fun zeroVectorHandledGracefully() {
        val zero = FloatArray(13) { 0.0f }
        // First call always returns A
        assertEquals(Speaker.A, identifier.identify(zero))
        // cosine similarity with zero vector returns 0, which is < threshold → Speaker B
        val nonZero = FloatArray(13) { 1.0f }
        assertEquals(Speaker.B, identifier.identify(nonZero))
    }

    @Test
    fun customThresholds() {
        // Very strict similarity threshold — even close vectors get classified as B
        val strict = SpeakerIdentifier(similarityThreshold = 0.99f, ambiguityMargin = 0.01f)
        strict.identify(vectorA) // establish A
        val slightlyDifferent = FloatArray(13) { if (it % 2 == 0) 1.0f else 0.2f }
        // cosine similarity between vectorA and slightlyDifferent is < 0.99
        assertEquals(Speaker.B, strict.identify(slightlyDifferent))
    }
}
