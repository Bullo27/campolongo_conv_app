package com.campolongo.convtimer.audio

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.sin

class MfccExtractorTest {

    private lateinit var extractor: MfccExtractor

    @Before
    fun setUp() {
        extractor = MfccExtractor()
    }

    @Test
    fun outputHas13Coefficients() {
        val samples = ShortArray(512) { 0 }
        val result = extractor.extract(samples)
        assertEquals(13, result.size)
    }

    @Test
    fun deterministic() {
        val samples = ShortArray(512) { (it * 7 % 1000).toShort() }
        val result1 = extractor.extract(samples)
        val result2 = extractor.extract(samples)
        assertArrayEquals(result1, result2, 0.0f)
    }

    @Test
    fun noNanOrInfForSilence() {
        val silence = ShortArray(512) { 0 }
        val result = extractor.extract(silence)
        for (i in result.indices) {
            assertFalse("Coefficient $i is NaN", result[i].isNaN())
            assertFalse("Coefficient $i is Inf", result[i].isInfinite())
        }
    }

    @Test
    fun noNanOrInfForMaxAmplitude() {
        val loud = ShortArray(512) { if (it % 2 == 0) Short.MAX_VALUE else Short.MIN_VALUE }
        val result = extractor.extract(loud)
        for (i in result.indices) {
            assertFalse("Coefficient $i is NaN", result[i].isNaN())
            assertFalse("Coefficient $i is Inf", result[i].isInfinite())
        }
    }

    @Test
    fun noNanOrInfForShortInput() {
        // Input shorter than FFT size (should be zero-padded)
        val short = ShortArray(64) { 1000 }
        val result = extractor.extract(short)
        for (i in result.indices) {
            assertFalse("Coefficient $i is NaN", result[i].isNaN())
            assertFalse("Coefficient $i is Inf", result[i].isInfinite())
        }
    }

    @Test
    fun differentInputsProduceDifferentOutputs() {
        val silence = ShortArray(512) { 0 }
        // 440Hz sine wave at 16kHz sample rate
        val sineWave = ShortArray(512) { i ->
            (sin(2.0 * Math.PI * 440.0 * i / 16000.0) * 16000).toInt().toShort()
        }
        val resultSilence = extractor.extract(silence)
        val resultSine = extractor.extract(sineWave)
        // At least one coefficient must differ
        var anyDifferent = false
        for (i in resultSilence.indices) {
            if (resultSilence[i] != resultSine[i]) {
                anyDifferent = true
                break
            }
        }
        assertTrue("Silence and sine wave should produce different MFCCs", anyDifferent)
    }

    @Test
    fun fullSizeInputDoesNotThrow() {
        // Exactly fftSize = 512 samples
        val samples = ShortArray(512) { (it % 256).toShort() }
        val result = extractor.extract(samples)
        assertEquals(13, result.size)
    }
}
