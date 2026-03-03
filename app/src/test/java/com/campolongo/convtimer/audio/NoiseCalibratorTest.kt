package com.campolongo.convtimer.audio

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class NoiseCalibratorTest {

    private lateinit var calibrator: NoiseCalibrator

    @Before
    fun setUp() {
        calibrator = NoiseCalibrator()
    }

    @Test
    fun emptyFramesReturnsQuiet() {
        assertEquals(NoiseLevel.QUIET, calibrator.calibrate(emptyList()))
    }

    @Test
    fun silenceReturnsQuiet() {
        // All-zero samples → dBFS = -96
        val frames = listOf(ShortArray(512) { 0 })
        assertEquals(NoiseLevel.QUIET, calibrator.calibrate(frames))
    }

    @Test
    fun lowAmplitudeReturnsQuiet() {
        // RMS ~10 → dBFS = 20*log10(10/32767) ≈ -70.3 dBFS → QUIET
        val frames = listOf(ShortArray(512) { 10 })
        assertEquals(NoiseLevel.QUIET, calibrator.calibrate(frames))
    }

    @Test
    fun mediumAmplitudeReturnsModerate() {
        // Need dBFS between -40 and -25
        // dBFS = -30 → rms = 32767 * 10^(-30/20) = 32767 * 0.0316 ≈ 1036
        val frames = listOf(ShortArray(512) { 1036 })
        val result = calibrator.calibrate(frames)
        assertEquals(NoiseLevel.MODERATE, result)
    }

    @Test
    fun highAmplitudeReturnsNoisy() {
        // Need dBFS >= -25
        // dBFS = -20 → rms = 32767 * 10^(-20/20) = 32767 * 0.1 ≈ 3277
        val frames = listOf(ShortArray(512) { 3277 })
        val result = calibrator.calibrate(frames)
        assertEquals(NoiseLevel.NOISY, result)
    }

    @Test
    fun multipleFramesAveraged() {
        // Two frames: one quiet, one loud → average should be somewhere in between
        val quiet = ShortArray(512) { 10 }
        val loud = ShortArray(512) { 3277 }
        val result = calibrator.calibrate(listOf(quiet, loud))
        // RMS will be sqrt((10^2*512 + 3277^2*512) / 1024) ≈ 2318 → dBFS ≈ -23 → NOISY
        assertEquals(NoiseLevel.NOISY, result)
    }

    @Test
    fun noiseLevelLabels() {
        assertEquals("Quiet", NoiseLevel.QUIET.label)
        assertEquals("Moderate", NoiseLevel.MODERATE.label)
        assertEquals("Noisy", NoiseLevel.NOISY.label)
    }
}
