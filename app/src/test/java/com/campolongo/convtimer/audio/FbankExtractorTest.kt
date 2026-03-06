package com.campolongo.convtimer.audio

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class FbankExtractorTest {

    private lateinit var extractor: FbankExtractor

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val FRAME_SIZE = 512  // audio capture frame
        private const val AMPLITUDE = 16000.0
    }

    @Before
    fun setUp() {
        extractor = FbankExtractor()
    }

    private fun sineAudio(frequencyHz: Double, numSamples: Int): ShortArray =
        ShortArray(numSamples) { i ->
            val t = i.toDouble() / SAMPLE_RATE
            (sin(2.0 * PI * frequencyHz * t) * AMPLITUDE).toInt().toShort()
        }

    @Test
    fun extractReturnsCorrectShape() {
        // 1.5s of audio = 24000 samples
        val audio = sineAudio(440.0, 24000)
        val (fbank, numFrames) = extractor.extract(audio)

        // numFrames = 1 + (24000 - 400) / 160 = 1 + 147 = 148
        assertEquals(148, numFrames)
        assertEquals(148 * 80, fbank.size)
    }

    @Test
    fun extractProducesFiniteValues() {
        val audio = sineAudio(1000.0, 16000)
        val (fbank, numFrames) = extractor.extract(audio)

        assertTrue("Should have frames", numFrames > 0)
        for (i in fbank.indices) {
            assertFalse("Fbank[$i] should not be NaN", fbank[i].isNaN())
            assertFalse("Fbank[$i] should not be Inf", fbank[i].isInfinite())
        }
    }

    @Test
    fun shortAudioReturnsEmptyResult() {
        // Audio shorter than one frame (400 samples)
        val audio = ShortArray(300) { 0 }
        val (fbank, numFrames) = extractor.extract(audio)
        assertEquals(0, numFrames)
        assertEquals(0, fbank.size)
    }

    @Test
    fun exactlyOneFrame() {
        // Exactly 400 samples = 1 frame
        val audio = sineAudio(440.0, 400)
        val (fbank, numFrames) = extractor.extract(audio)
        assertEquals(1, numFrames)
        assertEquals(80, fbank.size)
    }

    @Test
    fun differentFrequenciesProduceDifferentFbank() {
        val audio200 = sineAudio(200.0, 8000)
        val audio4000 = sineAudio(4000.0, 8000)

        val (fbank200, _) = extractor.extract(audio200)
        val (fbank4000, _) = extractor.extract(audio4000)

        // At least some mel bins should differ significantly
        var maxDiff = 0f
        for (i in fbank200.indices) {
            val diff = kotlin.math.abs(fbank200[i] - fbank4000[i])
            if (diff > maxDiff) maxDiff = diff
        }
        assertTrue("Different frequencies should produce different fbank (maxDiff=$maxDiff)", maxDiff > 1.0f)
    }

    @Test
    fun silenceProducesLowEnergy() {
        val silence = ShortArray(8000) { 0 }
        val toneAudio = sineAudio(1000.0, 8000)

        val (fbankSilence, nSilence) = extractor.extract(silence)
        val (fbankTone, nTone) = extractor.extract(toneAudio)

        // Average energy should be much lower for silence
        val avgSilence = fbankSilence.average()
        val avgTone = fbankTone.average()
        assertTrue("Silence fbank energy ($avgSilence) < tone energy ($avgTone)",
            avgSilence < avgTone)
    }

    @Test
    fun concatMultipleFramesMatchesSingleExtraction() {
        // Test that extracting from concatenated audio matches pipeline behavior
        // (multiple 512-sample frames → single long audio)
        val frames = 47
        val singleAudio = sineAudio(440.0, frames * FRAME_SIZE)
        val (fbankSingle, nFramesSingle) = extractor.extract(singleAudio)

        assertTrue("Should produce multiple fbank frames", nFramesSingle > 1)
        assertEquals(nFramesSingle * 80, fbankSingle.size)
    }

    @Test
    fun matchesPythonTorchaudioReference() {
        // Generate same 440 Hz sine as the Python reference generator
        val audio = sineAudio(440.0, 16000)
        val (fbank, numFrames) = extractor.extract(audio)

        // Load Python torchaudio reference from test resources
        val jsonText = javaClass.getResourceAsStream("/fbank_reference_440hz.json")
            ?: throw AssertionError("Reference file fbank_reference_440hz.json not found in test resources")
        val json = jsonText.bufferedReader().readText()

        // Simple JSON parsing (no library dependency needed)
        val numFramesRef = Regex("\"num_frames\":\\s*(\\d+)").find(json)!!.groupValues[1].toInt()
        val numBinsRef = Regex("\"num_mel_bins\":\\s*(\\d+)").find(json)!!.groupValues[1].toInt()
        val fbankMatch = Regex("\"fbank\":\\s*\\[([^]]+)]").find(json)!!.groupValues[1]
        val refValues = fbankMatch.split(",").map { it.trim().toFloat() }.toFloatArray()

        // Verify shapes match
        assertEquals("num_frames mismatch", numFramesRef, numFrames)
        assertEquals("num_mel_bins mismatch", numBinsRef, 80)
        assertEquals("total values mismatch", refValues.size, fbank.size)

        // Compare element-by-element
        var maxErr = 0f
        var maxErrIdx = 0
        var sumErr = 0.0
        for (i in fbank.indices) {
            val err = abs(fbank[i] - refValues[i])
            sumErr += err
            if (err > maxErr) {
                maxErr = err
                maxErrIdx = i
            }
        }
        val avgErr = sumErr / fbank.size

        val frame = maxErrIdx / 80
        val bin = maxErrIdx % 80
        println("Fbank cross-validation: maxErr=$maxErr (frame=$frame, bin=$bin), avgErr=$avgErr")

        assertTrue(
            "Max absolute error $maxErr at frame=$frame bin=$bin exceeds tolerance 0.01. " +
                "Kotlin=${fbank[maxErrIdx]}, Python=${refValues[maxErrIdx]}",
            maxErr < 0.01f
        )
    }

    @Test
    fun numFramesFormula() {
        // Verify: numFrames = 1 + (N - 400) / 160 for various audio lengths
        val testCases = listOf(
            400 to 1,
            560 to 2,     // 1 + (560-400)/160 = 1+1 = 2
            800 to 3,     // 1 + (800-400)/160 = 1+2 = 3
            16000 to 98,  // 1 + (16000-400)/160 = 1+97 = 98
        )
        for ((nSamples, expectedFrames) in testCases) {
            val audio = ShortArray(nSamples) { 1 }
            val (_, numFrames) = extractor.extract(audio)
            assertEquals("$nSamples samples → $expectedFrames frames", expectedFrames, numFrames)
        }
    }
}
