package com.campolongo.convtimer.audio

import kotlin.math.*

/**
 * Kaldi-compatible Fbank (filter-bank) feature extractor.
 *
 * NOT thread-safe: reuses internal buffers across calls. Callers must
 * ensure single-threaded access or use separate instances per thread.
 *
 * Matches torchaudio.compliance.kaldi.fbank() with these parameters:
 *   sample_frequency=16000, frame_length=25ms, frame_shift=10ms,
 *   num_mel_bins=80, dither=0.0, energy_floor=1.0, window_type="hamming",
 *   preemphasis_coefficient=0.97, remove_dc_offset=true, snip_edges=true,
 *   round_to_power_of_two=true, use_energy=false, use_log_fbank=true,
 *   use_power=true, raw_energy=true, low_freq=20.0, high_freq=0.0 (=Nyquist)
 */
class FbankExtractor(
    private val sampleRate: Int = 16000,
    private val frameLengthMs: Float = 25f,
    private val frameShiftMs: Float = 10f,
    private val numMelBins: Int = 80,
    private val preemphCoeff: Float = 0.97f,
    private val lowFreq: Float = 20f,
    private val highFreq: Float = 0f // 0 means Nyquist
) {
    // Frame parameters
    private val windowSize = (sampleRate * frameLengthMs * 0.001f).toInt()   // 400
    private val windowShift = (sampleRate * frameShiftMs * 0.001f).toInt()   // 160
    private val fftSize = nextPowerOf2(windowSize)                            // 512

    // Precomputed Hamming window (symmetric, N-1 denominator)
    private val hammingWindow = FloatArray(windowSize) { i ->
        val a = 2.0 * PI / (windowSize - 1)
        (0.54 - 0.46 * cos(a * i)).toFloat()
    }

    // Sparse mel filterbank: weights + bin ranges per filter
    private val melFilterbank: Array<FloatArray>
    private val melStartBin: IntArray   // first non-zero FFT bin per mel filter
    private val melEndBin: IntArray     // last non-zero FFT bin + 1 per mel filter

    // Precomputed FFT twiddle factors
    private val twiddleReal: FloatArray
    private val twiddleImag: FloatArray

    // Reusable buffers
    private val frameBuf = FloatArray(fftSize)
    private val fftReal = FloatArray(fftSize)
    private val fftImag = FloatArray(fftSize)

    init {
        val numFftBins = fftSize / 2 + 1  // 257
        val effectiveHighFreq = if (highFreq <= 0f) sampleRate / 2f else highFreq  // 8000

        // Mel filterbank construction (Kaldi natural-log mel scale)
        val melLow = hzToMel(lowFreq)
        val melHigh = hzToMel(effectiveHighFreq)
        val melDelta = (melHigh - melLow) / (numMelBins + 1)
        val fftBinWidth = sampleRate.toFloat() / fftSize  // 31.25

        melFilterbank = Array(numMelBins) { bin ->
            val leftMel = melLow + bin * melDelta
            val centerMel = melLow + (bin + 1) * melDelta
            val rightMel = melLow + (bin + 2) * melDelta
            FloatArray(numFftBins) { i ->
                val freq = fftBinWidth * i
                val mel = hzToMel(freq)
                val upSlope = if (centerMel > leftMel) (mel - leftMel) / (centerMel - leftMel) else 0f
                val downSlope = if (rightMel > centerMel) (rightMel - mel) / (rightMel - centerMel) else 0f
                maxOf(0f, minOf(upSlope, downSlope))
            }
        }

        // Precompute non-zero bin ranges for sparse filterbank application
        melStartBin = IntArray(numMelBins)
        melEndBin = IntArray(numMelBins)
        for (bin in 0 until numMelBins) {
            val filter = melFilterbank[bin]
            var start = 0
            while (start < numFftBins && filter[start] == 0f) start++
            var end = numFftBins
            while (end > start && filter[end - 1] == 0f) end--
            melStartBin[bin] = start
            melEndBin[bin] = end
        }

        // Precompute FFT twiddle factors
        twiddleReal = FloatArray(fftSize / 2)
        twiddleImag = FloatArray(fftSize / 2)
        for (i in 0 until fftSize / 2) {
            val angle = -2.0 * PI * i / fftSize
            twiddleReal[i] = cos(angle).toFloat()
            twiddleImag[i] = sin(angle).toFloat()
        }
    }

    /**
     * Extract Fbank features from int16 audio.
     * @return Pair of (flattened [T*80] FloatArray, number of frames T)
     */
    fun extract(audioInt16: ShortArray): Pair<FloatArray, Int> {
        // Convert to float32 normalized to [-1, 1]
        val audio = FloatArray(audioInt16.size) { audioInt16[it].toFloat() / 32768f }
        return extractFromFloat(audio)
    }

    /**
     * Extract Fbank features from float32 audio (already normalized to [-1, 1]).
     * @return Pair of (flattened [T*80] FloatArray, number of frames T)
     */
    fun extractFromFloat(audio: FloatArray): Pair<FloatArray, Int> {
        // snip_edges=true: number of frames
        if (audio.size < windowSize) return Pair(FloatArray(0), 0)
        val numFrames = 1 + (audio.size - windowSize) / windowShift

        val numFftBins = fftSize / 2 + 1
        val output = FloatArray(numFrames * numMelBins)

        for (f in 0 until numFrames) {
            val offset = f * windowShift

            // Extract frame
            for (i in 0 until windowSize) {
                frameBuf[i] = audio[offset + i]
            }

            // Step 1: DC offset removal (per-frame mean subtraction)
            var mean = 0f
            for (i in 0 until windowSize) mean += frameBuf[i]
            mean /= windowSize
            for (i in 0 until windowSize) frameBuf[i] -= mean

            // Step 2: Pre-emphasis (Kaldi: first sample *= (1 - coeff))
            // Process from end to beginning to avoid overwrite issues
            for (i in windowSize - 1 downTo 1) {
                frameBuf[i] -= preemphCoeff * frameBuf[i - 1]
            }
            frameBuf[0] *= (1f - preemphCoeff)

            // Step 3: Apply Hamming window
            for (i in 0 until windowSize) {
                frameBuf[i] *= hammingWindow[i]
            }

            // Step 4: Zero-pad to FFT size
            for (i in windowSize until fftSize) {
                frameBuf[i] = 0f
            }

            // Step 5: Real FFT
            frameBuf.copyInto(fftReal)
            fftImag.fill(0f)
            fft(fftReal, fftImag, fftSize)

            // Step 6: Power spectrum (real^2 + imag^2, no normalization)
            // Only need first numFftBins (0..256) = fftSize/2 + 1
            for (i in 0 until numFftBins) {
                val re = fftReal[i]
                val im = fftImag[i]
                frameBuf[i] = re * re + im * im
            }

            // Step 7: Apply mel filterbank (sparse) and log
            val outOffset = f * numMelBins
            for (bin in 0 until numMelBins) {
                var energy = 0f
                val filter = melFilterbank[bin]
                for (i in melStartBin[bin] until melEndBin[bin]) {
                    energy += filter[i] * frameBuf[i]
                }
                output[outOffset + bin] = ln(maxOf(energy, EPSILON))
            }
        }

        return Pair(output, numFrames)
    }

    /** Kaldi mel scale: mel = 1127 * ln(1 + hz / 700) */
    private fun hzToMel(hz: Float): Float = 1127f * ln(1f + hz / 700f)

    /** In-place Cooley-Tukey FFT (iterative, radix-2). */
    private fun fft(real: FloatArray, imag: FloatArray, n: Int) {
        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                var tmp = real[i]; real[i] = real[j]; real[j] = tmp
                tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp
            }
            var m = n / 2
            while (m >= 1 && j >= m) {
                j -= m
                m /= 2
            }
            j += m
        }

        // Butterfly stages
        var step = 2
        while (step <= n) {
            val halfStep = step / 2
            val twiddleStride = n / step
            for (k in 0 until n step step) {
                for (s in 0 until halfStep) {
                    val twIdx = s * twiddleStride
                    val tRe = twiddleReal[twIdx]
                    val tIm = twiddleImag[twIdx]
                    val idx1 = k + s
                    val idx2 = k + s + halfStep
                    val re2 = real[idx2] * tRe - imag[idx2] * tIm
                    val im2 = real[idx2] * tIm + imag[idx2] * tRe
                    real[idx2] = real[idx1] - re2
                    imag[idx2] = imag[idx1] - im2
                    real[idx1] += re2
                    imag[idx1] += im2
                }
            }
            step *= 2
        }
    }

    companion object {
        private const val EPSILON = 1.1920929e-7f  // Float32 machine epsilon, matches torch.finfo(float32).eps
        private const val PI = kotlin.math.PI

        private fun nextPowerOf2(n: Int): Int {
            var v = n - 1
            v = v or (v shr 1)
            v = v or (v shr 2)
            v = v or (v shr 4)
            v = v or (v shr 8)
            v = v or (v shr 16)
            return v + 1
        }
    }
}
