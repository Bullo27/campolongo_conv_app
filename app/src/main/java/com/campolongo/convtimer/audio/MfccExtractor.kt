package com.campolongo.convtimer.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin MFCC feature extraction.
 * Extracts 13 MFCC coefficients from a short audio frame.
 *
 * NOT thread-safe — pre-allocated buffers are reused across calls.
 */
class MfccExtractor(
    private val sampleRate: Int = 16000,
    private val numCoeffs: Int = 13,
    private val numFilters: Int = 26,
    private val fftSize: Int = 512,
) {
    // Pre-computed constant tables
    private val melFilterbank: Array<FloatArray> = createMelFilterbank()
    private val hammingWindow: FloatArray = FloatArray(fftSize) {
        (0.54 - 0.46 * cos(2.0 * PI * it / (fftSize - 1))).toFloat()
    }
    private val twiddleReal: List<FloatArray>
    private val twiddleImag: List<FloatArray>
    private val dctBasis: Array<FloatArray> = Array(numCoeffs) { i ->
        FloatArray(numFilters) { j ->
            cos(PI * i * (j + 0.5) / numFilters).toFloat()
        }
    }

    // Pre-allocated working buffers (reused across calls)
    private val frame = FloatArray(fftSize)
    private val fftReal = FloatArray(fftSize)
    private val fftImag = FloatArray(fftSize)
    private val powerBuf = FloatArray(fftSize / 2 + 1)
    private val melEnergies = FloatArray(numFilters)
    private val mfccBuf = FloatArray(numCoeffs)

    init {
        val reals = mutableListOf<FloatArray>()
        val imags = mutableListOf<FloatArray>()
        var step = 1
        while (step < fftSize) {
            val angleStep = -PI / step
            val r = FloatArray(step)
            val im = FloatArray(step)
            for (pair in 0 until step) {
                val angle = angleStep * pair
                r[pair] = cos(angle).toFloat()
                im[pair] = sin(angle).toFloat()
            }
            reals.add(r)
            imags.add(im)
            step *= 2
        }
        twiddleReal = reals
        twiddleImag = imags
    }

    fun extract(samples: ShortArray): FloatArray {
        val copyLen = minOf(samples.size, fftSize)

        // Pre-emphasis + Hamming window into frame buffer
        frame[0] = samples[0].toFloat() * hammingWindow[0]
        for (i in 1 until copyLen) {
            frame[i] = (samples[i].toFloat() - 0.97f * samples[i - 1].toFloat()) * hammingWindow[i]
        }
        for (i in copyLen until fftSize) {
            frame[i] = 0f
        }

        // FFT → power spectrum
        computePowerSpectrum()

        // Apply Mel filterbank
        val specSize = powerBuf.size
        for (i in 0 until numFilters) {
            var sum = 0f
            val filter = melFilterbank[i]
            for (j in 0 until specSize) {
                sum += filter[j] * powerBuf[j]
            }
            melEnergies[i] = if (sum > 1e-10f) ln(sum.toDouble()).toFloat() else -23.0f
        }

        // DCT using pre-computed basis
        for (i in 0 until numCoeffs) {
            var sum = 0f
            val basis = dctBasis[i]
            for (j in 0 until numFilters) {
                sum += melEnergies[j] * basis[j]
            }
            mfccBuf[i] = sum
        }

        return mfccBuf.copyOf()
    }

    private fun computePowerSpectrum() {
        frame.copyInto(fftReal)
        fftImag.fill(0f)

        // Bit-reversal permutation
        val n = fftSize
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = fftReal[i]; fftReal[i] = fftReal[j]; fftReal[j] = tempR
                val tempI = fftImag[i]; fftImag[i] = fftImag[j]; fftImag[j] = tempI
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // FFT butterfly with pre-computed twiddle factors
        var step = 1
        var stageIdx = 0
        while (step < n) {
            val wr = twiddleReal[stageIdx]
            val wi = twiddleImag[stageIdx]
            for (group in 0 until n step step * 2) {
                for (pair in 0 until step) {
                    val idx1 = group + pair
                    val idx2 = idx1 + step
                    val tr = wr[pair] * fftReal[idx2] - wi[pair] * fftImag[idx2]
                    val ti = wr[pair] * fftImag[idx2] + wi[pair] * fftReal[idx2]
                    fftReal[idx2] = fftReal[idx1] - tr
                    fftImag[idx2] = fftImag[idx1] - ti
                    fftReal[idx1] += tr
                    fftImag[idx1] += ti
                }
            }
            step *= 2
            stageIdx++
        }

        // Power spectrum (first half + 1)
        val specSize = n / 2 + 1
        for (i in 0 until specSize) {
            powerBuf[i] = (fftReal[i] * fftReal[i] + fftImag[i] * fftImag[i]) / n
        }
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val specSize = fftSize / 2 + 1
        val lowMel = hzToMel(0f)
        val highMel = hzToMel(sampleRate / 2f)

        val melPoints = FloatArray(numFilters + 2)
        for (i in melPoints.indices) {
            melPoints[i] = lowMel + i * (highMel - lowMel) / (numFilters + 1)
        }

        val binPoints = IntArray(melPoints.size)
        for (i in melPoints.indices) {
            val hz = melToHz(melPoints[i])
            binPoints[i] = floor(hz * (fftSize + 1) / sampleRate).toInt()
        }

        val filterbank = Array(numFilters) { FloatArray(specSize) }
        for (i in 0 until numFilters) {
            for (j in binPoints[i] until binPoints[i + 1]) {
                if (j < specSize) {
                    filterbank[i][j] = (j - binPoints[i]).toFloat() /
                            maxOf(1, binPoints[i + 1] - binPoints[i])
                }
            }
            for (j in binPoints[i + 1] until binPoints[i + 2]) {
                if (j < specSize) {
                    filterbank[i][j] = (binPoints[i + 2] - j).toFloat() /
                            maxOf(1, binPoints[i + 2] - binPoints[i + 1])
                }
            }
        }
        return filterbank
    }

    private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)
}
