package com.campolongo.convtimer.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin MFCC feature extraction.
 * Extracts 13 MFCC coefficients from a short audio segment.
 */
class MfccExtractor(
    private val sampleRate: Int = 16000,
    private val numCoeffs: Int = 13,
    private val numFilters: Int = 26,
    private val fftSize: Int = 512,
) {
    private val melFilterbank: Array<FloatArray> = createMelFilterbank()

    fun extract(samples: ShortArray): FloatArray {
        // Convert to float and apply pre-emphasis
        val signal = FloatArray(samples.size)
        signal[0] = samples[0].toFloat()
        for (i in 1 until samples.size) {
            signal[i] = samples[i].toFloat() - 0.97f * samples[i - 1].toFloat()
        }

        // Take center portion or pad to fftSize
        val frame = FloatArray(fftSize)
        val copyLen = minOf(signal.size, fftSize)
        val offset = maxOf(0, (signal.size - fftSize) / 2)
        for (i in 0 until copyLen) {
            frame[i] = signal[offset + i]
        }

        // Apply Hamming window
        for (i in frame.indices) {
            frame[i] *= (0.54f - 0.46f * cos(2.0 * PI * i / (fftSize - 1)).toFloat())
        }

        // Compute FFT and power spectrum
        val powerSpectrum = computePowerSpectrum(frame)

        // Apply Mel filterbank
        val melEnergies = FloatArray(numFilters)
        for (i in 0 until numFilters) {
            var sum = 0f
            for (j in powerSpectrum.indices) {
                sum += melFilterbank[i][j] * powerSpectrum[j]
            }
            melEnergies[i] = if (sum > 1e-10f) ln(sum.toDouble()).toFloat() else -23.0f // ln(1e-10)
        }

        // Apply DCT to get MFCCs
        val mfcc = FloatArray(numCoeffs)
        for (i in 0 until numCoeffs) {
            var sum = 0f
            for (j in 0 until numFilters) {
                sum += melEnergies[j] * cos(PI * i * (j + 0.5) / numFilters).toFloat()
            }
            mfcc[i] = sum
        }

        return mfcc
    }

    private fun computePowerSpectrum(frame: FloatArray): FloatArray {
        // Radix-2 Cooley-Tukey FFT
        val n = frame.size
        val real = frame.copyOf()
        val imag = FloatArray(n)

        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = real[i]; real[i] = real[j]; real[j] = tempR
                val tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // FFT butterfly
        var step = 1
        while (step < n) {
            val angleStep = -PI / step
            for (group in 0 until n step step * 2) {
                for (pair in 0 until step) {
                    val angle = angleStep * pair
                    val wr = cos(angle).toFloat()
                    val wi = sin(angle).toFloat()
                    val idx1 = group + pair
                    val idx2 = idx1 + step
                    val tr = wr * real[idx2] - wi * imag[idx2]
                    val ti = wr * imag[idx2] + wi * real[idx2]
                    real[idx2] = real[idx1] - tr
                    imag[idx2] = imag[idx1] - ti
                    real[idx1] += tr
                    imag[idx1] += ti
                }
            }
            step *= 2
        }

        // Power spectrum (only first half + 1)
        val specSize = n / 2 + 1
        val power = FloatArray(specSize)
        for (i in 0 until specSize) {
            power[i] = (real[i] * real[i] + imag[i] * imag[i]) / n
        }
        return power
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val specSize = fftSize / 2 + 1
        val lowMel = hzToMel(0f)
        val highMel = hzToMel(sampleRate / 2f)

        // Create equally spaced points in Mel scale
        val melPoints = FloatArray(numFilters + 2)
        for (i in melPoints.indices) {
            melPoints[i] = lowMel + i * (highMel - lowMel) / (numFilters + 1)
        }

        // Convert back to Hz and then to FFT bin indices
        val binPoints = IntArray(melPoints.size)
        for (i in melPoints.indices) {
            val hz = melToHz(melPoints[i])
            binPoints[i] = floor(hz * (fftSize + 1) / sampleRate).toInt()
        }

        // Create triangular filters
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
