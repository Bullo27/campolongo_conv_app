package com.campolongo.convtimer.audio

import android.util.Log
import com.konovalov.vad.silero.config.Mode
import kotlin.math.log10
import kotlin.math.sqrt

enum class NoiseLevel(val label: String, val vadMode: Mode) {
    QUIET("Quiet", Mode.NORMAL),
    MODERATE("Moderate", Mode.NORMAL),      // Use NORMAL for moderate too — AGGRESSIVE may be too strict
    NOISY("Noisy", Mode.VERY_AGGRESSIVE);
}

class NoiseCalibrator {

    companion object {
        private const val TAG = "NoiseCalibrator"
        private const val QUIET_THRESHOLD_DBFS = -40.0
        private const val NOISY_THRESHOLD_DBFS = -25.0
    }

    fun calibrate(frames: List<ShortArray>): NoiseLevel {
        if (frames.isEmpty()) return NoiseLevel.QUIET

        // Compute RMS energy across all frames
        var sumSquares = 0.0
        var count = 0
        for (frame in frames) {
            for (sample in frame) {
                sumSquares += sample.toDouble() * sample.toDouble()
                count++
            }
        }

        val rms = sqrt(sumSquares / count)
        // Convert to dBFS (16-bit audio, max value = 32767)
        val dbfs = if (rms > 0) 20.0 * log10(rms / 32767.0) else -96.0

        Log.d(TAG, "Calibration: RMS=${"%.1f".format(rms)}, dBFS=${"%.1f".format(dbfs)}")

        return when {
            dbfs < QUIET_THRESHOLD_DBFS -> NoiseLevel.QUIET
            dbfs < NOISY_THRESHOLD_DBFS -> NoiseLevel.MODERATE
            else -> NoiseLevel.NOISY
        }
    }
}
