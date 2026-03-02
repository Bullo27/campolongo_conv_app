package com.campolongo.convtimer.audio

import android.util.Log
import kotlin.math.sqrt

/**
 * Identifies speakers by comparing MFCC embeddings via cosine similarity.
 * First speaker detected = Speaker A; a distinct new voice = Speaker B.
 */
class SpeakerIdentifier(
    private val similarityThreshold: Float = 0.80f,
    private val ambiguityMargin: Float = 0.10f, // hysteresis zone: threshold +/- margin
    private val emaAlpha: Float = 0.1f, // exponential moving average update rate
) {
    companion object {
        private const val TAG = "SpeakerID"
    }

    private var speakerARef: FloatArray? = null
    private var speakerBRef: FloatArray? = null
    private var lastSpeaker: Speaker? = null

    fun identify(mfcc: FloatArray): Speaker {
        val refA = speakerARef
        if (refA == null) {
            // First speech ever — this is Speaker A
            speakerARef = mfcc.copyOf()
            lastSpeaker = Speaker.A
            Log.d(TAG, "Speaker A reference established")
            return Speaker.A
        }

        val simA = cosineSimilarity(mfcc, refA)
        val refB = speakerBRef

        if (refB == null) {
            // Speaker B not yet established
            if (simA >= similarityThreshold) {
                // Still Speaker A
                updateReference(Speaker.A, mfcc)
                lastSpeaker = Speaker.A
                return Speaker.A
            } else {
                // New speaker detected — this is Speaker B
                speakerBRef = mfcc.copyOf()
                lastSpeaker = Speaker.B
                Log.d(TAG, "Speaker B reference established (simA=%.3f)".format(simA))
                return Speaker.B
            }
        }

        // Both speakers established — classify
        val simB = cosineSimilarity(mfcc, refB)
        Log.v(TAG, "simA=%.3f, simB=%.3f".format(simA, simB))

        val speaker = when {
            // Clear Speaker A
            simA > simB + ambiguityMargin -> Speaker.A
            // Clear Speaker B
            simB > simA + ambiguityMargin -> Speaker.B
            // Ambiguous — keep previous speaker (hysteresis)
            else -> lastSpeaker ?: Speaker.A
        }

        updateReference(speaker, mfcc)
        lastSpeaker = speaker
        return speaker
    }

    fun reset() {
        speakerARef = null
        speakerBRef = null
        lastSpeaker = null
    }

    private fun updateReference(speaker: Speaker, mfcc: FloatArray) {
        val ref = if (speaker == Speaker.A) speakerARef else speakerBRef
        if (ref != null) {
            // EMA update
            for (i in ref.indices) {
                ref[i] = (1 - emaAlpha) * ref[i] + emaAlpha * mfcc[i]
            }
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA * normB)
        return if (denom > 0f) dot / denom else 0f
    }
}
