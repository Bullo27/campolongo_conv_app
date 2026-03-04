package com.campolongo.convtimer.audio

import android.util.Log
import kotlin.math.sqrt

/**
 * Identifies speakers by comparing MFCC embeddings via cosine similarity.
 * First speaker detected = Speaker A; a distinct new voice = Speaker B.
 *
 * B confirmation: requires [bConfirmFrames] consecutive below-threshold decisions
 * before establishing Speaker B's reference. Prevents single noisy frames from
 * creating a false Speaker B.
 */
class SpeakerIdentifier(
    private val similarityThreshold: Float = 0.80f,
    private val ambiguityMargin: Float = 0.10f,
    private val emaAlpha: Float = 0.1f,
    private val bConfirmFrames: Int = 2,
) {
    companion object {
        private const val TAG = "SpeakerID"
    }

    private var speakerARef: FloatArray? = null
    private var speakerBRef: FloatArray? = null
    private var lastSpeaker: Speaker? = null

    // B confirmation state
    private var bCandidateCount = 0
    private var bCandidateMfcc: FloatArray? = null

    fun identify(mfcc: FloatArray): Speaker {
        val refA = speakerARef
        if (refA == null) {
            speakerARef = mfcc.copyOf()
            lastSpeaker = Speaker.A
            Log.d(TAG, "Speaker A reference established")
            return Speaker.A
        }

        val simA = cosineSimilarity(mfcc, refA)
        val refB = speakerBRef

        if (refB == null) {
            if (simA >= similarityThreshold) {
                // Still Speaker A — reset B candidate streak
                bCandidateCount = 0
                bCandidateMfcc = null
                updateReference(Speaker.A, mfcc)
                lastSpeaker = Speaker.A
                return Speaker.A
            } else {
                // Below threshold — candidate B frame
                bCandidateCount++
                val candidateAcc = bCandidateMfcc
                if (candidateAcc == null) {
                    bCandidateMfcc = mfcc.copyOf()
                } else {
                    for (i in candidateAcc.indices) {
                        candidateAcc[i] += mfcc[i]
                    }
                }
                if (bCandidateCount >= bConfirmFrames) {
                    // Confirmed B — establish from averaged candidate MFCCs
                    val avg = bCandidateMfcc!!
                    for (i in avg.indices) {
                        avg[i] /= bCandidateCount
                    }
                    speakerBRef = avg
                    bCandidateCount = 0
                    bCandidateMfcc = null
                    lastSpeaker = Speaker.B
                    Log.d(TAG, "Speaker B confirmed after $bConfirmFrames frames (simA=%.3f)".format(simA))
                    return Speaker.B
                }
                // Not yet confirmed — return A as safe default
                // Don't EMA-update ref_a with B-candidate speech (would contaminate A's reference)
                lastSpeaker = Speaker.A
                return Speaker.A
            }
        }

        // Both speakers established — classify
        val simB = cosineSimilarity(mfcc, refB)
        Log.v(TAG, "simA=%.3f, simB=%.3f".format(simA, simB))

        val speaker = when {
            simA > simB + ambiguityMargin -> Speaker.A
            simB > simA + ambiguityMargin -> Speaker.B
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
        bCandidateCount = 0
        bCandidateMfcc = null
    }

    private fun updateReference(speaker: Speaker, mfcc: FloatArray) {
        val ref = if (speaker == Speaker.A) speakerARef else speakerBRef
        if (ref != null) {
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
