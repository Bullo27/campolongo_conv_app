package com.campolongo.convtimer.audio

import android.util.Log
import kotlin.math.sqrt

/**
 * Identifies speakers by comparing neural embeddings via cosine similarity.
 * First speaker detected = Speaker A; a distinct new voice = Speaker B.
 *
 * v2 neural pipeline:
 * - margin=0: pure nearest-neighbor classification
 * - References freeze after B is established (no EMA drift)
 * - minFramesForB: flush segments (< 47 frames) cannot trigger B establishment
 * - Exposes lastSimA/lastSimB for dual-assignment overlap detection
 */
class SpeakerIdentifier(
    private val similarityThreshold: Float = 0.80f,
    private val ambiguityMargin: Float = 0.0f,
    private val emaAlpha: Float = 0.1f,
    private val bConfirmFrames: Int = 2,
    private val minFramesForB: Int = 47,
) {
    companion object {
        private const val TAG = "SpeakerID"
    }

    var speakerARef: FloatArray? = null
        private set
    var speakerBRef: FloatArray? = null
        private set
    val isBEstablished: Boolean get() = speakerBRef != null
    private var lastSpeaker: Speaker? = null
    private var refsFrozen = false

    // B confirmation state
    private var bCandidateCount = 0
    private var bCandidateEmb: FloatArray? = null

    // Exposed for dual-assignment and adaptive noise
    var lastSimA: Float = 0f
        private set
    var lastSimB: Float = 0f
        private set

    /**
     * Identify the primary speaker for the given embedding.
     * @param embedding speaker embedding (e.g., 192-dim from WeSpeaker)
     * @param nFrames number of raw audio frames in this segment (0 = unknown/full)
     * @return Speaker.A or Speaker.B (never Speaker.BOTH — dual-assignment is in AudioPipeline)
     */
    fun identify(embedding: FloatArray, nFrames: Int = 0): Speaker {
        val refA = speakerARef

        // Step 1: First call — establish Speaker A
        if (refA == null) {
            speakerARef = embedding.copyOf()
            lastSpeaker = Speaker.A
            lastSimA = 1.0f
            lastSimB = 0.0f
            Log.d(TAG, "Speaker A reference established")
            return Speaker.A
        }

        val simA = cosineSimilarity(embedding, refA)
        lastSimA = simA
        val refB = speakerBRef

        // Step 2: Only A established — try to discover B
        if (refB == null) {
            lastSimB = 0.0f
            if (simA >= similarityThreshold) {
                // Still Speaker A — reset B candidate streak
                bCandidateCount = 0
                bCandidateEmb = null
                updateReference(Speaker.A, embedding)
                lastSpeaker = Speaker.A
                return Speaker.A
            } else {
                // Below threshold — potential B candidate
                val isFullSegment = nFrames == 0 || nFrames >= minFramesForB
                if (isFullSegment) {
                    bCandidateCount++
                    val acc = bCandidateEmb
                    if (acc == null) {
                        bCandidateEmb = embedding.copyOf()
                    } else {
                        for (i in acc.indices) acc[i] += embedding[i]
                    }
                    if (bCandidateCount >= bConfirmFrames) {
                        // Confirmed B — establish from averaged candidate embeddings
                        val avg = bCandidateEmb!!
                        for (i in avg.indices) avg[i] /= bCandidateCount
                        speakerBRef = avg
                        refsFrozen = true
                        bCandidateCount = 0
                        bCandidateEmb = null
                        lastSpeaker = Speaker.B
                        lastSimB = 1.0f
                        Log.d(TAG, "Speaker B confirmed after $bConfirmFrames segments (simA=%.3f)".format(simA))
                        return Speaker.B
                    }
                }
                // Not yet confirmed (or flush segment) — return A as safe default
                lastSpeaker = Speaker.A
                return Speaker.A
            }
        }

        // Step 3: Both speakers established — nearest-neighbor classification
        val simB = cosineSimilarity(embedding, refB)
        lastSimB = simB
        Log.v(TAG, "simA=%.3f, simB=%.3f".format(simA, simB))

        val speaker = when {
            simA > simB + ambiguityMargin -> Speaker.A
            simB > simA + ambiguityMargin -> Speaker.B
            else -> lastSpeaker ?: Speaker.A
        }

        // References are frozen after B established — no EMA updates
        lastSpeaker = speaker
        return speaker
    }

    fun reset() {
        speakerARef = null
        speakerBRef = null
        lastSpeaker = null
        refsFrozen = false
        bCandidateCount = 0
        bCandidateEmb = null
        lastSimA = 0f
        lastSimB = 0f
    }

    private fun updateReference(speaker: Speaker, embedding: FloatArray) {
        if (refsFrozen) return
        val ref = if (speaker == Speaker.A) speakerARef else speakerBRef
        if (ref != null) {
            for (i in ref.indices) {
                ref[i] = (1 - emaAlpha) * ref[i] + emaAlpha * embedding[i]
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
