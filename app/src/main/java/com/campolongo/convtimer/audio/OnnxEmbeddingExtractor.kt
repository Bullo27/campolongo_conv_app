package com.campolongo.convtimer.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.Closeable
import java.nio.FloatBuffer

/**
 * Extracts 192-dim speaker embeddings using WeSpeaker ECAPA-TDNN-512 ONNX model.
 *
 * Pipeline: int16 audio → Fbank features [T,80] → ONNX inference → 192-dim embedding.
 */
class OnnxEmbeddingExtractor(context: Context) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val fbankExtractor = FbankExtractor()

    init {
        val modelBytes = context.assets.open(MODEL_NAME).readBytes()
        session = env.createSession(modelBytes)
        inputName = session.inputNames.first()
    }

    /**
     * Extract a 192-dim speaker embedding from raw int16 audio.
     * @param audioInt16 raw 16kHz mono int16 audio samples
     * @return 192-dim FloatArray embedding
     */
    fun extractEmbedding(audioInt16: ShortArray): FloatArray {
        // 1. Extract Fbank features
        val (fbank, numFrames) = fbankExtractor.extract(audioInt16)
        if (numFrames == 0) return FloatArray(EMBEDDING_DIM)

        // 2. Create ONNX input tensor: shape [1, T, 80]
        val shape = longArrayOf(1, numFrames.toLong(), NUM_MEL_BINS.toLong())
        val buffer = FloatBuffer.wrap(fbank)
        val inputTensor = OnnxTensor.createTensor(env, buffer, shape)

        // 3. Run inference
        val results = session.run(mapOf(inputName to inputTensor))

        // 4. Extract embedding
        @Suppress("UNCHECKED_CAST")
        val outputArray = results[0].value as Array<FloatArray>
        val embedding = outputArray[0].copyOf()

        // 5. Cleanup
        inputTensor.close()
        results.close()

        return embedding
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val MODEL_NAME = "voxceleb_ECAPA512_LM.onnx"
        const val EMBEDDING_DIM = 192
        private const val NUM_MEL_BINS = 80
    }
}
