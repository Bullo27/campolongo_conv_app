package com.campolongo.convtimer.audio

import android.content.Context
import android.util.Log
import com.konovalov.vad.silero.Vad
import com.konovalov.vad.silero.VadSilero
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import com.konovalov.vad.silero.config.SampleRate
import java.io.Closeable

class VadEngine : Closeable {

    companion object {
        private const val TAG = "VadEngine"
    }

    private var vad: VadSilero? = null

    fun initialize(context: Context, mode: Mode = Mode.NORMAL) {
        vad = Vad.builder()
            .setContext(context)
            .setSampleRate(SampleRate.SAMPLE_RATE_16K)
            .setFrameSize(FrameSize.FRAME_SIZE_512)
            .setMode(mode)
            .setSilenceDurationMs(300)
            .setSpeechDurationMs(50)
            .build()
        Log.d(TAG, "VAD initialized with mode: $mode")
    }

    fun setMode(mode: Mode) {
        // VadSilero doesn't support changing mode after creation,
        // but we can access the internal threshold. For now, we'll
        // just log it - a full re-initialization would be needed.
        Log.d(TAG, "Mode change requested: $mode (requires re-init)")
    }

    fun isSpeech(frame: ShortArray): Boolean {
        return vad?.isSpeech(frame) ?: false
    }

    override fun close() {
        vad?.close()
        vad = null
        Log.d(TAG, "VadEngine closed")
    }
}
