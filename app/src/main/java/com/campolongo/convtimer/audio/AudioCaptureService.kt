package com.campolongo.convtimer.audio

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.Closeable

class AudioCaptureService : Closeable {

    companion object {
        const val SAMPLE_RATE = 16000
        const val FRAME_SIZE = 512 // samples per frame (~32ms at 16kHz)
        private const val TAG = "AudioCapture"
    }

    private var audioRecord: AudioRecord? = null
    private val _frames = MutableSharedFlow<ShortArray>(extraBufferCapacity = 64)
    val frames: SharedFlow<ShortArray> = _frames

    @Volatile
    var isCapturing = false
        private set

    fun initialize(context: android.content.Context): Boolean {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "RECORD_AUDIO permission not granted")
            return false
        }

        val bufferSize = maxOf(
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            ),
            FRAME_SIZE * 4 * 2 // at least 4 frames worth
        )

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        return audioRecord?.state == AudioRecord.STATE_INITIALIZED
    }

    suspend fun startCapturing() = withContext(Dispatchers.IO) {
        val recorder = audioRecord ?: return@withContext
        recorder.startRecording()
        isCapturing = true
        Log.d(TAG, "Started capturing audio")

        val buffer = ShortArray(FRAME_SIZE)
        while (isActive && isCapturing) {
            val read = recorder.read(buffer, 0, FRAME_SIZE)
            if (read == FRAME_SIZE) {
                _frames.emit(buffer.copyOf())
            } else if (read < 0) {
                Log.e(TAG, "AudioRecord.read error: $read")
                break
            }
        }
    }

    fun pauseCapturing() {
        isCapturing = false
        audioRecord?.stop()
        Log.d(TAG, "Paused capturing")
    }

    override fun close() {
        isCapturing = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        Log.d(TAG, "AudioCaptureService closed")
    }
}
