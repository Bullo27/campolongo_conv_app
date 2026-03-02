package com.campolongo.convtimer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.campolongo.convtimer.ui.theme.ConversationTimerTheme
import com.campolongo.convtimer.ui.screen.RecordingScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ConversationTimerTheme {
                RecordingScreen()
            }
        }
    }
}
