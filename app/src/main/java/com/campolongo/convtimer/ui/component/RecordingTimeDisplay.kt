package com.campolongo.convtimer.ui.component

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.campolongo.convtimer.ui.theme.TimerGreen

@Composable
fun RecordingTimeDisplay(trtMs: Long, modifier: Modifier = Modifier) {
    val seconds = trtMs / 1000.0
    Text(
        text = "Recording Time: ${"%.1f".format(seconds)}s",
        fontSize = 18.sp,
        fontWeight = FontWeight.Medium,
        color = TimerGreen,
        modifier = modifier
    )
}
