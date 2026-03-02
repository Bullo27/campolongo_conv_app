package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.campolongo.convtimer.ui.theme.RecordGreen
import com.campolongo.convtimer.ui.theme.PauseOrange
import com.campolongo.convtimer.ui.theme.ResumeBlue
import com.campolongo.convtimer.ui.theme.StopRed
import com.campolongo.convtimer.viewmodel.RecordingState

@Composable
fun ControlButtons(
    recordingState: RecordingState,
    onRecord: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            ActionButton(
                text = "Record",
                icon = Icons.Default.FiberManualRecord,
                color = RecordGreen,
                enabled = recordingState == RecordingState.IDLE || recordingState == RecordingState.STOPPED,
                onClick = onRecord
            )
            ActionButton(
                text = "Pause",
                icon = Icons.Default.Pause,
                color = PauseOrange,
                enabled = recordingState == RecordingState.RECORDING,
                onClick = onPause
            )
            ActionButton(
                text = "Resume",
                icon = Icons.Default.PlayArrow,
                color = ResumeBlue,
                enabled = recordingState == RecordingState.PAUSED,
                onClick = onResume
            )
        }
        ActionButton(
            text = "Stop",
            icon = Icons.Default.Stop,
            color = StopRed,
            enabled = recordingState == RecordingState.RECORDING || recordingState == RecordingState.PAUSED,
            onClick = onStop
        )
    }
}

@Composable
private fun ActionButton(
    text: String,
    icon: ImageVector,
    color: Color,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        colors = ButtonDefaults.buttonColors(containerColor = color),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.height(40.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = text,
            tint = Color.White,
            modifier = Modifier.size(16.dp)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(text = text, fontSize = 14.sp, color = Color.White)
    }
}
