package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campolongo.convtimer.ui.theme.MicRed

@Composable
fun MicrophoneIcon(modifier: Modifier = Modifier) {
    Icon(
        imageVector = Icons.Default.Mic,
        contentDescription = "Microphone",
        modifier = modifier.size(72.dp),
        tint = MicRed
    )
}
