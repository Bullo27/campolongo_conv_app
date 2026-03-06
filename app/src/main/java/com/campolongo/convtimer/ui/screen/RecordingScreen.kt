package com.campolongo.convtimer.ui.screen

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.campolongo.convtimer.audio.NoiseLevel
import com.campolongo.convtimer.audio.OverlapMode
import com.campolongo.convtimer.ui.theme.PauseOrange
import com.campolongo.convtimer.ui.component.ControlButtons
import com.campolongo.convtimer.ui.component.MetricsGrid
import com.campolongo.convtimer.ui.component.MicrophoneIcon
import com.campolongo.convtimer.ui.component.RecordingTimeDisplay
import com.campolongo.convtimer.viewmodel.RecordingState
import com.campolongo.convtimer.viewmodel.RecordingViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordingScreen(viewModel: RecordingViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            viewModel.onPermissionGranted()
            viewModel.onRecord()
        } else {
            viewModel.onPermissionDenied()
        }
    }

    LaunchedEffect(uiState.permissionNeeded) {
        if (uiState.permissionNeeded) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Voice Recorder",
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                }
            )
        }
    ) { paddingValues ->
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            Spacer(modifier = Modifier.height(16.dp))

            MicrophoneIcon()

            Spacer(modifier = Modifier.height(8.dp))

            if (uiState.recordingState == RecordingState.CALIBRATING) {
                Text(
                    text = "Calibrating noise level...",
                    fontSize = 16.sp,
                    color = PauseOrange
                )
            } else {
                RecordingTimeDisplay(trtMs = uiState.metrics.trt)
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Noise level indicator
            if (uiState.recordingState != RecordingState.IDLE) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Noise:", fontSize = 12.sp)
                    NoiseLevel.entries.forEach { level ->
                        FilterChip(
                            selected = uiState.noiseLevel == level,
                            onClick = { viewModel.onNoiseLevel(level) },
                            label = { Text(level.label, fontSize = 11.sp) }
                        )
                    }
                }
                Spacer(modifier = Modifier.height(4.dp))
                // Overlap mode toggle
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Overlap:", fontSize = 12.sp)
                    FilterChip(
                        selected = uiState.overlapMode == OverlapMode.LOW_NOISE,
                        onClick = { viewModel.onOverlapMode(OverlapMode.LOW_NOISE) },
                        label = { Text("Low Noise", fontSize = 11.sp) }
                    )
                    FilterChip(
                        selected = uiState.overlapMode == OverlapMode.ADAPTIVE_NOISE,
                        onClick = { viewModel.onOverlapMode(OverlapMode.ADAPTIVE_NOISE) },
                        label = { Text("Adaptive", fontSize = 11.sp) }
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            ControlButtons(
                recordingState = uiState.recordingState,
                onRecord = viewModel::onRecord,
                onPause = viewModel::onPause,
                onResume = viewModel::onResume,
                onStop = viewModel::onStop
            )

            Spacer(modifier = Modifier.height(24.dp))

            MetricsGrid(
                metrics = uiState.metrics,
                modifier = Modifier.weight(1f)
            )
        }
    }
}
