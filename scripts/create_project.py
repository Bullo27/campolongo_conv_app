#!/usr/bin/env python3
"""
Create the Conversation Timer Android project scaffolding.
Generates all Gradle files, manifest, and directory structure.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = PROJECT_ROOT / "app"
SRC_MAIN = APP_DIR / "src" / "main"
SRC_TEST = APP_DIR / "src" / "test"
SRC_ANDROID_TEST = APP_DIR / "src" / "androidTest"
PKG_PATH = "com/campolongo/convtimer"
JAVA_MAIN = SRC_MAIN / "java" / PKG_PATH
JAVA_TEST = SRC_TEST / "java" / PKG_PATH
JAVA_ANDROID_TEST = SRC_ANDROID_TEST / "java" / PKG_PATH
RES = SRC_MAIN / "res"


def create_dirs():
    """Create all project directories."""
    dirs = [
        PROJECT_ROOT / "gradle",
        APP_DIR / "src" / "main" / "assets",
        JAVA_MAIN / "audio",
        JAVA_MAIN / "state",
        JAVA_MAIN / "viewmodel",
        JAVA_MAIN / "ui" / "theme",
        JAVA_MAIN / "ui" / "screen",
        JAVA_MAIN / "ui" / "component",
        JAVA_TEST / "state",
        JAVA_TEST / "audio",
        JAVA_ANDROID_TEST / "audio",
        RES / "values",
        RES / "mipmap-hdpi",
        RES / "mipmap-mdpi",
        RES / "mipmap-xhdpi",
        RES / "mipmap-xxhdpi",
        RES / "mipmap-xxxhdpi",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d.relative_to(PROJECT_ROOT)}")


def write_file(path, content):
    """Write a file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Wrote: {path.relative_to(PROJECT_ROOT)}")


def create_gradle_files():
    """Create all Gradle configuration files."""

    # Version catalog
    write_file(PROJECT_ROOT / "gradle" / "libs.versions.toml", """\
[versions]
agp = "8.7.3"
kotlin = "2.1.0"
compose-bom = "2025.01.01"
lifecycle = "2.8.7"
activity-compose = "1.9.3"
android-vad-silero = "2.0.10"
coroutines = "1.9.0"
junit = "4.13.2"
junit-ext = "1.2.1"
espresso = "3.6.1"

[libraries]
compose-bom = { group = "androidx.compose", name = "compose-bom", version.ref = "compose-bom" }
compose-ui = { group = "androidx.compose.ui", name = "ui" }
compose-material3 = { group = "androidx.compose.material3", name = "material3" }
compose-material-icons = { group = "androidx.compose.material", name = "material-icons-extended" }
compose-ui-tooling-preview = { group = "androidx.compose.ui", name = "ui-tooling-preview" }
compose-ui-tooling = { group = "androidx.compose.ui", name = "ui-tooling" }
lifecycle-viewmodel-compose = { group = "androidx.lifecycle", name = "lifecycle-viewmodel-compose", version.ref = "lifecycle" }
lifecycle-runtime-compose = { group = "androidx.lifecycle", name = "lifecycle-runtime-compose", version.ref = "lifecycle" }
activity-compose = { group = "androidx.activity", name = "activity-compose", version.ref = "activity-compose" }
android-vad-silero = { module = "com.github.gkonovalov.android-vad:silero", version.ref = "android-vad-silero" }
kotlinx-coroutines-android = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }
junit = { group = "junit", name = "junit", version.ref = "junit" }
junit-ext = { group = "androidx.test.ext", name = "junit", version.ref = "junit-ext" }
espresso-core = { group = "androidx.test.espresso", name = "espresso-core", version.ref = "espresso" }

[plugins]
android-application = { id = "com.android.application", version.ref = "agp" }
kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
kotlin-compose = { id = "org.jetbrains.kotlin.plugin.compose", version.ref = "kotlin" }
""")

    # Root build.gradle.kts
    write_file(PROJECT_ROOT / "build.gradle.kts", """\
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
""")

    # Settings
    write_file(PROJECT_ROOT / "settings.gradle.kts", """\
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "ConversationTimer"
include(":app")
""")

    # Gradle properties
    write_file(PROJECT_ROOT / "gradle.properties", """\
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
android.nonTransitiveRClass=true
""")

    # App build.gradle.kts
    write_file(APP_DIR / "build.gradle.kts", """\
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.campolongo.convtimer"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.campolongo.convtimer"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.android.vad.silero)
    implementation(libs.kotlinx.coroutines.android)

    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.junit)
    androidTestImplementation(libs.junit.ext)
    androidTestImplementation(libs.espresso.core)
}
""")

    # ProGuard rules (minimal for now)
    write_file(APP_DIR / "proguard-rules.pro", """\
# Conversation Timer ProGuard rules
# Keep ONNX Runtime classes if we add neural speaker embeddings later
# -keep class ai.onnxruntime.** { *; }
""")


def create_manifest():
    """Create AndroidManifest.xml."""
    write_file(SRC_MAIN / "AndroidManifest.xml", """\
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.RECORD_AUDIO" />

    <application
        android:name=".ConversationTimerApp"
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:supportsRtl="true"
        android:theme="@style/Theme.ConversationTimer">

        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@style/Theme.ConversationTimer">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
""")


def create_resources():
    """Create Android resource files."""
    write_file(RES / "values" / "strings.xml", """\
<resources>
    <string name="app_name">Conversation Timer</string>
</resources>
""")

    write_file(RES / "values" / "colors.xml", """\
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="white">#FFFFFFFF</color>
    <color name="black">#FF000000</color>
</resources>
""")

    write_file(RES / "values" / "themes.xml", """\
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <style name="Theme.ConversationTimer" parent="android:Theme.Material.Light.NoActionBar" />
</resources>
""")


def create_kotlin_sources():
    """Create all Kotlin source files."""

    # Application class
    write_file(JAVA_MAIN / "ConversationTimerApp.kt", """\
package com.campolongo.convtimer

import android.app.Application

class ConversationTimerApp : Application()
""")

    # MainActivity
    write_file(JAVA_MAIN / "MainActivity.kt", """\
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
""")

    # Theme
    write_file(JAVA_MAIN / "ui" / "theme" / "Color.kt", """\
package com.campolongo.convtimer.ui.theme

import androidx.compose.ui.graphics.Color

val RecordGreen = Color(0xFF4CAF50)
val PauseOrange = Color(0xFFFF9800)
val ResumeBlue = Color(0xFF2196F3)
val StopRed = Color(0xFFF44336)
val MetricBoxBackground = Color(0xFFF5F5F5)
val MetricValueBlue = Color(0xFF1565C0)
val MicRed = Color(0xFFE53935)
val TimerGreen = Color(0xFF43A047)
val TextDark = Color(0xFF212121)
val TextMedium = Color(0xFF616161)
val TextLight = Color(0xFF9E9E9E)
""")

    write_file(JAVA_MAIN / "ui" / "theme" / "Theme.kt", """\
package com.campolongo.convtimer.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColorScheme = lightColorScheme(
    primary = RecordGreen,
    secondary = PauseOrange,
    tertiary = ResumeBlue,
    error = StopRed,
    background = androidx.compose.ui.graphics.Color.White,
    surface = androidx.compose.ui.graphics.Color.White,
)

@Composable
fun ConversationTimerTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColorScheme,
        content = content
    )
}
""")

    # Metrics data model
    write_file(JAVA_MAIN / "state" / "MetricsAccumulator.kt", """\
package com.campolongo.convtimer.state

data class MetricsSnapshot(
    val trt: Long = 0L,
    val wta: Long = 0L,
    val wtb: Long = 0L,
    val sta: Long = 0L,
    val stb: Long = 0L,
    val stm: Long = 0L,
) {
    val cta: Long get() = wta + sta
    val ctb: Long get() = wtb + stb
    val tct: Long get() = cta + stm + ctb
    val tst: Long get() = sta + stb + stm
    val bfst: Long get() = trt - tct
}
""")

    # ViewModel
    write_file(JAVA_MAIN / "viewmodel" / "RecordingViewModel.kt", """\
package com.campolongo.convtimer.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.campolongo.convtimer.state.MetricsSnapshot
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

enum class RecordingState {
    IDLE, RECORDING, PAUSED, STOPPED
}

data class UiState(
    val recordingState: RecordingState = RecordingState.IDLE,
    val metrics: MetricsSnapshot = MetricsSnapshot(),
)

class RecordingViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var timerJob: Job? = null
    private var trtMs: Long = 0L

    fun onRecord() {
        _uiState.value = UiState(recordingState = RecordingState.RECORDING)
        trtMs = 0L
        startTimer()
    }

    fun onPause() {
        timerJob?.cancel()
        _uiState.value = _uiState.value.copy(recordingState = RecordingState.PAUSED)
    }

    fun onResume() {
        _uiState.value = _uiState.value.copy(recordingState = RecordingState.RECORDING)
        startTimer()
    }

    fun onStop() {
        timerJob?.cancel()
        _uiState.value = _uiState.value.copy(recordingState = RecordingState.STOPPED)
    }

    private fun startTimer() {
        timerJob = viewModelScope.launch {
            while (true) {
                delay(100L)
                trtMs += 100L
                _uiState.value = _uiState.value.copy(
                    metrics = _uiState.value.metrics.copy(trt = trtMs)
                )
            }
        }
    }
}
""")

    # UI Components
    write_file(JAVA_MAIN / "ui" / "component" / "MicrophoneIcon.kt", """\
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
""")

    write_file(JAVA_MAIN / "ui" / "component" / "RecordingTimeDisplay.kt", """\
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
""")

    write_file(JAVA_MAIN / "ui" / "component" / "ControlButtons.kt", """\
package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
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
                color = RecordGreen,
                enabled = recordingState == RecordingState.IDLE || recordingState == RecordingState.STOPPED,
                onClick = onRecord
            )
            ActionButton(
                text = "Pause",
                color = PauseOrange,
                enabled = recordingState == RecordingState.RECORDING,
                onClick = onPause
            )
            ActionButton(
                text = "Resume",
                color = ResumeBlue,
                enabled = recordingState == RecordingState.PAUSED,
                onClick = onResume
            )
        }
        ActionButton(
            text = "Stop",
            color = StopRed,
            enabled = recordingState == RecordingState.RECORDING || recordingState == RecordingState.PAUSED,
            onClick = onStop
        )
    }
}

@Composable
private fun ActionButton(
    text: String,
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
        Text(text = text, fontSize = 14.sp, color = Color.White)
    }
}
""")

    write_file(JAVA_MAIN / "ui" / "component" / "MetricBox.kt", """\
package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.campolongo.convtimer.ui.theme.MetricValueBlue
import com.campolongo.convtimer.ui.theme.TextDark
import com.campolongo.convtimer.ui.theme.TextLight

@Composable
fun MetricBox(
    abbreviation: String,
    value: Long,
    description: String,
    modifier: Modifier = Modifier,
) {
    val displayValue = "%.1f".format(value / 1000.0)

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .border(1.dp, TextLight, RoundedCornerShape(8.dp))
            .padding(12.dp)
            .fillMaxWidth()
    ) {
        Text(
            text = abbreviation,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = TextDark
        )
        Text(
            text = displayValue,
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = MetricValueBlue
        )
        Text(
            text = description,
            fontSize = 10.sp,
            color = TextLight,
            lineHeight = 12.sp
        )
    }
}
""")

    write_file(JAVA_MAIN / "ui" / "component" / "MetricsGrid.kt", """\
package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campolongo.convtimer.state.MetricsSnapshot

data class MetricDef(
    val abbreviation: String,
    val description: String,
    val valueSelector: (MetricsSnapshot) -> Long,
)

private val metricDefs = listOf(
    MetricDef("TRT", "Total recording time") { it.trt },
    MetricDef("TCT", "Total conversation time\\n= CTA+STM+CTB") { it.tct },
    MetricDef("WTA", "Spoken words time\\nof speaker A") { it.wta },
    MetricDef("WTB", "Spoken words time\\nof speaker B") { it.wtb },
    MetricDef("STA", "Silence time between\\nwords of speaker A") { it.sta },
    MetricDef("STB", "Silence time between\\nwords of speaker B") { it.stb },
    MetricDef("CTA", "Conversation time\\nof speaker A = WTA+STA") { it.cta },
    MetricDef("CTB", "Conversation time\\nof speaker B = WTB+STB") { it.ctb },
    MetricDef("STM", "Silence time between\\nspeaker A and speaker B") { it.stm },
    MetricDef("TST", "Total silence time\\n= STA+STB+STM") { it.tst },
    MetricDef("BFST", "Beginning and final\\nsilence time = TRT-TCT") { it.bfst },
)

@Composable
fun MetricsGrid(metrics: MetricsSnapshot, modifier: Modifier = Modifier) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier.verticalScroll(rememberScrollState())
    ) {
        for (i in metricDefs.indices step 2) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                MetricBox(
                    abbreviation = metricDefs[i].abbreviation,
                    value = metricDefs[i].valueSelector(metrics),
                    description = metricDefs[i].description,
                    modifier = Modifier.weight(1f)
                )
                if (i + 1 < metricDefs.size) {
                    MetricBox(
                        abbreviation = metricDefs[i + 1].abbreviation,
                        value = metricDefs[i + 1].valueSelector(metrics),
                        description = metricDefs[i + 1].description,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}
""")

    # Recording Screen
    write_file(JAVA_MAIN / "ui" / "screen" / "RecordingScreen.kt", """\
package com.campolongo.convtimer.ui.screen

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.campolongo.convtimer.ui.component.ControlButtons
import com.campolongo.convtimer.ui.component.MetricsGrid
import com.campolongo.convtimer.ui.component.MicrophoneIcon
import com.campolongo.convtimer.ui.component.RecordingTimeDisplay
import com.campolongo.convtimer.viewmodel.RecordingViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordingScreen(viewModel: RecordingViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()

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

            RecordingTimeDisplay(trtMs = uiState.metrics.trt)

            Spacer(modifier = Modifier.height(16.dp))

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
""")

    # Placeholder files for audio package (to be implemented in later phases)
    write_file(JAVA_MAIN / "audio" / "AudioPipelineEvent.kt", """\
package com.campolongo.convtimer.audio

sealed class AudioPipelineEvent {
    data class SpeechDetected(val speaker: Speaker) : AudioPipelineEvent()
    data object SilenceDetected : AudioPipelineEvent()
    data class Error(val message: String) : AudioPipelineEvent()
}

enum class Speaker { A, B }
""")


def create_gradle_wrapper():
    """Create a script to generate the Gradle wrapper."""
    # We'll download the Gradle wrapper jar and scripts
    write_file(PROJECT_ROOT / "scripts" / "setup_gradle_wrapper.py", """\
#!/usr/bin/env python3
\"\"\"Download and set up Gradle wrapper for the project.\"\"\"
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    env = {**os.environ,
           'ANDROID_HOME': os.path.expanduser('~/android-sdk'),
           'PATH': os.path.expanduser('~/android-sdk/cmdline-tools/latest/bin') + ':' +
                   os.path.expanduser('~/android-sdk/platform-tools') + ':' +
                   os.environ['PATH']}

    # Check if gradle is available
    result = subprocess.run(['which', 'gradle'], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        # Download gradle wrapper using a minimal Gradle
        print("Downloading Gradle wrapper...")
        import urllib.request
        import zipfile
        import shutil

        gradle_url = "https://services.gradle.org/distributions/gradle-8.11.1-bin.zip"
        zip_path = Path("/tmp/gradle-bin.zip")
        extract_dir = Path("/tmp/gradle-extract")

        if not (extract_dir / "gradle-8.11.1").exists():
            if not zip_path.exists():
                print(f"Downloading Gradle from {gradle_url}...")
                urllib.request.urlretrieve(gradle_url, zip_path)
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

        gradle_bin = str(extract_dir / "gradle-8.11.1" / "bin" / "gradle")
        os.chmod(gradle_bin, 0o755)
    else:
        gradle_bin = result.stdout.strip()

    # Generate wrapper
    print("Generating Gradle wrapper...")
    subprocess.run(
        [gradle_bin, 'wrapper', '--gradle-version', '8.11.1'],
        cwd=str(PROJECT_ROOT),
        check=True,
        env=env
    )

    # Make gradlew executable
    gradlew = PROJECT_ROOT / "gradlew"
    os.chmod(str(gradlew), 0o755)
    print(f"Gradle wrapper created at {gradlew}")

if __name__ == "__main__":
    main()
""")


def create_build_script():
    """Create the build helper script."""
    write_file(PROJECT_ROOT / "scripts" / "build_app.py", """\
#!/usr/bin/env python3
\"\"\"Build the Conversation Timer APK.\"\"\"
import subprocess
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    env = {**os.environ,
           'ANDROID_HOME': os.path.expanduser('~/android-sdk'),
           'PATH': os.path.expanduser('~/android-sdk/cmdline-tools/latest/bin') + ':' +
                   os.path.expanduser('~/android-sdk/platform-tools') + ':' +
                   os.environ['PATH']}

    build_type = sys.argv[1] if len(sys.argv) > 1 else "assembleDebug"
    gradlew = PROJECT_ROOT / "gradlew"

    print(f"Building {build_type}...")
    result = subprocess.run(
        [str(gradlew), build_type],
        cwd=str(PROJECT_ROOT),
        env=env
    )

    if result.returncode == 0:
        apk_dir = PROJECT_ROOT / "app" / "build" / "outputs" / "apk" / "debug"
        if apk_dir.exists():
            apks = list(apk_dir.glob("*.apk"))
            if apks:
                print(f"\\nAPK built: {apks[0]}")
    else:
        print("Build failed!", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
""")


def main():
    print("=" * 60)
    print("  Creating Conversation Timer project structure")
    print("=" * 60)

    print("\nCreating directories...")
    create_dirs()

    print("\nWriting Gradle configuration...")
    create_gradle_files()

    print("\nWriting AndroidManifest...")
    create_manifest()

    print("\nWriting resources...")
    create_resources()

    print("\nWriting Kotlin sources...")
    create_kotlin_sources()

    print("\nCreating Gradle wrapper setup script...")
    create_gradle_wrapper()

    print("\nCreating build script...")
    create_build_script()

    print("\n" + "=" * 60)
    print("  Project structure created!")
    print("  Next: run scripts/setup_gradle_wrapper.py")
    print("  Then: run scripts/build_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
