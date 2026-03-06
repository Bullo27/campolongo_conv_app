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
    MetricDef("TCT", "Total conversation time\n= CTA+STM+CTB") { it.tct },
    MetricDef("WTA", "Spoken words time\nof speaker A") { it.wta },
    MetricDef("WTB", "Spoken words time\nof speaker B") { it.wtb },
    MetricDef("STA", "Silence time between\nwords of speaker A") { it.sta },
    MetricDef("STB", "Silence time between\nwords of speaker B") { it.stb },
    MetricDef("CTA", "Conversation time\nof speaker A = WTA+STA") { it.cta },
    MetricDef("CTB", "Conversation time\nof speaker B = WTB+STB") { it.ctb },
    MetricDef("STM", "Silence time between\nspeaker A and speaker B") { it.stm },
    MetricDef("TST", "Total silence time\n= STA+STB+STM") { it.tst },
    MetricDef("BFST", "Beginning and final\nsilence time = TRT-TCT") { it.bfst },
    MetricDef("OVT", "Overlap time\n(both speaking)") { it.ovt },
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
