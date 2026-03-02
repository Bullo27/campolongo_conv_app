package com.campolongo.convtimer.ui.component

import androidx.compose.foundation.background
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.campolongo.convtimer.ui.theme.MetricBoxBackground
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
            .background(MetricBoxBackground, RoundedCornerShape(8.dp))
            .border(1.dp, TextLight.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
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
            textAlign = TextAlign.Center,
            lineHeight = 12.sp
        )
    }
}
