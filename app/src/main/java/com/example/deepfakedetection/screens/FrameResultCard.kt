package com.example.deepfakedetection.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import com.example.deepfakedetection.FrameResult
import java.util.Locale

@Composable
fun FrameResultCard(result: FrameResult) {
    val rawProbs = result.probabilities ?: emptyList()
    // Determine labels and probabilities dynamically; handle binary real vs fake
    val (labels, probs) = when (rawProbs.size) {
        1 -> Pair(listOf("Real", "Fake"), listOf(rawProbs[0], 1f - rawProbs[0]))
        2 -> Pair(listOf("Real", "Fake"), rawProbs)
        else -> Pair(listOf("Real", "FE_Fake", "EFS_Fake", "FR_Fake", "FS_Fake"), rawProbs)
    }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp)),
        elevation = CardDefaults.cardElevation(6.dp)
    ) {
        Column(
            modifier = Modifier
                .background(MaterialTheme.colorScheme.surface)
                .padding(16.dp)
        ) {
            Text(
                text = result.label,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(Modifier.height(12.dp))
            Image(
                bitmap = result.bitmap.asImageBitmap(),
                contentDescription = result.label,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp)
                    .clip(RoundedCornerShape(12.dp))
            )
            Spacer(Modifier.height(12.dp))
            if (probs.isNotEmpty()) {
                Column {
                    labels.forEachIndexed { i, label ->
                        val value = probs.getOrNull(i) ?: 0f
                        val percent = value * 100
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(vertical = 2.dp)
                        ) {
                            Text(
                                text = label,
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.width(80.dp)
                            )
                            LinearProgressIndicator(
                                progress = value.coerceIn(0f, 1f),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                            )
                            Text(
                                text = String.format(Locale.US, " %.1f%%", percent),
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(start = 8.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}