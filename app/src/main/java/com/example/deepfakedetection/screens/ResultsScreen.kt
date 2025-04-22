package com.example.deepfakedetection.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.material3.MaterialTheme
import java.util.Locale
import com.example.deepfakedetection.FrameResult
import com.example.deepfakedetection.screens.FrameResultCard

@Composable
fun ResultsScreen(
    results: List<FrameResult>,
    videoConfidence: Float?,
    onAnalyzeAnother: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(48.dp))

        // Show overall video confidence if this is a video
        videoConfidence?.let { conf ->
            Text(
                text = String.format(Locale.US, "Video Confidence: %.1f%%", conf * 100),
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(16.dp))
        }

        Button(onClick = onAnalyzeAnother) {
            Text("Analyze Another")
        }
        Spacer(modifier = Modifier.height(16.dp))
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            results.forEach { frameResult ->
                FrameResultCard(frameResult)
                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}