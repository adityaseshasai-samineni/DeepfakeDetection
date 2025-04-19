package com.example.deepfakedetection

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.graphics.scale
import com.example.deepfakedetection.ui.theme.DeepfakeDetectionTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

sealed class Screen {
    data object Upload : Screen()
    data object Progress : Screen()
    data object Results : Screen()
}

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DeepfakeDetectionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current
    var tflite by remember { mutableStateOf<Interpreter?>(null) }
    var isModelLoaded by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var results by remember { mutableStateOf(listOf<FrameResult>()) }

    var screen by remember { mutableStateOf<Screen>(Screen.Upload) }
    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var selectedType by remember { mutableStateOf("") }
    var progress by remember { mutableFloatStateOf(0f) }
    var statusText by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                val assetManager = context.assets
                val modelFd = assetManager.openFd("model/efficientnet_lite_l4_model.tflite")
                val inputStream = modelFd.createInputStream()
                val modelBytes = inputStream.readBytes()
                val byteBuffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                    order(ByteOrder.nativeOrder())
                    put(modelBytes)
                }
                val options = Interpreter.Options()
                tflite = Interpreter(byteBuffer, options)
                isModelLoaded = true
            } catch (e: Exception) {
                errorMessage = "Failed to load model"
            }
        }
    }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedUri = it
            selectedType = "image"
            errorMessage = null
            screen = Screen.Progress
        }
    }
    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedUri = it
            selectedType = "video"
            errorMessage = null
            screen = Screen.Progress
        }
    }

    LaunchedEffect(selectedUri) {
        selectedUri?.let { uri ->
            withContext(Dispatchers.Default) {
                try {
                    if (selectedType == "image") {
                        statusText = "Loading image..."
                        progress = 0.1f
                        val bitmap = withContext(Dispatchers.IO) {
                            context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                        } ?: throw Exception("Failed to load image")
                        statusText = "Processing image..."
                        progress = 0.4f
                        val scaledBitmap = withContext(Dispatchers.Default) { bitmap.scale(256, 256, filter = true) }
                        val predictions = tflite?.let { interpreter ->
                            withContext(Dispatchers.Default) { runInference(interpreter, scaledBitmap) }
                        }
                        progress = 1f
                        results = listOf(
                            FrameResult(
                                label = "Uploaded Image",
                                bitmap = scaledBitmap,
                                probabilities = predictions?.toList()
                            )
                        )
                    } else {
                        val retriever = withContext(Dispatchers.IO) {
                            MediaMetadataRetriever().apply { setDataSource(context, uri) }
                        }
                        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                        val totalSeconds = (durationMs / 1000).toInt().coerceAtLeast(1)
                        val frameIndices = if (totalSeconds > 50) (0 until totalSeconds).shuffled().take(50) else (0 until totalSeconds).toList()
                        val frames = mutableListOf<FrameResult>()
                        for ((idx, sec) in frameIndices.withIndex()) {
                            statusText = "Processing frame ${idx + 1}/${frameIndices.size}"
                            progress = (idx + 1) / frameIndices.size.toFloat()
                            val frameBitmap = withContext(Dispatchers.IO) { retriever.getFrameAtTime(sec * 1_000_000L) }
                            frameBitmap?.let {
                                val scaled = withContext(Dispatchers.Default) { it.scale(256, 256, filter = true) }
                                val framePredictions = tflite?.let { interpreter ->
                                    withContext(Dispatchers.Default) { runInference(interpreter, scaled) }
                                }
                                frames.add(FrameResult("Frame at ${sec + 1} sec", scaled, framePredictions?.toList()))
                            }
                        }
                        withContext(Dispatchers.IO) { retriever.release() }
                        progress = 1f
                        results = frames
                    }
                    screen = Screen.Results
                } catch (e: Exception) {
                    errorMessage = "Failed to process media"
                    screen = Screen.Upload
                }
            }
        }
    }

    when (screen) {
        is Screen.Upload -> {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(MaterialTheme.colorScheme.background)
                    .padding(horizontal = 24.dp, vertical = 16.dp)
                    .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Deepfake Detection",
                            style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = "Upload an image or video to detect deepfakes using advanced AI.",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onBackground,
                            modifier = Modifier.padding(top = 8.dp, bottom = 24.dp)
                        )
                        Column(
                            verticalArrangement = Arrangement.spacedBy(16.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Button(
                                onClick = { imagePickerLauncher.launch("image/*") },
                                enabled = isModelLoaded,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Upload Image")
                            }
                            Button(
                                onClick = { videoPickerLauncher.launch("video/*") },
                                enabled = isModelLoaded,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Upload Video")
                            }
                        }
                    }
                }
                errorMessage?.let {
                    Text(
                        text = it,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(top = 24.dp)
                    )
                }
            }
        }
        is Screen.Progress -> {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = statusText,
                    style = MaterialTheme.typography.bodyLarge
                )
                Spacer(Modifier.height(16.dp))
                LinearProgressIndicator(progress = { progress })
                Spacer(Modifier.height(8.dp))
                Text(
                    text = String.format(Locale.US, "%.0f%%", progress * 100),
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
        is Screen.Results -> {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp)
                    .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                results.forEach { frameResult ->
                    FrameResultCard(frameResult)
                    Spacer(Modifier.height(16.dp))
                }
                Spacer(Modifier.weight(1f))
                Button(
                    onClick = { screen = Screen.Upload }
                ) {
                    Text("Analyze Another")
                }
            }
        }
    }
}

@Composable
fun FrameResultCard(result: FrameResult) {
    val labels = listOf("Real", "FE_Fake", "EFS_Fake", "FR_Fake", "FS_Fake")
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
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
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
            result.probabilities?.let { probabilitiesList ->
                Column {
                    labels.forEachIndexed { i, label ->
                        val percent = (probabilitiesList.getOrNull(i) ?: 0f) * 100
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
                                progress = { (probabilitiesList.getOrNull(i) ?: 0f).coerceIn(0f, 1f) },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp))
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

data class FrameResult(
    val label: String,
    val bitmap: Bitmap,
    val probabilities: List<Float>?
)

fun runInference(tflite: Interpreter, bitmap: Bitmap): FloatArray {
    val inputBuffer = convertBitmapToByteBuffer(bitmap)
    val output = Array(1) { FloatArray(5) }
    tflite.run(inputBuffer, output)
    return output[0]
}

fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    val inputBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4)
    inputBuffer.order(ByteOrder.nativeOrder())
    val resizedBitmap = bitmap.scale(256, 256, filter = true)
    val pixels = IntArray(256 * 256)
    resizedBitmap.getPixels(pixels, 0, 256, 0, 0, 256, 256)
    for (pixel in pixels) {
        val r = ((pixel shr 16) and 0xFF) / 255.0f
        val g = ((pixel shr 8) and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f
        inputBuffer.putFloat(r)
        inputBuffer.putFloat(g)
        inputBuffer.putFloat(b)
    }
    inputBuffer.rewind()
    return inputBuffer
}