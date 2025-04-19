package com.example.deepfakedetection

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.deepfakedetection.ui.theme.DeepfakeDetectionTheme
import org.tensorflow.lite.Interpreter
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import androidx.core.graphics.scale

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
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var results by remember { mutableStateOf(listOf<FrameResult>()) }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            isLoading = true
            errorMessage = null
            results = emptyList()
            try {
                val inputStream: InputStream? = context.contentResolver.openInputStream(it)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                val scaledBitmap = bitmap.scale(256, 256, filter = true)
                val predictions = tflite?.let { interpreter ->
                    runInference(interpreter, scaledBitmap)
                }
                results = listOf(
                    FrameResult(
                        label = "Uploaded Image",
                        bitmap = scaledBitmap,
                        probabilities = predictions?.toList()
                    )
                )
            } catch (e: Exception) {
                errorMessage = "Failed to upload image"
            }
            isLoading = false
        }
    }

    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            isLoading = true
            errorMessage = null
            results = emptyList()
            try {
                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(context, it)
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLongOrNull() ?: 0L
                val frameResults = mutableListOf<FrameResult>()
                for (timeMs in 0 until durationMs step 1000) {
                    val frameBitmap = retriever.getFrameAtTime(timeMs * 1000)
                    frameBitmap?.let { bitmap ->
                        val scaledBitmap = bitmap.scale(256, 256, filter = true)
                        val predictions = tflite?.let { interpreter ->
                            runInference(interpreter, scaledBitmap)
                        }
                        frameResults.add(
                            FrameResult(
                                label = "Frame at ${timeMs / 1000} sec",
                                bitmap = scaledBitmap,
                                probabilities = predictions?.toList()
                            )
                        )
                    }
                }
                retriever.release()
                results = frameResults
            } catch (e: Exception) {
                errorMessage = "Failed to process video"
            }
            isLoading = false
        }
    }

    LaunchedEffect(Unit) {
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

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(horizontal = 24.dp, vertical = 16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Top half: Title, subtitle, and upload buttons centered vertically
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
                        enabled = isModelLoaded && !isLoading,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Upload Image")
                    }
                    Button(
                        onClick = { videoPickerLauncher.launch("video/*") },
                        enabled = isModelLoaded && !isLoading,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Upload Video")
                    }
                }
            }
        }

        // Bottom half: Progress, errors, and results
        AnimatedVisibility(
            visible = isLoading,
            enter = fadeIn(),
            exit = fadeOut()
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator()
                Text(
                    text = "Processing...",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp)
                )
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

        Spacer(Modifier.height(16.dp))

        results.forEach { frameResult ->
            FrameResultCard(frameResult)
            Spacer(Modifier.height(16.dp))
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
            result.probabilities?.let { probs ->
                Column {
                    labels.forEachIndexed { i, label ->
                        val percent = (probs.getOrNull(i) ?: 0f) * 100
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
                                progress = (probs.getOrNull(i) ?: 0f).coerceIn(0f, 1f),
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