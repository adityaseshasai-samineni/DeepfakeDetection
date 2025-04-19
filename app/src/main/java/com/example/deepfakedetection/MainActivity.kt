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
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.graphics.scale
import com.example.deepfakedetection.screens.ProgressScreen
import com.example.deepfakedetection.screens.ResultsScreen
import com.example.deepfakedetection.screens.UploadScreen
import com.example.deepfakedetection.ui.theme.DeepfakeDetectionTheme
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

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
    var processingJob by remember { mutableStateOf<Job?>(null) }

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
            processingJob = currentCoroutineContext()[Job]!!
            try {
                withContext(Dispatchers.Default) {
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
                }
                screen = Screen.Results
            } catch (e: CancellationException) {
                // processing was cancelled, ignore
            } catch (e: Exception) {
                errorMessage = "Failed to process media"
                screen = Screen.Upload
            }
        }
    }

    when (screen) {
        is Screen.Upload -> UploadScreen(
            isModelLoaded = isModelLoaded,
            errorMessage = errorMessage,
            onImageSelected = { imagePickerLauncher.launch("image/*") },
            onVideoSelected = { videoPickerLauncher.launch("video/*") }
        )
        is Screen.Progress -> ProgressScreen(
            statusText = statusText,
            progress = progress,
            onCancel = {
                processingJob?.cancel()
                selectedUri = null
                screen = Screen.Upload
            }
        )
        is Screen.Results -> ResultsScreen(
            results = results,
            onAnalyzeAnother = { screen = Screen.Upload }
        )
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