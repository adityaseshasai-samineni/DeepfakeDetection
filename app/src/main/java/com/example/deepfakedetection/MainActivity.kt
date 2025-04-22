package com.example.deepfakedetection

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.media.MediaMetadataRetriever.OPTION_CLOSEST
import android.net.Uri
import android.os.Bundle
import android.util.Log
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
                val modelFiles = assetManager.list("model")?.toList() ?: emptyList()
                val tfliteFile = modelFiles.find { it.endsWith(".tflite") }
                val h5File = modelFiles.find { it.endsWith(".h5") }
                val modelName = tfliteFile ?: h5File ?: throw Exception("No model file found")
                val assetPath = "model/$modelName"
                if (modelName.endsWith(".tflite")) {
                    // load tflite model
                    val modelFd = assetManager.openFd(assetPath)
                    val inputStream = modelFd.createInputStream()
                    val modelBytes = inputStream.readBytes()
                    val byteBuffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                        order(ByteOrder.nativeOrder())
                        put(modelBytes)
                    }
                    val options = Interpreter.Options()
                    tflite = Interpreter(byteBuffer, options)
                    isModelLoaded = true
                } else if (modelName.endsWith(".h5")) {
                    // TODO: implement .h5 loading (requires TensorFlow Java APIs or conversion)
                    errorMessage = "H5 model support not implemented"
                }
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
            Log.d("MainActivity", "Processing media URI: $uri, type: $selectedType")
            try {
                withContext(Dispatchers.Default) {
                    if (selectedType == "image") {
                        Log.d("MainActivity", "Opening image input stream for URI: $uri")
                        statusText = "Loading image..."
                        progress = 0.1f
                        val bitmap = withContext(Dispatchers.IO) {
                            context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                        } ?: throw Exception("Failed to load image")
                        Log.d("MainActivity", "Decoded image size: ${bitmap.width}x${bitmap.height}")
                        statusText = "Processing image..."
                        progress = 0.4f
                        val scaledBitmap = withContext(Dispatchers.Default) { bitmap.scale(256, 256, filter = true) }
                        Log.d("MainActivity", "Scaled image to 256x256")
                        val predictions = tflite?.let { interpreter ->
                            withContext(Dispatchers.Default) { runInference(interpreter, scaledBitmap) }
                        }
                        Log.d("MainActivity", "Image inference output: ${predictions?.joinToString()}")
                        progress = 1f
                        results = listOf(
                            FrameResult(
                                label = "Uploaded Image",
                                bitmap = scaledBitmap,
                                probabilities = predictions?.toList()
                            )
                        )
                    } else {
                        Log.d("MainActivity", "Setting video data source for URI: $uri")
                        // Open video URI via AssetFileDescriptor to set data source reliably
                        val (retriever, afd) = withContext(Dispatchers.IO) {
                            val afd = context.contentResolver.openAssetFileDescriptor(uri, "r")
                                ?: throw Exception("Unable to open video URI: $uri")
                            val retriever = MediaMetadataRetriever().apply {
                                setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                            }
                            Pair(retriever, afd)
                        }
                        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                        Log.d("MainActivity", "Video duration: $durationMs ms")
                        val totalSeconds = (durationMs / 1000).toInt().coerceAtLeast(1)
                        val frameIndices = if (totalSeconds > 50) (0 until totalSeconds).shuffled().take(50) else (0 until totalSeconds).toList()
                        Log.d("MainActivity", "Frame indices to process: $frameIndices")
                        val frames = mutableListOf<FrameResult>()
                        for ((idx, sec) in frameIndices.withIndex()) {
                            Log.d("MainActivity", "Fetching frame at second: $sec")
                            statusText = "Processing frame ${idx + 1}/${frameIndices.size}"
                            progress = (idx + 1) / frameIndices.size.toFloat()
                            // Extract closest frame to avoid null bitmaps
                            val frameBitmap = withContext(Dispatchers.IO) { retriever.getFrameAtTime(sec * 1_000_000L, OPTION_CLOSEST) }
                            if (frameBitmap == null) {
                                Log.w("MainActivity", "Null frame at second: $sec, skipping")
                                continue
                            }
                            val scaled = withContext(Dispatchers.Default) { frameBitmap.scale(256, 256, filter = true) }
                            Log.d("MainActivity", "Processed frame ${idx + 1}, scaled to 256x256")
                            val framePredictions = tflite?.let { interpreter ->
                                withContext(Dispatchers.Default) { runInference(interpreter, scaled) }
                            }
                            Log.d("MainActivity", "Frame inference output for frame ${idx + 1}: ${framePredictions?.joinToString()}")
                            frames.add(FrameResult("Frame at ${sec + 1} sec", scaled, framePredictions?.toList()))
                        }
                        Log.d("MainActivity", "Total frames processed: ${frames.size}")
                        // Release resources
                        withContext(Dispatchers.IO) {
                            retriever.release()
                            afd.close()
                        }
                        progress = 1f
                        results = frames
                    }
                }
                screen = Screen.Results
            } catch (e: CancellationException) {
                // processing was cancelled, ignore
            } catch (e: Exception) {
                Log.e("MainActivity", "Error processing media", e)
                errorMessage = "Error processing media: ${e.localizedMessage ?: e.toString()}"
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
    // Dynamically retrieve number of output classes from the model
    val outputTensor = tflite.getOutputTensor(0)
    val numClasses = outputTensor.shape()[1]
    val output = Array(1) { FloatArray(numClasses) }
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