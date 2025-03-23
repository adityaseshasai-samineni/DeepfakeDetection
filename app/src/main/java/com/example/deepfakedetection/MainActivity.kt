package com.example.deepfakedetection

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : Activity() {

    private val requestImageUpload = 1
    private val requestVideoUpload = 2

    private lateinit var resultLayout: LinearLayout  // Single container for both image and video frames
    private lateinit var tflite: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultLayout = findViewById(R.id.resultLayout)
        val btnUpload: Button = findViewById(R.id.btnUpload)
        val btnUploadVideo: Button = findViewById(R.id.btnUploadVideo)

        loadModel()

        btnUpload.setOnClickListener {
            uploadImage()
        }

        btnUploadVideo.setOnClickListener {
            uploadVideo()
        }
    }

    // Select Image from Device
    private fun uploadImage() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, requestImageUpload)
    }

    // Select Video from Device
    private fun uploadVideo() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "video/*"
        startActivityForResult(intent, requestVideoUpload)
    }

    // Handle the selected Image or Video
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                requestImageUpload -> {
                    try {
                        val imageUri = data?.data
                        val inputStream: InputStream? = imageUri?.let { contentResolver.openInputStream(it) }
                        val bitmap = BitmapFactory.decodeStream(inputStream)
                        inputStream?.close()

                        // Clear previous results
                        resultLayout.removeAllViews()

                        // Scale and process the uploaded image
                        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
                        val predictions = runInference(scaledBitmap)
                        displayFrameResult(scaledBitmap, predictions, "Uploaded Image")
                    } catch (e: Exception) {
                        e.printStackTrace()
                        Toast.makeText(this, "Failed to upload image", Toast.LENGTH_SHORT).show()
                    }
                }
                requestVideoUpload -> {
                    data?.data?.let { videoUri ->
                        // Clear previous results
                        resultLayout.removeAllViews()
                        processVideo(videoUri)
                    }
                }
            }
        }
    }

    // Load the TensorFlow Lite model
    private fun loadModel() {
        try {
            val modelFd = assets.openFd("model/efficientnet_lite_l4_model.tflite")
            val inputStream = modelFd.createInputStream()
            val modelBytes = inputStream.readBytes()
            val byteBuffer = ByteBuffer.allocateDirect(modelBytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(modelBytes)
            }
            val options = Interpreter.Options()
            tflite = Interpreter(byteBuffer, options)
            Toast.makeText(this, "Model Loaded Successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to Load Model", Toast.LENGTH_SHORT).show()
        }
    }

    // Run inference on the provided bitmap and return predictions
    private fun runInference(bitmap: Bitmap): FloatArray {
        val inputBuffer = convertBitmapToByteBuffer(bitmap)
        val output = Array(1) { FloatArray(5) }  // Assuming 5 classes
        tflite.run(inputBuffer, output)
        return output[0]
    }

    // Process Video: extract frames at 1-sec intervals and process each frame
    private fun processVideo(videoUri: Uri) {
        try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(this, videoUri)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLongOrNull() ?: 0L

            for (timeMs in 0 until durationMs step 1000) {
                val frameBitmap = retriever.getFrameAtTime(timeMs * 1000)  // convert ms to µs
                frameBitmap?.let { bitmap ->
                    val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
                    val predictions = runInference(scaledBitmap)
                    displayFrameResult(scaledBitmap, predictions, "Frame at ${timeMs / 1000} sec")
                }
            }
            retriever.release()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to process video", Toast.LENGTH_SHORT).show()
        }
    }

    // Dynamically add an ImageView and TextView to display the image/frame with its prediction results
    private fun displayFrameResult(frame: Bitmap, probabilities: FloatArray, label: String) {
        val labels = listOf("Real", "FE_Fake", "EFS_Fake", "FR_Fake", "FS_Fake")

        // Create and add ImageView
        val iv = ImageView(this).apply {
            setImageBitmap(frame)
            adjustViewBounds = true
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        // Create and add TextView for the prediction
        val tv = TextView(this).apply {
            val sb = StringBuilder()
            sb.append("$label:\n")
            for (i in probabilities.indices) {
                sb.append("${labels[i]}: ${probabilities[i]}\n")
            }
            text = sb.toString()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        resultLayout.addView(iv)
        resultLayout.addView(tv)
    }

    // Convert Bitmap to ByteBuffer for model input
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
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
        return inputBuffer
    }
}