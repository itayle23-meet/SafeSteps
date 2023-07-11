package com.example.newas;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;

public class GenderClassificationHelper {
    private Interpreter interpreter;
    private ImageProcessor imageProcessor;

    public GenderClassificationHelper(Context context, String modelPath) {
        try {
            interpreter = new Interpreter(loadModelFile(context, modelPath));
            imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(128, 128, ResizeOp.ResizeMethod.BILINEAR))
                    .build();
        } catch (IOException e) {
            Log.e("GenderClassification", "Error initializing TensorFlow Lite interpreter", e);
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        return FileUtil.loadMappedFile(context, modelPath);
    }

    public boolean isModelLoaded() {
        return interpreter != null;
    }

    public boolean predictGender(Bitmap image) {
        try {
            if (interpreter != null) {
                TensorImage tensorImage = new TensorImage(DataType.UINT8);
                tensorImage.load(image);
                tensorImage = imageProcessor.process(tensorImage);

                float[][] genderOutputArray = new float[1][2];
                ByteBuffer genderOutputBuffer = ByteBuffer.allocateDirect(4 * 2);
                genderOutputBuffer.order(java.nio.ByteOrder.nativeOrder());
                interpreter.run(tensorImage.getBuffer(), genderOutputBuffer);
                genderOutputBuffer.rewind();
                genderOutputBuffer.asFloatBuffer().get(genderOutputArray[0]);

                return genderOutputArray[0][0] > genderOutputArray[0][1];
            }
        } catch (Exception e) {
            Log.e("GenderClassification", "Error performing gender classification", e);
        }
        return false;
    }
}
