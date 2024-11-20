import React, { useState } from "react";
import * as ort from "onnxruntime-web";

export const UploadImage = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [prediction, setPrediction] = useState("");

    const classNames = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ];

    const loadModelAndPredict = async (tensor) => {
        try {
            const session = await ort.InferenceSession.create('/modelCIFAR10.onnx');

            const feeds = { input: tensor };
            const results = await session.run(feeds);

            // Use first output tensor, likely named 'output'
            const outputTensor = results.output;
            const outputData = outputTensor.data;

            // Find index of maximum value
            const predictedClassIndex = outputData.reduce(
                (maxIndex, current, index, arr) =>
                    current > arr[maxIndex] ? index : maxIndex,
                0
            );

            setPrediction(`Predicted Class: ${classNames[predictedClassIndex]}`);
        } catch (error) {
            console.error("Error running ONNX model:", error);
            setPrediction("Prediction failed");
        }
    };

    const preprocessImage = async (file) => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = () => {
                const canvas = document.createElement("canvas");
                canvas.width = 32;
                canvas.height = 32;

                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, 32, 32);

                const imageData = ctx.getImageData(0, 0, 32, 32);
                const data = new Float32Array(3 * 32 * 32);

                // Separate channels, maintain NCHW format
                for (let i = 0; i < 32 * 32; i++) {
                    const r = imageData.data[i * 4] / 255.0;
                    const g = imageData.data[i * 4 + 1] / 255.0;
                    const b = imageData.data[i * 4 + 2] / 255.0;

                    // Explicit normalization matching original transform
                    data[i] = r * 2 - 1;
                    data[32 * 32 + i] = g * 2 - 1;
                    data[2 * 32 * 32 + i] = b * 2 - 1;
                }

                const tensor = new ort.Tensor('float32', data, [1, 3, 32, 32]);
                loadModelAndPredict(tensor);
            };
            img.onerror = reject;
        });
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedImage(URL.createObjectURL(file));
            preprocessImage(file);
        }
    };

    return (
        <div className="upload-container">
            <h2>Upload an Image</h2>
            <h3>Choose an image of a CIFAR-10 class:</h3>
            <p> "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"</p>
            <input type="file" accept="image/*" onChange={handleFileChange}/>
            <div className="divider"/>
            {selectedImage && (
                <div className="image-preview">
                    <img src={selectedImage} alt="Selected" width={200}/>
                </div>
            )}
            {prediction && <h3>{prediction}</h3>}
        </div>
    );
};