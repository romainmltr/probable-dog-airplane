import React, { useRef, useState, useEffect } from "react";
import * as ort from "onnxruntime-web";

export const Webcam = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [prediction, setPrediction] = useState("");
    const [capturedImage, setCapturedImage] = useState(null);

    const classNames = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ];

    useEffect(() => {
        const startWebcam = async () => {
            const constraints = { video: true };
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                const video = videoRef.current;
                if (video) {
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        video.play().catch(e => console.error("Error playing video:", e));
                    };
                }
            } catch (error) {
                console.error("Error accessing webcam:", error);
            }
        };

        startWebcam();
    }, []);

    const loadModelAndPredict = async (tensor) => {
        try {
            const session = await ort.InferenceSession.create('/modelCIFAR10.onnx');
            const feeds = { input: tensor };
            const results = await session.run(feeds);

            const outputTensor = results.output;
            const outputData = outputTensor.data;

            const predictedClassIndex = outputData.reduce(
                (maxIndex, current, index, arr) =>
                    current > arr[maxIndex] ? index : maxIndex,
                0
            );

            setPrediction(`Predicted: ${classNames[predictedClassIndex]}`);
        } catch (error) {
            console.error("Error running ONNX model:", error);
            setPrediction("Prediction failed");
        }
    };

    const captureAndPredict = () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;

        canvas.width = 32;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 32, 32);

        // Capture image for display
        const capturedImageUrl = canvas.toDataURL();
        setCapturedImage(capturedImageUrl);

        // Prepare tensor for prediction
        const imageData = ctx.getImageData(0, 0, 32, 32);
        const data = new Float32Array(3 * 32 * 32);

        for (let i = 0; i < 32 * 32; i++) {
            const r = imageData.data[i * 4] / 255.0;
            const g = imageData.data[i * 4 + 1] / 255.0;
            const b = imageData.data[i * 4 + 2] / 255.0;

            data[i] = r * 2 - 1;
            data[32 * 32 + i] = g * 2 - 1;
            data[2 * 32 * 32 + i] = b * 2 - 1;
        }

        const tensor = new ort.Tensor('float32', data, [1, 3, 32, 32]);
        loadModelAndPredict(tensor);
    };

    return (
        <div className="webcam-container">
            <button onClick={captureAndPredict}>Capture & Predict</button>

            <div className="video-preview">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>

            {capturedImage && (
                <div className="captured-image">
                    <img src={capturedImage} alt="Captured" />
                </div>
            )}

            {prediction && <h3>{prediction}</h3>}
        </div>
    );
};