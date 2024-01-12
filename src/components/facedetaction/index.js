import React, { useRef, useState } from 'react'
import './index.css'
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import '@mediapipe/face_detection';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import * as faceDetection from '@tensorflow-models/face-detection';

export default function FaceDetection(props) {
    const [imgData, setImgData] = useState(null)
    const [predictions, setPredictions] = useState([]);
    const [facepredictions, setFacePredictions] = useState([]);
    const [isLoading, setLoading] = useState(false);
    const faceDetectionImg = useRef();
    const imageRef = useRef();
    // const blazeface = require('@tensorflow-models/blazeface')

    const hanldeOpenSelectImg = () => {
        if (faceDetectionImg?.current) {
            faceDetectionImg?.current?.click()
        }
    }

    const isEmptyPredictions = !predictions || predictions.length === 0;
    const normalizePredictions = (predictions, imgSize) => {
        if (!predictions || !imgSize || !imageRef) return predictions || [];
        return predictions.map((prediction) => {
            const { bbox } = prediction;
            const oldX = bbox[0];
            const oldY = bbox[1];
            const oldWidth = bbox[2];
            const oldHeight = bbox[3];

            const imgWidth = imageRef.current.width;
            const imgHeight = imageRef.current.height;
            const x = (oldX * imgWidth) / imgSize.width;
            const y = (oldY * imgHeight) / imgSize.height;
            const width = (oldWidth * imgWidth) / imgSize.width;
            const height = (oldHeight * imgHeight) / imgSize.height;

            return { ...prediction, bbox: [x, y, width, height] };
        });
    };

    const facePredictions = (predictions, imgSize) => {
        if (!predictions || !imgSize || !imageRef) return predictions || [];
        return predictions.map((prediction) => {
            const { box } = prediction;
            const oldXMax = box?.xMax;
            const oldXMin = box?.xMin;
            const oldYMax = box?.yMax;
            const oldYMin = box?.yMin;
            const oldWidth = box?.width;
            const oldHeight = box?.height;

            const imgWidth = imageRef.current.width;
            const imgHeight = imageRef.current.height;
            const xMax = (oldXMax * imgWidth) / imgSize.width;
            const xMin = (oldXMin * imgWidth) / imgSize.width;
            const yMax = (oldYMax * imgHeight) / imgSize.height;
            const yMin = (oldYMin * imgHeight) / imgSize.height;
            const width = (oldWidth * imgWidth) / imgSize.width;
            const height = (oldHeight * imgHeight) / imgSize.height;

            return { ...prediction, box: [xMax, , xMin, yMax, yMin, width, height] };
        });
    };

    const detectObjectsOnImage = async (imageElement, imgSize) => {
        //Object Detection
        const model = await cocoSsd.load({});
        const predictions = await model.detect(imageElement, 6);
        const normalizedPredictions = normalizePredictions(predictions, imgSize);
        setPredictions(normalizedPredictions);

        //Face Detection
        const models = faceDetection.SupportedModels.MediaPipeFaceDetector;
        const detectorConfig = {
            runtime: 'mediapipe',
            solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection'
        };
        const detector = await faceDetection.createDetector(models, detectorConfig);
        const estimationConfig = { flipHorizontal: false };
        const faces = await detector.estimateFaces(imageElement, estimationConfig);
        const facesPredictions = facePredictions(faces, imgSize);
        setFacePredictions(facesPredictions)
    };

    const readImage = (file) => {
        return new Promise((rs, rj) => {
            const fileReader = new FileReader();
            fileReader.onload = () => rs(fileReader.result);
            fileReader.onerror = () => rj(fileReader.error);
            fileReader.readAsDataURL(file);
        });
    };

    const handleChangeImg = async (e) => {
        setPredictions([]);
        setLoading(true);

        const file = e.target.files[0]
        const imgData = await readImage(file)
        setImgData(imgData)

        const imageElement = document.createElement("img");
        imageElement.src = imgData;

        imageElement.onload = async () => {
            const imgSize = {
                width: imageElement.width,
                height: imageElement.height,
            };
            await detectObjectsOnImage(imageElement, imgSize);
            setLoading(false);
        }
    }

    return (
        <div className='objectDetectorContainer'>
            <div className='detectorContainer'>
                <img src={imgData} ref={imageRef} accept=".jpeg, .png, .jpg" />
                {!isEmptyPredictions &&
                    predictions.map((prediction, idx) => (
                        <div
                            className='targetBox'
                            style={{
                                left: `${prediction.bbox[0]}px`,
                                top: `${prediction.bbox[1]}px`,
                                width: `${prediction.bbox[2]}px`,
                                height: `${prediction.bbox[3]}px`
                            }}
                            key={idx}
                        />
                    ))}
                {!isEmptyPredictions &&
                    facepredictions.map((prediction, idx) => (
                        <div
                            className='targetBox'
                            style={{
                                left: `${prediction?.box?.[2]}px`,
                                top: `${prediction?.box?.[4]}px`,
                                width: `${prediction?.box?.[5]}px`,
                                height: `${prediction?.box?.[6]}px`
                            }}
                            key={idx}
                        />
                    ))}
            </div>
            <input type='file' ref={faceDetectionImg} onChange={handleChangeImg} />
            <button onClick={hanldeOpenSelectImg}>{isLoading ? "Recognizing..." : "Select Image"}</button>
        </div>
    )
}
