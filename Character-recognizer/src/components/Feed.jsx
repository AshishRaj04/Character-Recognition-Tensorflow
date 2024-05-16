import React, { useState, useEffect } from "react";
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const Feed = () => {
  const [data, setData] = useState({
    image: null,
  });
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    // Load the TensorFlow model from the public directory
    const loadModel = async () => {
      try {
        const model = await loadGraphModel('../first_model.keras');
        setModel(model);
      } catch (error) {
        console.error("Error loading the model", error);
      }
    };
    loadModel();
  }, []);

  const handleChange = (event) => {
    const { name, files } = event.target;
    setData({ [name]: files[0] });
  };

  const preprocessImage = async (imageFile) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(imageFile);

    return new Promise((resolve) => {
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, 28, 28);
        const imageData = ctx.getImageData(0, 0, 28, 28);
        
        // Convert to grayscale and flatten
        const grayscaleData = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
          const avg = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
          grayscaleData.push(avg / 255); // Normalize to [0, 1]
        }

        // Convert to a tensor and reshape to match model input
        const tensor = tf.tensor2d(grayscaleData, [1, 784]);
        resolve(tensor);
      };
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!data.image) {
      alert("Please select an image");
      return;
    }
    if (!model) {
      alert("Model is not loaded yet");
      return;
    }

    const preprocessedImage = await preprocessImage(data.image);
    const logits = model.predict(preprocessedImage);
    const prediction = tf.nn.softmax(logits).dataSync();
    const predictedClass = prediction.indexOf(Math.max(...prediction));
    setPrediction(predictedClass);
  };

  return (
    <>
      <div>
        <div>
          <h2>Upload the image of the character you want to recognize.</h2>
        </div>
        <form onSubmit={handleSubmit}>
          <div>
            <label>Upload Image: </label>
            <input
              type="file"
              name="image"
              accept=".png,.jpg,.jpeg"
              onChange={handleChange}
            />
          </div>
          <div>
            <button type="submit">Upload</button>
          </div>
        </form>

        <div>
          <h2>Recognized character as per the AI model:</h2>
          {prediction !== null && (
            <div>
              <p>Predicted Character: {String.fromCharCode(prediction)}</p>
              <img src={URL.createObjectURL(data.image)} alt="Uploaded Character" />
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default Feed;
