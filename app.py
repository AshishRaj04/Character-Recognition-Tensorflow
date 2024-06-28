from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("model", "digit_recognition_model.h5")
model = tf.keras.models.load_model(model_path)

# Function to preprocess image data
def preprocess_image(image_data):
    # Convert to numpy array
    image_array = np.array(image_data)
    
    # Reshape to (1, 784) assuming image_data is already flattened
    processed_image = image_array.reshape((1, 784)).astype('float32')
    
    # Normalize pixel values to [0, 1]
    processed_image /= 255.0
    
    return processed_image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_data = data.get("image")  # Assuming data["image"] is a list/array of pixel values
        print(image_data)
        # Preprocess the image data
        processed_image = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction, axis=1)[0]

        return jsonify({"prediction": int(predicted_digit)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error message with 500 status code

if __name__ == '__main__':
    app.run(debug=True)
