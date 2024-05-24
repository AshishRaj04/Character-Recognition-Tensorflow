from flask import Flask , request , jsonify , render_template
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model_path = os.path.join("model" , "digit_recognition_model.h5")
model = tf.keras.models.load_model(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict" , methods = ["POST"])
def predict():
    data = request.get_json(force=True)
    image = np.array(np.array(data["image"]).reshape((1,784)))
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction , axis = 1)[0].encode('utf-8')
    return jsonify({"prediction":int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)