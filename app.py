from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model
model_path = r"P:\Wheat Disease Classification\wheat_model.h5"
model = tf.keras.models.load_model(model_path)

# Get model input size automatically
_, height, width, channels = model.input_shape
IMAGE_SIZE = (height, width)

# Label mapping
labels = {0: "Brown rust", 1: "Healthy", 2: "Loose Smut", 3: "Yellow rust"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save temporarily
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Preprocess using model's own input size
        img = image.load_img(filepath, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return jsonify(
            {"class": labels[predicted_class], "confidence": round(confidence * 100, 2)}
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
