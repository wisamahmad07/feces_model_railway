import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Define class names
class_names = ['Coccidiosis', 'healthy', 'Newcastle', 'Salmonella']

# Load the model (Ensure model is in the same directory or provide full path)
model_path = "model.h5"  # Change if needed
print(f"Loading model from: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except OSError as e:
    print("❌ Error loading model:", e)
    exit(1)  # Exit if model loading fails

# Image preprocessing function
def preprocess_image(image, image_size=256):
    image = tf.image.resize(image, (image_size, image_size))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction function
def predict_model(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = class_names[np.argmax(predictions)]
    return predicted_label

# Flask Routes
@app.route("/")
def index():
    return "🐔 Welcome to the Poultry Pathology Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file received for prediction'})

    # Receive the image file
    image_file = request.files['file']
    print(f"Filename: {image_file.filename}")
    print(f"Content Type: {image_file.content_type}")

    try:
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process image', 'message': str(e)})

    # Predict the class
    predicted_class_name = predict_model(img)

    return jsonify({'predicted_class': predicted_class_name})

# Run Flask app locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
