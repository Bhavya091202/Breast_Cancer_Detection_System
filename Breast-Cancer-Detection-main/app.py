from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)

# Load the pre-trained model
model = load_model('./breast_cancer_model.keras')  # Replace 'model.h5' with your actual model file path

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image for prediction
    img = img.resize((224, 224))  # Resize the image to match your model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Predict the result
    prediction = model.predict(img_array)
    result = 'Cancerous' if prediction[0] > 0.5 else 'Non Cancerous' # Assuming binary classification

    # Return the prediction
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
