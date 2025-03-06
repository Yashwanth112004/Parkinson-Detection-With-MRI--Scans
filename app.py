from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Image preprocessing function
def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.
    :param image_path: Path to the MRI scan image.
    :return: Preprocessed and scaled image array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        raise ValueError("Invalid image path")
    img = cv2.resize(img, (128, 128))  # Resize to the required dimensions
    img = img.flatten().reshape(1, -1)  # Flatten and reshape
    img = scaler.transform(img)  # Scale using the loaded scaler
    return img

# HTML form for image upload
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class of an uploaded image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        preprocessed_image = preprocess_image(file_path)
        prediction = model.predict(preprocessed_image)[0]
        result = 'Parkinson Detected' if prediction == 1 else 'Normal MRI'
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
