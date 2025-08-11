#!/usr/bin/env python3
"""
Enhanced Flask API for Wheat Disease Classification
Clean version without embedded HTML
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG = {
    'model_path': 'best_enhanced_wheat_model.h5',
    'backup_model_path': 'best_enhanced_wheat_model.h5',
    'upload_folder': 'uploads',
    'max_file_size': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'confidence_threshold': 0.5
}

# Global variables
model = None
IMAGE_SIZE = (224, 224)  # Match classifier training size
labels = {0: "Brown rust", 1: "Healthy", 2: "Loose Smut", 3: "Yellow rust"}

def load_model():
    """Load the enhanced model with fallback options"""
    global model, IMAGE_SIZE
    
    try:
        if os.path.exists(CONFIG['model_path']):
            model = tf.keras.models.load_model(CONFIG['model_path'])
            logger.info(f"Loaded model: {CONFIG['model_path']}")
        elif os.path.exists(CONFIG['backup_model_path']):
            model = tf.keras.models.load_model(CONFIG['backup_model_path'])
            logger.info(f"Loaded backup model: {CONFIG['backup_model_path']}")
        else:
            raise FileNotFoundError("No model file found")
        
        # Get actual input size from model
        _, height, width, channels = model.input_shape
        IMAGE_SIZE = (height, width)
        logger.info(f"Model input size: {IMAGE_SIZE}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG['allowed_extensions']

def preprocess_image(file_path):
    """Enhanced image preprocessing"""
    img = image.load_img(file_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def get_prediction_details(predictions):
    """Get detailed prediction information"""
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    
    # Get all class probabilities
    class_probabilities = {}
    for i, prob in enumerate(predictions[0]):
        class_probabilities[labels[i]] = round(float(prob) * 100, 2)
    
    return {
        'predicted_class': labels[predicted_class],
        'confidence': round(confidence * 100, 2),
        'class_probabilities': class_probabilities,
        'is_confident': confidence >= CONFIG['confidence_threshold']
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Use: {", ".join(CONFIG["allowed_extensions"])}'}), 400
        
        # Create upload directory and save file
        os.makedirs(CONFIG['upload_folder'], exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(CONFIG['upload_folder'], filename)
        file.save(filepath)
        
        # Process and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)
        result = get_prediction_details(predictions)
        
        # Log and cleanup
        logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']}%) for {filename}")
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'image_size': IMAGE_SIZE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'classes': labels,
        'confidence_threshold': CONFIG['confidence_threshold']
    })

@app.route('/')
def home():
    """Serve the main HTML page"""
    return app.send_static_file('enhanced_index.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    try:
        load_model()
        app.config['MAX_CONTENT_LENGTH'] = CONFIG['max_file_size']
        logger.info("Starting Enhanced Wheat Disease Classification API")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)