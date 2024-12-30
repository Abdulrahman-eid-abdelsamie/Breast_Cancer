
import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Autoencoder model
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    autoencoder_model_path = os.path.join(current_dir, "Autoencoder_breast_cancer16.h5")
    autoencoder = load_model(autoencoder_model_path)
    logging.info("Anomaly detection model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load anomaly detection model: {e}")
    raise RuntimeError(f"Anomaly detection model could not be loaded: {e}")

try:
    tflite_model_path = os.path.join(current_dir, "SAVED_Breast_Cancer1.keras.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load TFLite model: {e}")
    raise RuntimeError(f"TFLite model could not be loaded: {e}")

# Types of allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in (ext.lower() for ext in ALLOWED_EXTENSIONS)

def preprocess_image(image_path, target_size=(50, 50)):
    """Preprocess the image for the model."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or invalid format.")
        
        if np.all(img == 0) or np.all(img == 1) or np.mean(img) < 0.3:
            logger.info("This is not a valid breast image.")
            return True
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise ValueError("Image preprocessing failed.")

def is_anomalous(image_path, threshold= 0.03 * 0.02):
    """Check if the image is anomalous using Autoencoder."""
    try:
        image = preprocess_image(image_path)
        if image is None:
            return True
        reconstructed = autoencoder.predict(image)
        error = np.mean((image - reconstructed) ** 2)
        logger.info(f"Reconstruction error: {error}")
        return error > threshold
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return True

def predict_image(image_path):
    """Make a prediction using the TFLite model."""
    try:
        image = preprocess_image(image_path)
        if image is None:
            return "This is not a valid breast image."
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        return "Cancer" if output > 0.5 else "Normal"
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "This is not a valid breast image."

def predict_and_format_result(image_path):
    """Combine anomaly detection and TFLite prediction."""
    if is_anomalous(image_path):
        return "This is not a valid breast image."
    return predict_image(image_path)

