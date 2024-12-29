
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
try:
    autoencoder_model_path = r"D:\API_Brast Cancer\Autoencoder_breast_cancer16.h5"
    autoencoder = load_model(autoencoder_model_path)
    logger.info("Autoencoder model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Autoencoder model: {e}")
    raise RuntimeError("Autoencoder model could not be loaded.")

# Load TFLite model
try:
    tflite_model_path = r"D:\API_Brast Cancer\SAVED_Breast_Cancer1.keras.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {e}")
    raise RuntimeError("TFLite model could not be loaded.")

# أنواع الملفات المسموح بها
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """التحقق مما إذا كان للملف امتداد صالح."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in (ext.lower() for ext in ALLOWED_EXTENSIONS)

def preprocess_image(image_path, target_size=(50, 50)):
    """معالجة الصورة للنموذج."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or invalid format.")
        if np.all(img == 0) or np.all(img >= 0.90) or np.mean(img) < 0.4:
            logger.info("This is not a valid breast cancer image.")
            return True
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise ValueError("Image preprocessing failed.")

def is_anomalous(image_path, threshold=0.03 * 0.4):
    """التحقق مما إذا كانت الصورة شاذة باستخدام Autoencoder."""
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
    """إجراء توقع باستخدام نموذج TFLite."""
    try:
        image = preprocess_image(image_path)
        if image is None:
            return "This is not a valid breast cancer image."
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        return "Cancer" if output > 0.5 else "Normal"
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "This is not a valid breast cancer image."

def predict_and_format_result(image_path):
    """دمج الكشف عن الشذوذ وتوقع TFLite."""
    if is_anomalous(image_path):
        return "This is not a valid breast cancer image."
    return predict_image(image_path)
