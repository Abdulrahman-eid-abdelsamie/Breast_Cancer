import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    autoencoder_model_path = os.path.join(current_dir, "Autoencoder_breast_cancer16.h5")
    autoencoder = load_model(autoencoder_model_path)
    logging.info("Anomaly detection model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load anomaly detection model: {e}")
    raise RuntimeError(f"Anomaly detection model could not be loaded: {e}")

# تحميل نموذج TFLite
try:
    tflite_model_path = os.path.join(current_dir, "SAVED_Breast_Cancer1.keras.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load TFLite model: {e}")
    raise RuntimeError(f"TFLite model could not be loaded: {e}")

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        # فحص إذا كانت الصورة سوداء بالكامل أو بيضاء بالكامل أو داكنة جدًا
        if np.all(img == 0) or np.all(img >= 0.90) or np.mean(img) < 0.3:
            logging.info("This is not a valid breast cancer image.")
            raise ValueError("This is not a valid breast cancer image.")
        return True
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise ValueError("Image preprocessing failed.")

def is_anomalous(image_path, threshold=0.03*0.4):
    """التحقق مما إذا كانت الصورة شاذة باستخدام Autoencoder."""
    try:
        image = preprocess_image(image_path)
        reconstructed = autoencoder.predict(image)
        error = np.mean((image - reconstructed) ** 2)
        logger.info(f"Reconstruction error: {error}")
        return error > threshold
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        # raise ValueError("Anomaly detection failed.")
        return True



def predict_image(image_path):
    """إجراء توقع باستخدام نموذج TFLite."""
    try:
        image = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        return "Cancer" if output > 0.5 else "Normal"
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError("Prediction failed.")

def predict_and_format_result(image_path):
    """دمج الكشف عن الشذوذ وتوقع TFLite."""
    if is_anomalous(image_path):
        return "This is not a valid breast cancer image."
    return predict_image(image_path)
