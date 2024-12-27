# import logging
# import cv2
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model # type: ignore

# # إعداد تسجيل الأخطاء والمعلومات
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # تحميل نموذج Autoencoder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# try:
#     autoencoder_model_path = os.path.join(current_dir, "D:\Api\Autoencoder_breast_cancer16.h5")
#     autoencoder = load_model(autoencoder_model_path)
#     logging.info("Anomaly detection model loaded successfully.")
# except Exception as e:
#     logging.error(f"Failed to load anomaly detection model: {e}")
#     raise RuntimeError(f"Anomaly detection model could not be loaded: {e}")

# # تحميل نموذج TFLite
# try:
#     tflite_model_path = os.path.join(current_dir, "D:/Api/SAVED_Breast_Cancer1.keras.tflite")
#     interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     logging.info("TFLite model loaded successfully.")
# except Exception as e:
#     logging.error(f"Failed to load TFLite model: {e}")
#     raise RuntimeError(f"TFLite model could not be loaded: {e}")

# # دالة معالجة الصورة
# def preprocess_image(image_path, target_size=(50, 50)):
#     try:
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # قراءة الصورة بالألوان
#         if img is None:
#             logging.error(f"Error: Couldn't load the image at {image_path}")
#             return None
        
#         # تحويل الصورة من BGR إلى RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # إضافة هذا السطر لتحويل الصورة إلى RGB
        
#         img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
#         img_array = np.array(img_resized, dtype=np.float32)  # تحويل إلى float32
#         img_normalized = img_array / 255.0  # تطبيع القيم إلى [0, 1]
        
#         if img_normalized.shape[-1] != 3:
#             raise ValueError(f"Expected image with 3 channels (RGB), but got {img_normalized.shape[-1]} channels.")
        
#         return img_normalized
#     except Exception as e:
#         logging.error(f"Error in image preprocessing: {e}")
#         raise ValueError(f"Image preprocessing failed: {e}")

# # دالة للكشف عن الشذوذ باستخدام autoencoder
# def is_anomalous(image_path, threshold=0.03*0.4):
#     try:
#         processed_image = preprocess_image(image_path)
#         if processed_image is None:
#             return False
        
#         processed_image = np.expand_dims(processed_image, axis=0)  # إضافة بُعد الدُفعة
#         reconstructed = autoencoder.predict(processed_image)
#         reconstruction_error = np.mean((processed_image - reconstructed) ** 2)
#         logging.info(f"Reconstruction error: {reconstruction_error}")
#         return reconstruction_error > threshold
#     except Exception as e:
#         logging.error(f"Anomaly detection failed: {e}")
#         raise ValueError(f"Anomaly detection failed: {e}")

# # دالة التنبؤ باستخدام TFLite
# def predict_image(image_path, interpreter):
#     try:
#         preprocessed_img = preprocess_image(image_path)
#         if preprocessed_img is None:
#             return "Image preprocessing failed."
        
#         input_image = np.expand_dims(preprocessed_img, axis=0)  # الشكل: (1, 50, 50, 3)
#         if input_image.shape != (1, 50, 50, 3):
#             raise ValueError(f"Expected shape (1, 50, 50, 3), but got {input_image.shape}")
        
#         interpreter.set_tensor(input_details[0]['index'], input_image)
#         interpreter.invoke()

#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         prediction = output_data[0][0]  # استخراج الاحتمالية للفئة
#         predicted_class = "Cancer" if prediction > 0.5 else "Normal"
#         return f"Predicted Class: {predicted_class}, Probability: {prediction:.4f}"
#     except Exception as e:
#         logging.error(f"Prediction failed: {e}")
#         raise ValueError(f"Prediction failed: {e}")

# # دالة التنبؤ مع تنسيق النتيجة
# def predict_and_format_result(image_path):
#     try:
#         if is_anomalous(image_path):
#             logging.info("Image detected as anomalous.")
#             return "This is not a Breast Cancer image."
        
#         result = predict_image(image_path, interpreter)
#         return result
#     except Exception as e:
#         logging.error(f"Prediction failed: {e}")
#         return f"Prediction failed: {e}"

# # تنفيذ التنبؤ
# if __name__ == "__main__":
#     image_path = r"D:\API_Brast Cancer\wallpaperflare.com_wallpaper (12).jpg"  # ضع هنا مسار الصورة الفعلي
#     result = predict_and_format_result(image_path)
#     print(result)
    
    
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
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
        raise ValueError("Anomaly detection failed.")

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