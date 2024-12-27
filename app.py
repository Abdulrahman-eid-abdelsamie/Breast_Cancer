from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from utils import predict_and_format_result, allowed_file
import cv2

# Configure Flask app
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  #  upload limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_message = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            error_message = "Invalid file type. Please upload a valid image."
            return render_template('index.html', result=None, error=error_message)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"File saved at: {filepath}")  # التأكد من حفظ الملف بنجاح

        try:
            result = predict_and_format_result(filepath)
        except Exception as e:
            error_message = f"Prediction failed: {e}"

    return render_template('index.html', result=result, error=error_message)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle API predictions."""
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        logger.error("Invalid file type.")
        return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict_and_format_result(filepath)
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=500)