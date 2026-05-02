from flask import Flask, request, jsonify, render_template
import os
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS
from google import genai
import gdown

# ==========================
# LOAD ENV VARIABLES
# ==========================
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ==========================
# GEMINI API KEY
# ==========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY in .env")

# ==========================
# UPLOAD CONFIG
# ==========================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================
# PATHS
# ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# GOOGLE DRIVE MODEL LINKS
# 👉 REPLACE FILE_IDs
# ==========================
MODEL_LINKS = {
    "cancer_type_model.keras": "https://drive.google.com/file/d/1S_PjeZxAZacng7K1Vj_RCcaUUdHBJBSV/view?usp=drive_link",
    "breast_cancer_model.keras": "https://drive.google.com/file/d/184UmuXEBbgtvGULP5kdADnEjeFVY3qUR/view?usp=drive_link",
    "lung_cancer_model.keras": "https://drive.google.com/file/d/1RIstW0kHfUAOIHradn3bGP816Obpoc6s/view?usp=drive_link",
    "skin_cancer_model.keras": "https://drive.google.com/file/d/1RIstW0kHfUAOIHradn3bGP816Obpoc6s/view?usp=drive_link",
}

# ==========================
# DOWNLOAD MODELS
# ==========================
def download_models():
    for filename, file_id in MODEL_LINKS.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, path, quiet=False)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

# Download before loading
download_models()

# ==========================
# LOAD MODELS
# ==========================
try:
    cancer_type_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "cancer_type_model.keras")
    )

    models_dict = {
        "breast_cancer": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "breast_cancer_model.keras")
        ),
        "lung_cancer": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "lung_cancer_model.keras")
        ),
        "skin_cancer": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "skin_cancer_model.keras")
        ),
    }

    cancer_types = ["breast_cancer", "lung_cancer", "skin_cancer"]
    models_loaded = True
    print("✅ Models loaded successfully")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    models_loaded = False

# ==========================
# HELPER FUNCTIONS
# ==========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_cancer(image_path):
    if not models_loaded:
        return {"error": "Models not loaded"}

    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict cancer type
        cancer_type_pred = cancer_type_model.predict(img_array)
        cancer_index = np.argmax(cancer_type_pred)
        detected_cancer = cancer_types[cancer_index]
        cancer_confidence = float(cancer_type_pred[0][cancer_index])

        # Predict benign/malignant
        malignancy_pred = models_dict[detected_cancer].predict(img_array)
        malignancy_result = "Malignant" if malignancy_pred[0][0] > 0.5 else "Benign"
        malignancy_confidence = float(malignancy_pred[0][0])

        return {
            "cancer_type": detected_cancer,
            "cancer_confidence": cancer_confidence,
            "malignancy": malignancy_result,
            "malignancy_confidence": malignancy_confidence
        }

    except Exception as e:
        return {"error": str(e)}

# ==========================
# ROUTES
# ==========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/lifestyle')
def lifestyle():
    return render_template('lifestyle.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty file name"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_cancer(filepath)
        return jsonify(result)

    return jsonify({"error": "Invalid file type"})

# ==========================
# CHAT ROUTE (GEMINI)
# ==========================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"response": "Hey 👋 Ask me anything about cancer."})

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        system_prompt = """
You are a friendly Cancer Support Assistant.

- Talk like a supportive friend
- Answer only cancer-related topics
- Use bullet points
- Keep it simple

If unrelated:
"I’m here to help only with cancer-related questions 😊"
"""

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=system_prompt + "\nUser: " + user_message
        )

        ai_response = response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        traceback.print_exc()
        ai_response = "Sorry 😔 Something went wrong."

    return jsonify({"response": ai_response})

# ==========================
# RUN APP
# ==========================
if __name__ == '__main__':
    app.run(debug=True)