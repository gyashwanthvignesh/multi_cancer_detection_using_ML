import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gdown

# ==========================
# PATHS
# ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# GOOGLE DRIVE LINKS (FULL LINKS)
# ==========================
MODEL_LINKS = {
    "cancer_type_model.keras": "https://drive.google.com/file/d/1S_PjeZxAZacng7K1Vj_RCcaUUdHBJBSV/view?usp=drive_link",
    "breast_cancer_model.keras": "https://drive.google.com/file/d/184UmuXEBbgtvGULP5kdADnEjeFVY3qUR/view?usp=drive_link",
    "lung_cancer_model.keras": "https://drive.google.com/file/d/1RIstW0kHfUAOIHradn3bGP816Obpoc6s/view?usp=drive_link",
    "skin_cancer_model.keras": "https://drive.google.com/file/d/1cvBGrZH_N1qXo1Y5-3QVPj_mgj19qKDL/view?usp=drive_link"
}

# ==========================
# EXTRACT FILE ID FROM LINK
# ==========================
def extract_file_id(drive_link):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', drive_link)
    return match.group(1) if match else None

# ==========================
# DOWNLOAD MODELS
# ==========================
def download_models():
    for filename, drive_link in MODEL_LINKS.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            print(f"⬇️ Downloading {filename}...")

            file_id = extract_file_id(drive_link)
            if not file_id:
                print(f"❌ Invalid Google Drive link: {drive_link}")
                continue

            url = f"https://drive.google.com/uc?id={file_id}"

            try:
                gdown.download(url, path, quiet=False)
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

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

    print("✅ Models loaded successfully")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_cancer(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict cancer type
    cancer_type_pred = cancer_type_model.predict(img_array)
    cancer_index = np.argmax(cancer_type_pred)
    detected_cancer = cancer_types[cancer_index]

    print(f"🔍 Cancer Type: {detected_cancer}")

    # Predict benign/malignant
    malignancy_pred = models_dict[detected_cancer].predict(img_array)
    result = "Malignant" if malignancy_pred[0][0] > 0.5 else "Benign"

    print(f"🔍 Result: {result}")

# ==========================
# TEST IMAGE (JENKINS SAFE)
# ==========================
test_image = os.path.join(SCRIPT_DIR, "../static/uploads/95.jpg")

if os.path.exists(test_image):
    print("🧪 Running test prediction...")
    predict_cancer(test_image)
    print("✅ Test completed successfully")
else:
    print("⚠️ No test image found, skipping test")
