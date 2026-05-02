import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define paths dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "../models")

# Check if model exists
cancer_type_model_path = os.path.join(MODEL_DIR, "cancer_type_model.keras")
if not os.path.exists(cancer_type_model_path):
    raise FileNotFoundError(f" Model file not found: {cancer_type_model_path}\nTrain the model before running check.py.")

# Load models
cancer_type_model = tf.keras.models.load_model(cancer_type_model_path)

models_dict = {
    "breast_cancer": tf.keras.models.load_model(os.path.join(MODEL_DIR, "breast_cancer_model.keras")),
    "lung_cancer": tf.keras.models.load_model(os.path.join(MODEL_DIR, "lung_cancer_model.keras")),
    "skin_cancer": tf.keras.models.load_model(os.path.join(MODEL_DIR, "skin_cancer_model.keras")),
}

# Class mapping
cancer_types = ["breast_cancer", "lung_cancer", "skin_cancer"]

def predict_cancer(image_path):
    if not os.path.exists(image_path):
        print(f" Error: Image '{image_path}' not found.")
        return
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Step 1: Predict Cancer Type
    cancer_type_pred = cancer_type_model.predict(img_array)
    cancer_index = np.argmax(cancer_type_pred)
    detected_cancer = cancer_types[cancer_index]
    print(f"🔍 Detected Cancer Type: {detected_cancer} (Confidence: {cancer_type_pred[0][cancer_index]:.4f})")

    # Step 2: Predict Benign/Malignant
    malignancy_pred = models_dict[detected_cancer].predict(img_array)
    malignancy_result = "Malignant" if malignancy_pred[0][0] > 0.5 else "Benign"
    print(f"🔍 {detected_cancer} Prediction: {malignancy_result} (Confidence: {malignancy_pred[0][0]:.4f})")

# Test Image
test_image = r"E:\multi_cancer_detection_using_ML\datasets\benign_malignant\Lung_cancer\benign\Bengin case (3).jpg"  # Change path to an actual test image
if os.path.exists(test_image):
    predict_cancer(test_image)
else:
    print(f" Error: Test image '{test_image}' not found.")
