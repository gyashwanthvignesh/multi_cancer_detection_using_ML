import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Set image size
IMG_SIZE = (224, 224)  # Use the same size used during training

def preprocess_image(image_path):
    """Preprocess an image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to load image at {image_path}")
    
    img = cv2.resize(img, IMG_SIZE)  # Resize to match training size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
