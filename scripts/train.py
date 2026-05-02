import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))

CANCER_TYPE_DIR = os.path.join(BASE_DIR, "Cancer_type")
BENIGN_MALIGNANT_DIR = os.path.join(BASE_DIR, "Benign_malignant")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Common parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators for Cancer Type classification
cancer_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen_cancer = cancer_datagen.flow_from_directory(
    CANCER_TYPE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen_cancer = cancer_datagen.flow_from_directory(
    CANCER_TYPE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Define and train Cancer Type Classification Model
cancer_type_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 cancer types
])

cancer_type_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("📂 Training Cancer Type Model...")
cancer_type_model.fit(train_gen_cancer, validation_data=val_gen_cancer, epochs=EPOCHS)
cancer_type_model.save(os.path.join(MODEL_DIR, "cancer_type_model.keras"))

# Train Benign/Malignant Models for each cancer type
cancer_types = ["breast_cancer", "lung_cancer", "skin_cancer"]

for cancer in cancer_types:
    print(f"📂 Training Benign/Malignant Model for {cancer}...")
    
    train_path = os.path.join(BENIGN_MALIGNANT_DIR, cancer)
    
    train_gen = cancer_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",  # Benign or Malignant
        subset="training"
    )

    val_gen = cancer_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    
    # Save model
    model.save(os.path.join(MODEL_DIR, f"{cancer}_model.keras"))

print("🎯 All models trained and saved successfully!")
