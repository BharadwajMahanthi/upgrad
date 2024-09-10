import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Correct import for ImageDataGenerator
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
MODEL_PATH = r'C:\Users\mbpd1\Downloads\upgrad\capstone\FF_SM_NSM\output\fire_detection_best_model.h5'
model = load_model(MODEL_PATH)

# Define image size and class names (these should match your training configuration)
IMG_SIZE = (128, 128)  # Ensure this matches the input size used during model training
CLASS_NAMES = ['non fire', 'fire', 'Smoke']  # Update class names to match those used during training

# Updated Hyperparameters
INIT_LR = 1e-5  # Reduced learning rate for finer updates
BATCH_SIZE = 16  # Smaller batch size to reduce memory load and potentially increase model generalization

# Set your directories for train and validation
base_dir = r'C:\Users\mbpd1\Downloads\upgrad\capstone\FF_SM_NSM'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Data augmentation and loading using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1.0/255.0  # Normalize images
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Flow from directory to load images batch-wise
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided!"})

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": f"{confidence*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
