# facenet.py

import numpy as np
from keras.models import load_model
from PIL import Image
import h5py

# Function to load the pre-trained Facenet model
def load_facenet_model(model_path):
    try:
        with h5py.File(model_path, 'r') as f:
            print("Model file is readable.")
        model = load_model(model_path)
        print("Facenet model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((160, 160))  # Resize image to 160x160 pixels
        img_array = np.array(img).astype('float32') / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to generate embedding
def generate_face_embedding(model, processed_image):
    if model is None or processed_image is None:
        print("Model or image not loaded.")
        return None
    try:
        embedding = model.predict(processed_image)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
