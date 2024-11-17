import h5py
from keras.models import load_model

model_path = r'D:\pythonn\face_recognition_project\facenet_keras.h5'

# Check if the model file is readable
try:
    with h5py.File(model_path, 'r') as f:
        print("Model file is readable.")
except Exception as e:
    print(f"Model file is corrupted or not readable: {e}")

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
