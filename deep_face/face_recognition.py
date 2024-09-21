import os
import pickle
from deepface import DeepFace
import numpy as np

# Path to the embeddings file and test image
EMBEDDINGS_PATH = r'D:\pythonn\deep_face\known_embeddings.pkl'
TEST_IMAGE_PATH = r'D:\pythonn\deep_face\test_image.jpg'  # Update this with your test image path

# Load the known embeddings
with open(EMBEDDINGS_PATH, 'rb') as f:
    known_embeddings = pickle.load(f)

# Function to recognize a person from a test image
def recognize_person(test_image_path):
    # Generate embedding for the test image
    print(f"Generating embedding for the test image {test_image_path}")
    test_embedding = DeepFace.represent(test_image_path, model_name='Facenet')

    # Initialize variables to store the best match
    best_match = None
    best_distance = float('inf')
    
    # Compare the test embedding with known embeddings
    for person_name, embeddings in known_embeddings.items():
        for embedding in embeddings:
            distance = np.linalg.norm(np.array(embedding) - np.array(test_embedding))
            if distance < best_distance:
                best_distance = distance
                best_match = person_name
    
    # Threshold to determine if the person is recognized
    threshold = 10  # You can adjust this value based on your needs
    
    if best_distance < threshold:
        print(f"Person recognized: {best_match} with distance {best_distance}")
    else:
        print("Person not recognized.")

# Recognize the person in the test image
if __name__ == "__main__":
    recognize_person(TEST_IMAGE_PATH)
