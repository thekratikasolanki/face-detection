# face_recognition.py

import pickle
import numpy as np
from facenet import load_facenet_model, preprocess_image, generate_face_embedding
from scipy.spatial.distance import cosine

# Path to the pre-trained Facenet model
model_path = r'D:\pythonn\face_recognition_project\facenet_keras.h5'  # Update this path if necessary

# Path to the known embeddings pickle file
embeddings_path = r'D:\pythonn\face_recognition_project\known_embeddings.pkl'  # Update this path if necessary

# Function to load known embeddings
def load_known_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['labels']

# Function to recognize a person from a test image
def recognize_person(test_image_path, model, known_embeddings, known_labels, threshold=0.5):
    # Preprocess the test image
    print(f"Preprocessing test image: {test_image_path}")
    processed_image = preprocess_image(test_image_path)
    
    # Generate embedding for the test image
    print("Generating embedding for the test image...")
    test_embedding = generate_face_embedding(model, processed_image)
    if test_embedding is None:
        print("Failed to generate embedding for the test image.")
        return
    
    test_embedding = test_embedding.flatten()
    
    # Compare with known embeddings
    similarities = []
    for known_embedding in known_embeddings:
        similarity = 1 - cosine(test_embedding, known_embedding)
        similarities.append(similarity)
    
    # Find the best match
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]
    
    print(f"Best similarity: {best_similarity}")
    
    if best_similarity >= threshold:
        recognized_person = known_labels[best_match_index]
        print(f"Recognized person: {recognized_person}")
    else:
        print("No match found.")

# Main function
def main():
    # Path to the test image
    test_image_path = r'D:\pythonn\face_recognition_project\test_images\test_face.jpg'  # Update this path
    
    # Load the Facenet model
    print("Loading Facenet model...")
    model = load_facenet_model(model_path)
    if model is None:
        print("Failed to load Facenet model. Exiting...")
        return
    
    # Load known embeddings and labels
    print("Loading known embeddings...")
    known_embeddings, known_labels = load_known_embeddings(embeddings_path)
    print(f"Loaded embeddings for {len(known_labels)} people.")
    
    # Recognize the person in the test image
    recognize_person(test_image_path, model, known_embeddings, known_labels, threshold=0.5)

if __name__ == '__main__':
    main()
