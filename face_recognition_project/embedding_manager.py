import os
import numpy as np
from facenet import preprocess_image, generate_face_embedding, load_facenet_model
import pickle

# Function to generate and save embeddings for known faces
def generate_known_embeddings(model, known_faces_folder):
    known_embeddings = []
    known_labels = []
    
    for person_name in os.listdir(known_faces_folder):
        person_folder = os.path.join(known_faces_folder, person_name)
        
        # Ensure this is a directory
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                image_path = os.path.join(person_folder, filename)
                
                # Preprocess the image and generate an embedding
                processed_image = preprocess_image(image_path)
                embedding = generate_face_embedding(model, processed_image)
                
                if embedding is not None:
                    known_embeddings.append(embedding)
                    known_labels.append(person_name)  # Label with the person's name
                    
    return known_embeddings, known_labels

# Function to save embeddings and labels to a file using pickle
def save_embeddings_to_file(embeddings, labels, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((embeddings, labels), f)

# Function to load embeddings and labels from a file using pickle
def load_embeddings_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Load model
    model_path = r'C:\path\to\facenet_keras.h5'
    model = load_facenet_model(model_path)
    
    # Folder containing known faces (organized by person)
    known_faces_folder = r'C:\path\to\known_faces'
    
    # Generate embeddings and labels for the known faces
    known_embeddings, known_labels = generate_known_embeddings(model, known_faces_folder)
    
    # Save embeddings and labels to a file
    save_embeddings_to_file(known_embeddings, known_labels, 'known_embeddings.pkl')
    
    print("Embeddings for known faces have been saved.")
