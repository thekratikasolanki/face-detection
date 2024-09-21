# generate_embeddings.py

import os
import pickle
from facenet import load_facenet_model, preprocess_image, generate_face_embedding

# Path to the pre-trained Facenet model
model_path = r'D:\pythonn\face_recognition_project\facenet_keras.h5'  # Update this path if necessary

# Directory containing subfolders of images for each person
dataset_dir = r'D:\pythonn\face_recognition_project\dataset'  # Update this path if necessary

# Function to generate and save embeddings
def generate_and_save_embeddings():
    # Load the Facenet model
    print("Loading Facenet model...")
    model = load_facenet_model(model_path)
    
    if model is None:
        print("Failed to load the model. Exiting...")
        return
    
    known_embeddings = []
    known_names = []
    
    # Iterate through each person's folder
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue  # Skip files, only process directories
        
        print(f"Processing images for {person_name}...")
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            print(f"Processing {image_path}...")
            processed_image = preprocess_image(image_path)
            if processed_image is not None:
                embedding = generate_face_embedding(model, processed_image)
                if embedding is not None:
                    known_embeddings.append(embedding.flatten())  # Flatten to 1D array
                    known_names.append(person_name)
    
    # Save the embeddings and names to a pickle file
    with open('known_embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': known_embeddings, 'labels': known_names}, f)
    print("Embeddings generated and saved to known_embeddings.pkl.")

if __name__ == '__main__':
    generate_and_save_embeddings()
