import os
import pickle
from deepface import DeepFace

KNOWN_PERSONS_DIR = r'D:\pythonn\deep_face\persons'
EMBEDDINGS_PATH = r'D:\pythonn\deep_face\known_embeddings.pkl'

def generate_and_save_embeddings():
    known_embeddings = {}

    for person_name in os.listdir(KNOWN_PERSONS_DIR):
        person_dir = os.path.join(KNOWN_PERSONS_DIR, person_name)
        if os.path.isdir(person_dir):
            known_embeddings[person_name] = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                print(f"Generating embedding for {image_path}")
                
                # Extract only the embedding array
                embedding = DeepFace.represent(image_path, model_name='Facenet')[0]['embedding']
                known_embeddings[person_name].append(embedding)

    # Save the embeddings to a file
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(known_embeddings, f)

if __name__ == "__main__":
    generate_and_save_embeddings()
