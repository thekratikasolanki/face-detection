# embedding_manager.py

import pickle
import os

# Path to the saved embeddings
EMBEDDINGS_PATH = r'pythonn/deep_face/known_embeddings.pkl'

# Function to load known embeddings from file
def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
        print("Embeddings loaded successfully.")
        return embeddings
    else:
        print("Embeddings file not found.")
        return None
