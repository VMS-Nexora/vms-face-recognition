from deepface import DeepFace
import numpy as np


def extract_face_descriptor(image_path):
    """Extract face embedding from an image."""
    embedding = DeepFace.represent(
        image_path, model_name='Facenet', enforce_detection=True)
    return np.array(embedding[0]['embedding']).tobytes()  # Convert to bytes


def compare_faces(descriptor, stored_descriptor):
    """Compare two face descriptors."""
    stored_embedding = np.frombuffer(bytes.fromhex(
        stored_descriptor))  # Decode hex to bytes
    distance = np.linalg.norm(np.frombuffer(descriptor) - stored_embedding)
    return distance < 0.6  # Threshold for match
