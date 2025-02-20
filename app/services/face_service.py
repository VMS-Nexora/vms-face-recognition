import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def process_images(images_data):
    """Extract face encodings from base64 images."""
    face_encodings = []
    for image_data in images_data:
        try:
            image = face_recognition.load_image_file(image_data)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings.append(encodings[0])
        except Exception as e:
            print(f"Error processing image: {e}")
    return face_encodings

def find_closest_match(face_encoding, users):
    """Find the closest matching user using k-NN."""
    encodings = []
    labels = []

    for user_id, faces in users.items():
        for face in faces:
            encodings.append(face)
            labels.append(user_id)

    if encodings:
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn.fit(encodings, labels)
        return knn.predict([face_encoding])[0]

    return None
