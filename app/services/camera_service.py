from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import base64
import io
import os
import json
import cv2
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

USERS_DIR = 'db/users'
os.makedirs(USERS_DIR, exist_ok=True)


def process_images(images_data):
    """Extract face encodings from base64 images"""
    face_encodings = []
    for image_data in images_data:
        try:
            image_data = image_data.split(
                ',')[1] if "," in image_data else image_data
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            np_image = np.array(image)

            encodings = face_recognition.face_encodings(np_image)
            if encodings:
                face_encodings.append(encodings[0])
        except Exception as e:
            print(f"Error processing image: {e}")

    return face_encodings


def anti_spoof_check(image_data):
    """Detect spoofing by checking multiple image properties"""
    is_sharp = check_image_sharpness(image_data)
    # has_no_reflection = detect_reflection(image_data)
    is_blinking = detect_blink(image_data)

    return is_sharp and is_blinking


def check_image_sharpness(image_data):
    """Check if image is sharp enough to be a real capture"""
    try:
        image_data = image_data.split(
            ',')[1] if "," in image_data else image_data
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        np_image = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # Calculate Laplacian variance (measure of sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Higher variance means sharper image
        return laplacian_var > 100  # Threshold can be adjusted
    except Exception as e:
        print(f"Error in sharpness detection: {e}")
        return False


def detect_reflection(image_data):
    """Detect screen reflections to prevent spoofing"""
    try:
        image_data = image_data.split(
            ',')[1] if "," in image_data else image_data
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        np_image = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Count the number of white pixels (highlights/reflections)
        reflection_ratio = np.sum(thresh == 255) / 2

        return reflection_ratio < 0.05  # If too many white pixels, likely a screen
    except Exception as e:
        print(f"Error in reflection detection: {e}")
        return False


def detect_blink(image_data):
    """Check if eyes are open or closed to prevent static image spoofing"""
    try:
        image_data = image_data.split(
            ',')[1] if "," in image_data else image_data
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        np_image = np.array(image)

        face_landmarks = face_recognition.face_landmarks(np_image)
        if not face_landmarks:
            return False

        left_eye = face_landmarks[0]['left_eye']
        right_eye = face_landmarks[0]['right_eye']

        def eye_aspect_ratio(eye):
            """Compute Eye Aspect Ratio (EAR) to detect blinking"""
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        return avg_ear > 0.2  # If too small, eyes are likely closed
    except Exception as e:
        print(f"Error in blink detection: {e}")
        return False


def save_user(user_id, name, email, face_encoding):
    """Save user info and face encoding in separate folders"""
    user_folder = os.path.join(USERS_DIR, user_id)
    faces_folder = os.path.join(user_folder, "faces")
    info_file = os.path.join(user_folder, "user.json")

    os.makedirs(faces_folder, exist_ok=True)

    # Save face encoding
    face_file = os.path.join(
        faces_folder, f"face_{len(os.listdir(faces_folder))}.json")
    with open(face_file, 'w') as f:
        json.dump(face_encoding.tolist(), f)

    # Append user info with timestamp
    user_data = {"name": name, "email": email,
                 "timestamp": datetime.now().isoformat()}
    user_info = []
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            user_info = json.load(f)

    user_info.append(user_data)  # Append new registration info
    with open(info_file, 'w') as f:
        json.dump(user_info, f)


def load_users():
    """Load all stored face encodings and user information"""
    users = {}
    user_info = {}

    for user_id in os.listdir(USERS_DIR):
        user_folder = os.path.join(USERS_DIR, user_id)
        faces_folder = os.path.join(user_folder, "faces")
        info_file = os.path.join(user_folder, "user.json")

        encodings = []
        for file in os.listdir(faces_folder):
            with open(os.path.join(faces_folder, file), 'r') as f:
                encodings.append(np.array(json.load(f)))
        users[user_id] = encodings

        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                user_history = json.load(f)
                # Return latest registration info
                user_info[user_id] = user_history[-1]

    return users, user_info


def find_closest_match(face_encoding, users):
    """Find the closest matching user using k-NN"""
    encodings = []
    labels = []

    for user_id, faces in users.items():
        for face in faces:
            encodings.append(face)
            labels.append(user_id)

    if encodings:
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn.fit(encodings, labels)
        user_id = knn.predict([face_encoding])[0]
        return user_id

    return None


@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user with face data"""
    data = request.json
    if 'images' not in data or 'name' not in data or 'email' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    if not all(anti_spoof_check(img) for img in data['images']):
        return jsonify({'error': 'Spoofing attempt detected'}), 403

    face_encodings = process_images(data['images'])
    if not face_encodings:
        return jsonify({'error': 'No face detected'}), 400

    users, user_info = load_users()
    best_match = find_closest_match(face_encodings[0], users)
    user_id = best_match if best_match else f"user_{len(users) + 1}"

    for encoding in face_encodings:
        save_user(user_id, data['name'], data['email'], encoding)

    return jsonify({'message': 'User registered successfully', 'user_id': user_id}), 201


@app.route('/api/checkin', methods=['POST'])
def checkin():
    """Check-in user by recognizing their face"""
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'Missing image'}), 400

    if not anti_spoof_check(data['image']):
        return jsonify({'error': 'Spoofing attempt detected'}), 403

    face_encodings = process_images([data['image']])
    if not face_encodings:
        return jsonify({'error': 'No face detected'}), 400

    users, user_info = load_users()
    best_match = find_closest_match(face_encodings[0], users)

    if best_match and best_match in user_info:
        return jsonify({'match': True, 'user_id': best_match, 'name': user_info[best_match]['name'], 'email': user_info[best_match]['email']}), 200

    return jsonify({'match': False, 'error': 'User not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)
