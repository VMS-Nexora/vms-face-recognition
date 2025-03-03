import os
import json
import numpy as np
from datetime import datetime

USERS_DIR = 'db/users'
os.makedirs(USERS_DIR, exist_ok=True)

def save_user(user_id, name, email, face_encoding):
    """Save user data and face encoding."""
    user_folder = os.path.join(USERS_DIR, user_id)
    faces_folder = os.path.join(user_folder, "faces")
    info_file = os.path.join(user_folder, "user.json")

    os.makedirs(faces_folder, exist_ok=True)

    # Save face encoding
    face_file = os.path.join(faces_folder, f"face_{len(os.listdir(faces_folder))}.json")
    with open(face_file, 'w') as f:
        json.dump(face_encoding.tolist(), f)

    # Append user info
    user_data = {"name": name, "email": email, "timestamp": datetime.now().isoformat()}
    user_info = []
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            user_info = json.load(f)

    user_info.append(user_data)
    with open(info_file, 'w') as f:
        json.dump(user_info, f)

def load_users():
    """Load stored face encodings & user info."""
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
                user_info[user_id] = json.load(f)[-1]  # Return latest registration info

    return users, user_info
