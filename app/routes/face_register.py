from flask import Flask, request, jsonify
from flask_cors import CORS
from face_service import process_images, find_closest_match
from anti_spoofing import anti_spoof_check
from user_service import save_user, load_users

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user with face data."""
    data = request.json
    if not all(k in data for k in ('images', 'name', 'email')):
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
    """Check-in user by recognizing their face."""
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
        return jsonify({'match': True, 'user_id': best_match, 'name': user_info[best_match]['name']}), 200

    return jsonify({'match': False, 'error': 'User not found'}), 404

