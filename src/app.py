from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
from telegram import Bot
import asyncio

app = Flask(__name__)

# Telegram setup
TELEGRAM_BOT_TOKEN = "8083127180:AAFHNKBMlP-n0_AYZKOu8iw6KyTZPBrzYH8"
TELEGRAM_CHAT_ID = "1660673482"
bot = Bot(TELEGRAM_BOT_TOKEN)

# Directory containing visitor images
VISITOR_IMAGES_DIR = "visitors"  # Adjust path as needed
known_visitors = {}  # {visitor_id: face_encoding}

# Load visitor images and encodings at startup


def load_visitor_encodings():
    if not os.path.exists(VISITOR_IMAGES_DIR):
        os.makedirs(VISITOR_IMAGES_DIR)
        print(
            f"Created directory: {VISITOR_IMAGES_DIR}. Please add visitor images.")
        return

    for filename in os.listdir(VISITOR_IMAGES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            visitor_id = os.path.splitext(filename)[0]  # Use filename as ID
            image_path = os.path.join(VISITOR_IMAGES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Ensure at least one face is found
                known_visitors[visitor_id] = encodings[0]
                print(f"Loaded visitor: {visitor_id}")
            else:
                print(f"No face found in {filename}")


# Load visitors when the app starts
load_visitor_encodings()


async def send_telegram_message(message):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)


@app.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        # Check if image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        # Read image into numpy array
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Detect and encode faces
        face_locations = face_recognition.face_locations(img)
        if not face_locations:
            return jsonify({"error": "No face detected"}), 400

        face_encodings = face_recognition.face_encodings(img, face_locations)
        if not face_encodings:
            return jsonify({"error": "Face encoding failed"}), 400

        # Compare with known visitors
        current_encoding = face_encodings[0]
        visitor_id = None

        if not known_visitors:
            return jsonify({"error": "No visitors registered"}), 404

        for known_id, known_encoding in known_visitors.items():
            match = face_recognition.compare_faces(
                [known_encoding], current_encoding, tolerance=0.6)[0]
            if match:
                visitor_id = known_id
                break

        if visitor_id:
            # Send to Telegram asynchronously
            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_telegram_message(
                f"Visitor authenticated: {visitor_id}"))
            return jsonify({"message": "Authentication successful", "visitorId": visitor_id}), 200
        else:
            return jsonify({"error": "Visitor not recognized"}), 401

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
