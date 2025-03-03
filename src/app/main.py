from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({"message": "Vadio Face Recognition API is running!"})


@app.route('/face/detect', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return jsonify({"faces_detected": len(faces)})


if __name__ == '__main__':
    app.run(debug=True)
