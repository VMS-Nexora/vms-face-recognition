import face_recognition
import numpy as np
from camera_service import check_image_sharpness, detect_blink

def process_images(images_data):
    """Extract face encodings from base64 images"""
    face_encodings = []
    for image_data in images_data:
        try:
            encodings = face_recognition.face_encodings(image_data)
            if encodings:
                face_encodings.append(encodings[0])
        except Exception as e:
            print(f"Error processing image: {e}")
    return face_encodings

def anti_spoof_check(image_data):
    """Detect spoofing by checking multiple image properties"""
    return check_image_sharpness(image_data) and detect_blink(image_data)

# camera_service.py
import cv2
import numpy as np
from PIL import Image
import io
import base64

def check_image_sharpness(image_data):
    """Check if image is sharp enough to be a real capture"""
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        np_image = np.array(image)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > 100  # Threshold can be adjusted
    except Exception as e:
        print(f"Error in sharpness detection: {e}")
        return False

def detect_blink(image_data):
    """Check if eyes are open or closed to prevent static image spoofing"""
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        np_image = np.array(image)
        face_landmarks = face_recognition.face_landmarks(np_image)
        if not face_landmarks:
            return False
        left_eye = face_landmarks[0]['left_eye']
        right_eye = face_landmarks[0]['right_eye']
        def eye_aspect_ratio(eye):
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (A + B) / (2.0 * C)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear > 0.2
    except Exception as e:
        print(f"Error in blink detection: {e}")
        return False
