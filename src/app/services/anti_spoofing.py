import cv2
import numpy as np
import face_recognition
from PIL import Image
import io
import base64

def decode_image(image_data):
    """Decode base64 image to NumPy array."""
    image_data = image_data.split(',')[1] if "," in image_data else image_data
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    return np.array(image)

def check_image_sharpness(image_data):
    """Check if image is sharp enough to be a real capture."""
    try:
        np_image = decode_image(image_data)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() > 100  # Adjustable threshold
    except Exception as e:
        print(f"Error in sharpness detection: {e}")
        return False

def detect_blink(image_data):
    """Detect blinking to prevent static image spoofing."""
    try:
        np_image = decode_image(image_data)
        landmarks = face_recognition.face_landmarks(np_image)
        if not landmarks:
            return False

        def eye_aspect_ratio(eye):
            """Compute Eye Aspect Ratio (EAR)."""
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (A + B) / (2.0 * C)

        left_eye = landmarks[0]['left_eye']
        right_eye = landmarks[0]['right_eye']
        return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0 > 0.2
    except Exception as e:
        print(f"Error in blink detection: {e}")
        return False

def anti_spoof_check(image_data):
    """Run all anti-spoofing checks."""
    return check_image_sharpness(image_data) and detect_blink(image_data)
