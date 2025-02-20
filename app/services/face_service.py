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


