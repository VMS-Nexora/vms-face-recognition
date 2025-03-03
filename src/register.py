# Register a visitor (run this separately or integrate into VMS)
import os
from pb_client import pb_client
from face_recognition import extract_face_descriptor


image_path = os.path.join(os.path.dirname(__file__), "image_face.jpeg")
image_path = os.path.abspath(image_path)  # Convert to absolute path

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

descriptor = extract_face_descriptor(image_path)
pb_client.create_visitor(
    name="Eithan",
    email="Eithan@example.com",
    phone="123-456-7890",
    face_descriptor=descriptor
)
