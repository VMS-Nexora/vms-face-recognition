# src/pb_client.py
import requests
from config import Config


class PocketBaseClient:
    def __init__(self):
        self.base_url = Config.PB_URL
        self.token = self._authenticate()

    def _authenticate(self):
        """Authenticate as admin and get token."""
        response = requests.post(
            f"{self.base_url}/api/admins/auth-with-password",
            json={
                "identity": Config.PB_ADMIN_EMAIL,
                "password": Config.PB_ADMIN_PASSWORD,
            }
        )
        response.raise_for_status()
        return response.json()["token"]

    def create_visitor(self, name, email, phone, face_descriptor):
        """Create a new visitor record."""
        response = requests.post(
            f"{self.base_url}/api/collections/visitors/records",
            headers={"Authorization": self.token},
            json={
                "name": name,
                "email": email,
                "phone": phone,
                "face_descriptor": face_descriptor.hex()  # Store as hex string
            }
        )
        response.raise_for_status()
        return response.json()

    def get_visitor_by_descriptor(self, descriptor):
        """Find a visitor by face descriptor."""
        hex_descriptor = descriptor.hex()
        response = requests.get(
            f"{self.base_url}/api/collections/visitors/records",
            headers={"Authorization": self.token},
            params={"filter": f'face_descriptor="{hex_descriptor}"'}
        )
        response.raise_for_status()
        records = response.json()["items"]
        return records[0] if records else None


# Initialize client
pb_client = PocketBaseClient()
