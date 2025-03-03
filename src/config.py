# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    PB_URL = os.getenv('PB_URL', 'https://eithan-bku.pockethost.io')
    PB_ADMIN_EMAIL = os.getenv('PB_ADMIN_EMAIL', 'admin@example.com')
    PB_ADMIN_PASSWORD = os.getenv('PB_ADMIN_PASSWORD', 'admin@example.com')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
