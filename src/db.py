# src/db.py
import psycopg2
from psycopg2.extras import RealDictCursor
from config import Config


def get_db_connection():
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        database=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        cursor_factory=RealDictCursor
    )


def get_visitor_by_descriptor(descriptor):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Assuming face_descriptor is stored as a binary string
            cur.execute(
                "SELECT id, name, email, phone FROM visitors WHERE face_descriptor = %s",
                (descriptor,)
            )
            return cur.fetchone()
    finally:
        conn.close()
