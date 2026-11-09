
import sqlite3
import hashlib
import secrets
from pathlib import Path

DB_FILE = Path(__file__).parent / "users.db"


def connect_db():
    return sqlite3.connect(DB_FILE)


def create_table():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS favorites (
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


def register_user(username, email, password):
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100000
    ).hex()

    try:
        with connect_db() as conn:
            conn.execute(
                """INSERT INTO users
                (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)""",
                (username, email, password_hash, salt)
            )
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def login_user(username_or_email, password):
    with connect_db() as conn:
        user = conn.execute(
            """SELECT id, username, email, password_hash, salt
            FROM users WHERE username = ? OR email = ?""",
            (username_or_email, username_or_email)
        ).fetchone()

    if user is None:
        return None

    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), user[4].encode(), 100000
    ).hex()

    if secrets.compare_digest(password_hash, user[3]):
        return {"id": user[0], "username": user[1], "email": user[2]}
    return None


def get_favorites(user_id):
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT movie_id FROM favorites WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return {row[0] for row in rows}


def add_favorite(user_id, movie_id):
    with connect_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO favorites (user_id, movie_id) VALUES (?, ?)",
            (user_id, movie_id)
        )


def remove_favorite(user_id, movie_id):
    with connect_db() as conn:
        conn.execute(
            "DELETE FROM favorites WHERE user_id = ? AND movie_id = ?",
            (user_id, movie_id)
        )


create_table()
