import sqlite3, os, bcrypt
from .username import get_user_by_username

def verify_user(username: str, password_input: str) -> bool:
    user = get_user_by_username(username)
    if not user:
        return False
    stored_hash = user["password"]
    return bcrypt.checkpw(password_input.encode(), stored_hash.encode())