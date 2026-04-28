import sqlite3, os, bcrypt

# 직접 get_connection 정의
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../database.db")

def get_connection():
    conn = sqlite3.connect(DB_NAME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_username(username: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user