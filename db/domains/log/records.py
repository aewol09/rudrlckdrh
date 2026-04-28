import sqlite3, os, bcrypt

# 직접 get_connection 정의
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../database.db")

def get_connection():
    conn = sqlite3.connect(DB_NAME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_connection()
conn.execute(
    "INSERT INTO cva_records (user_id, cva_value, status, image_path) VALUES (?, ?, ?, ?)",
    (user_id, cva, status, filepath)
)
conn.commit()
conn.close()