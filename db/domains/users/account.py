import sqlite3, os, bcrypt

# 직접 get_connection 정의
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../database.db")

def get_connection():
    conn = sqlite3.connect(DB_NAME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
    
def add_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # bcrypt 해시 생성
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed.decode())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # 이미 존재하는 사용자
        return False
    finally:
        conn.close()