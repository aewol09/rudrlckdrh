import sqlite3
import os
import bcrypt

from .domains.users.account import add_user
from .domains.auth.verify import verify_user
from .domains.auth.username import get_user_by_username

# DB 파일 경로
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def get_connection():
    """DB 연결 반환"""
    conn = sqlite3.connect(DB_NAME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """DB 초기화 (테이블 생성)"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS cva_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            cva_value REAL NOT NULL,
            status TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        
        CREATE TABLE IF NOT EXISTS full_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_path TEXT NOT NULL,
            timestamp TEXT NOT NULL,  -- 업로드/녹화 시간
            forward_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS highlight_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_path TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS checklist (
            user_id INTEGER PRIMARY KEY,
            cold_tingling_posture INTEGER DEFAULT 0,
            neck_cracking_fatigue_headaches INTEGER DEFAULT 0,
            bad_sleeping_habit INTEGER DEFAULT 0,
            computer_over_8h INTEGER DEFAULT 0,
            neck_shoulder_stiffness INTEGER DEFAULT 0,
            unrefreshing_sleep INTEGER DEFAULT 0,
            eye_strain_headaches INTEGER DEFAULT 0,
            updated_at DATETIME,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        
        CREATE TABLE IF NOT EXISTS guest_checklist (
            guest_user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_session_id TEXT UNIQUE,  -- 세션별 고유 ID
            cold_tingling_posture INTEGER DEFAULT 0,
            neck_cracking_fatigue_headaches INTEGER DEFAULT 0,
            bad_sleeping_habit INTEGER DEFAULT 0,
            computer_over_8h INTEGER DEFAULT 0,
            neck_shoulder_stiffness INTEGER DEFAULT 0,
            unrefreshing_sleep INTEGER DEFAULT 0,
            eye_strain_headaches INTEGER DEFAULT 0,
            updated_at DATETIME
        );

    """)

    conn.commit()
    conn.close()
