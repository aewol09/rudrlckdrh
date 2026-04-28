from flask import Blueprint, jsonify, session, render_template, request, url_for
import os, time, subprocess, base64
from datetime import datetime, timedelta
from db import get_connection

HIGHLIGHT_FOLDER = "static/uploads/highlight"
os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)

highlight_bp = Blueprint("highlight_videos", __name__)

@highlight_bp.route("/save_highlight", methods=["POST"])
def save_highlight():
    if "user_id" not in session:
        return jsonify({"error": "로그인 필요"}), 403

    kst_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = request.json
    video_data = data.get("video")
    if not video_data:
        return jsonify({"error": "video 필수"}), 400

    user_id = session["user_id"]
    timestamp = int(time.time())
    highlight_file = os.path.join(HIGHLIGHT_FOLDER, f"{timestamp}_{user_id}_highlight.webm")

# DataURL -> 파일 저장
    header, encoded = video_data.split(",", 1)
    with open(highlight_file, "wb") as f:
        f.write(base64.b64decode(encoded))

    # DB 저장
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO highlight_videos (user_id, video_path, timestamp) VALUES (?, ?, ?)",
            (user_id, highlight_file, kst_str)
        )
        conn.commit()

    return jsonify({"status": "highlight_saved", "path": highlight_file})

@highlight_bp.route("/videos_highlight")
def show_highlight_videos():
    if "user_id" not in session:
        return "로그인 필요", 403

    user_id = str(session["user_id"])
    
    if user_id.startswith("guest"):
        username = "게스트"
        videos = []
        return render_template("records/highlight.html", username=username, videos=videos)

    with get_connection() as conn:
        user = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        rows = conn.execute(
            "SELECT * FROM highlight_videos WHERE user_id = ? ORDER BY timestamp DESC", (user_id,)
        ).fetchall()

    videos = [{"path": url_for("static", filename=row["video_path"].replace("static/", "")), "timestamp": row["timestamp"]} for row in rows]

    return render_template("records/highlight.html", username=user["username"], videos=videos)