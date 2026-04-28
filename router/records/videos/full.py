from flask import Blueprint, request, jsonify, session, render_template, url_for
import os, base64, time
from db import get_connection
from datetime import datetime, timedelta

# 업로드 폴더
UPLOAD_FOLDER = "static/uploads/full_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

full_videos_bp = Blueprint("full_videos", __name__)

# =======================
# 1️⃣ 풀영상 저장
# =======================
@full_videos_bp.route("/full_videos", methods=["POST"])
def save_full_video():
    if "user_id" not in session:
        return jsonify({"error": "로그인 필요"}), 403

    kst_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = request.json
    user_id = session["user_id"]
    video_base64 = data.get("video")
    print(data)
    forward_count = data.get("count", 0) 
    print("Received forward_count:", forward_count)
    
    if not video_base64:
        return jsonify({"error": "영상 데이터 없음"}), 400

    # base64 디코딩
    try:
        if "," in video_base64:
            video_bytes = base64.b64decode(video_base64.split(",")[1])
        else:
            video_bytes = base64.b64decode(video_base64)
    except Exception as e:
        return jsonify({"error": f"base64 디코딩 실패: {e}"}), 400

    # 파일 저장
    filename = f"{int(time.time())}_{user_id}_full.webm"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        return jsonify({"error": f"파일 저장 실패: {e}"}), 500

    # DB 저장
    try:
        with get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO full_videos (user_id, video_path, timestamp, forward_count) VALUES (?, ?, ?, ?)",
                (user_id, filepath, kst_str, forward_count)
            )
            conn.commit()
            video_id = cur.lastrowid
    except Exception as e:
        return jsonify({"error": f"DB 저장 실패: {e}"}), 500

    return jsonify({"message": "풀영상 저장 완료", "path": filepath, "id": video_id, "count": forward_count})


# =======================
# 2️⃣ 풀영상 페이지 표시
# =======================
@full_videos_bp.route("/full_videos")
def show_full_videos():
    if "user_id" not in session:
        return "로그인 필요", 403

    user_id = str(session["user_id"])

    # 게스트 처리
    if user_id.startswith("guest"):
        username = "게스트"
        videos = []  # 게스트는 DB 조회 불가 → 비어있게
        return render_template("records/full.html", username=username, videos=videos)

    # 일반 유저
    conn = get_connection()
    user = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return "사용자 정보 없음", 404

    videos = conn.execute(
        "SELECT * FROM full_videos WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    return render_template("records/full.html", username=user["username"], videos=videos)