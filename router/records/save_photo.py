from flask import Blueprint, request, jsonify, session, render_template
import os, base64, time
from db import get_connection
from datetime import datetime, timedelta

UPLOAD_FOLDER = "static/uploads/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

save_photo_bp = Blueprint("save_photo", __name__)

# ---------------- 사진 저장 ----------------
@save_photo_bp.route("/save_photo", methods=["POST"])
def save_photo():
    if "user_id" not in session:
        return jsonify({"error": "로그인 필요"}), 403

    kst_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = request.json
    user_id = session["user_id"]
    cva = data.get("cva")
    status = data.get("status")
    image_base64 = data.get("image")

    if cva is None or status is None or image_base64 is None:
        return jsonify({"error": "값이 올바르지 않습니다"}), 400

    # 이미지 디코딩 및 저장
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    filename = f"{int(time.time())}_{user_id}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO cva_records (user_id, cva_value, status, image_path, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, cva, status, filepath, kst_str)
            )
            conn.commit()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "사진 저장 완료", "path": filepath})


images_bp = Blueprint("records_images", __name__)  # 이름 고유하게

@images_bp.route("/records_images")
def records_images():
    if "user_id" not in session:
        return redirect(url_for("index"))

    user_id = str(session["user_id"])
    
    if user_id.startswith("guest"):
        username = "게스트"
        videos = [] 
        return render_template("records/images.html", username=username, videos=videos)

    # DB에서 사용자 정보
    conn = get_connection()
    user = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    records = conn.execute(
        "SELECT * FROM cva_records WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    conn.close()

    return render_template("records/images.html", username=user["username"], records=records)