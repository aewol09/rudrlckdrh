from flask import Blueprint, render_template, request, redirect, url_for, session
from db import get_connection
from datetime import datetime, timedelta
import uuid

user_service_bp = Blueprint("user_service", __name__)

# 체크리스트 영어 key → 한국어 문장 매핑
FACTOR_KR = {
    "Cold, tingling hands and feet, and poor posture.": "손발이 차고 자주 저리고, 평소 구부정한 자세라는 말을 자주 듣는다.",
    "Neck cracking, frequent fatigue, and headaches.": "목을 이리저리 돌렸을 때 우두둑하는 소리가 나거나 쉽게 피곤하고 두통이 자주 생긴다.",
    "Bad sleeping habit.": "잠 버릇이 나쁘다.",
    "Over 8 hours of computer use daily.": "컴퓨터를 하루에 8시간 이상 하는 편이다.",
    "Frequent neck and shoulder stiffness.": "목과 어깨가 자주 결린다.",
    "Unrefreshing sleep.": "자고 일어났을 때 개운하지 않다.",
    "Eye strain and chronic headaches.": "눈이 쉽게 뻐근하고 만성 두통에 시달린다."
}

# 영어 key 리스트
FACTORS = list(FACTOR_KR.keys())

# 영어 key → DB column 이름 매핑
FACTOR_COLUMNS = {
    "Cold, tingling hands and feet, and poor posture.": "cold_tingling_posture",
    "Neck cracking, frequent fatigue, and headaches.": "neck_cracking_fatigue_headaches",
    "Bad sleeping habit.": "bad_sleeping_habit",
    "Over 8 hours of computer use daily.": "computer_over_8h",
    "Frequent neck and shoulder stiffness.": "neck_shoulder_stiffness",
    "Unrefreshing sleep.": "unrefreshing_sleep",
    "Eye strain and chronic headaches.": "eye_strain_headaches"
}

# =======================
# 체크리스트 저장
# =======================
def save_checklist(user_id, factors, all_factors, is_guest=False, guest_session_id=None):
    conn = get_connection()
    cur = conn.cursor()
    try:
        kst_now = datetime.utcnow() + timedelta(hours=9)
        data = {FACTOR_COLUMNS[f]: (1 if f in factors else 0) for f in all_factors}
        data["updated_at"] = kst_now

        if is_guest:
            # guest session 존재 여부 확인
            cur.execute("SELECT guest_user_id FROM guest_checklist WHERE guest_session_id=?", (guest_session_id,))
            row = cur.fetchone()
            if row:
                # UPDATE
                set_clause = ", ".join([f"{col}=?" for col in data])
                values = list(data.values()) + [guest_session_id]
                cur.execute(f"UPDATE guest_checklist SET {set_clause} WHERE guest_session_id=?", values)
            else:
                # INSERT
                columns = ", ".join(list(data.keys()) + ["guest_session_id"])
                placeholders = ", ".join(["?"] * (len(data)+1))
                values = list(data.values()) + [guest_session_id]
                cur.execute(f"INSERT INTO guest_checklist ({columns}) VALUES ({placeholders})", values)
        else:
            # 회원 체크리스트는 기존 로직 유지
            cur.execute("SELECT user_id FROM checklist WHERE user_id=?", (user_id,))
            exists = cur.fetchone()
            if exists:
                set_clause = ", ".join([f"{col}=?" for col in data])
                values = list(data.values()) + [user_id]
                cur.execute(f"UPDATE checklist SET {set_clause} WHERE user_id=?", values)
            else:
                columns = ", ".join(["user_id"] + list(data.keys()))
                placeholders = ", ".join(["?"] * (len(data)+1))
                values = [user_id] + list(data.values())
                cur.execute(f"INSERT INTO checklist ({columns}) VALUES ({placeholders})", values)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Error saving checklist:", e)
    finally:
        conn.close()

# 체크리스트 불러오기 수정
def get_user_checklist(user_id, is_guest=False, guest_session_id=None):
    conn = get_connection()
    cur = conn.cursor()
    if is_guest:
        cur.execute("SELECT * FROM guest_checklist WHERE guest_session_id=?", (guest_session_id,))
    else:
        cur.execute("SELECT * FROM checklist WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return {}

    return {f: row[FACTOR_COLUMNS[f]] for f in FACTORS}

# =======================
# 체크리스트 페이지
# =======================
@user_service_bp.route("/checklist", methods=["GET", "POST"])
def checklist():
    if "user_id" not in session:
        return redirect(url_for("signup.register"))

    user_id_raw = session["user_id"]
    is_guest = isinstance(user_id_raw, str) and user_id_raw.startswith("guest")

    # guest_session_id는 user_id 자체를 사용
    guest_session_id = user_id_raw if is_guest else None

    # guest라면 DB에 row 없으면 생성
    if is_guest:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT guest_user_id FROM guest_checklist WHERE guest_session_id=?", (guest_session_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO guest_checklist (guest_session_id, updated_at) VALUES (?, ?)",
                        (guest_session_id, datetime.utcnow() + timedelta(hours=9)))
            conn.commit()
        conn.close()

    user_id = user_id_raw if not is_guest else None

    if request.method == "POST":
        checked_factors = request.form.getlist("factors")
        checked_factors = [f for f in checked_factors if f in FACTORS]

        save_checklist(
            user_id=user_id,
            factors=checked_factors,
            all_factors=FACTORS,
            is_guest=is_guest,
            guest_session_id=guest_session_id
        )
        return redirect(url_for("user_service.checklist"))

    saved_data = get_user_checklist(
        user_id=user_id,
        is_guest=is_guest,
        guest_session_id=guest_session_id
    )

    # 통계 계산 (회원 + guest)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM checklist")
    total_users = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM guest_checklist")
    total_guests = cur.fetchone()[0] or 0
    total_all = total_users + total_guests or 1

    stats = {}
    for factor in FACTORS:
        col = FACTOR_COLUMNS[factor]
        cur.execute(f"SELECT COUNT(*) FROM checklist WHERE {col}=1")
        users_count = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM guest_checklist WHERE {col}=1")
        guests_count = cur.fetchone()[0]
        stats[factor] = round((users_count + guests_count)/total_all*100,1)
    conn.close()

    return render_template(
        "user_service/checklist.html",
        saved_data=saved_data,
        FACTORS=FACTORS,
        FACTOR_KR=FACTOR_KR,
        stats=stats
    )



# =======================
# 나머지 페이지
# =======================
@user_service_bp.route("/stretching")
def stretching():
    return render_template("user_service/stretching.html") 

@user_service_bp.route("/user_service")
def user_service():
    return render_template("user_service/user_service.html")

@user_service_bp.route("/statistics")
def statistics():
    conn = get_connection()
    cur = conn.cursor()
    # 통계 계산 (회원 + guest)
    cur.execute("SELECT COUNT(*) FROM checklist")
    total_users = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM guest_checklist")
    total_guests = cur.fetchone()[0] or 0
    total_all = total_users + total_guests or 1

    stats = {}
    for factor in FACTORS:
        col = FACTOR_COLUMNS[factor]
        cur.execute(f"SELECT COUNT(*) FROM checklist WHERE {col}=1")
        users_count = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM guest_checklist WHERE {col}=1")
        guests_count = cur.fetchone()[0]
        stats[factor] = round((users_count + guests_count)/total_all*100,1)
    conn.close()

    return render_template(
        "user_service/statistics.html",
        FACTORS=FACTORS,
        FACTOR_KR=FACTOR_KR,
        stats=stats
    )
