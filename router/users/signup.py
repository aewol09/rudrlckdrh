from flask import Blueprint, render_template, request, url_for, flash, redirect, session
from db import add_user
from werkzeug.security import generate_password_hash
from modules.utils import generate_hash

signup_bp = Blueprint('signup', __name__)

@signup_bp.route("/signup", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        password_confirm = request.form["password_confirm"]

        if not username or not password:
            flash("아이디와 비밀번호를 입력해주세요.")
            return redirect(url_for("signup.register"))

        if password != password_confirm:
            flash("비밀번호가 일치하지 않습니다.")
            return redirect(url_for("signup.register"))

        hashed_pw = generate_password_hash(password)

        if add_user(username, password):
            flash("회원가입 완료! 로그인 해주세요.")
            return redirect(url_for("login.login"))
        else:
            flash("이미 존재하는 사용자입니다.")
            return redirect(url_for("signup.register"))

    return render_template("signup/signup.html")


# 게스트 로그인
@signup_bp.route("/", methods=["POST"])
def guest_login():
    # 세션에 guest 사용자 설정
    session["user_id"] = "guest" + generate_hash(32)
    session["username"] = "게스트"
    return redirect(url_for("index"))  # 게스트용 홈 페이지