from flask import Blueprint, render_template, request, url_for, flash, redirect, session
from werkzeug.security import check_password_hash
from db import verify_user, get_user_by_username

login_bp = Blueprint('login', __name__)

@login_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("아이디와 비밀번호를 입력해주세요.")
            return redirect(url_for("login.login"))

        if verify_user(username, password):
            user = get_user_by_username(username)
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("로그인 성공!")
            return redirect(url_for("index"))  # 메인 페이지
        else:
            flash("아이디 또는 비밀번호가 잘못되었습니다.")
            return redirect(url_for("login.login"))

    return render_template("login/login.html")
