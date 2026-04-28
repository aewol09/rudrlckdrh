from flask import Flask, flash, redirect, render_template, Blueprint, session, request, jsonify, url_for, Response
import sqlite3, cv2
from dotenv import load_dotenv
import os
load_dotenv()

from db import add_user, verify_user, get_user_by_username, get_connection, init_db
from router.users.login import login_bp
from router.users.signup import signup_bp
from router.records.save_photo import save_photo_bp, images_bp
from router.records.records import records_bp
from router.records.stream import stream_bp
from router.records.videos.full import full_videos_bp
from router.records.videos.highlight import highlight_bp
from router.footerlinks.footer import footer_bp
from router.users.user_service import user_service_bp

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']
app.register_blueprint(signup_bp)
app.register_blueprint(login_bp)
app.register_blueprint(save_photo_bp)
app.register_blueprint(records_bp)
app.register_blueprint(stream_bp)
app.register_blueprint(images_bp)
app.register_blueprint(full_videos_bp)
app.register_blueprint(highlight_bp)
app.register_blueprint(footer_bp)
app.register_blueprint(user_service_bp)

init_db()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("index"))

@app.route("/cva")
def cva():
    return render_template("cva.html")

if __name__ == "__main__":
    app.run(os.environ['HOST_IP'], os.environ['HOST_PORT']) 