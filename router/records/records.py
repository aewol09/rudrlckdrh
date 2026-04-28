from flask import Blueprint, request, jsonify, session, render_template
import os
from db import get_connection

records_bp = Blueprint("records", __name__)  # 이름 고유하게

@records_bp.route("/records")
def records():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM full_videos ORDER BY timestamp DESC").fetchall()
    conn.close()
    return render_template("records/records.html", videos=rows)
