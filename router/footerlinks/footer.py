from flask import Blueprint, render_template

footer_bp = Blueprint('footer', __name__)

@footer_bp.route("/feedback")
def feedback():
    return render_template("footer-links/feedback.html")

@footer_bp.route("/FAQ")
def FAQ():
    return render_template("footer-links/FAQ.html")

@footer_bp.route("/security")
def security():
    return render_template("footer-links/security.html")

@footer_bp.route("/TAC")
def TAC():
    return render_template("footer-links/TAC.html")