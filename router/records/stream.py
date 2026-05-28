from flask import Blueprint, Response
import cv2

stream_bp = Blueprint("stream", __name__)

camera = None


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # 웹캠 사용
    return camera

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@stream_bp.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
