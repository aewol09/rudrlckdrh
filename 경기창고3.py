import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

st.title("실시간 CVA 거북목 측정")

# 웹캠 캡처 버튼
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])  # Streamlit에서 이미지 표시할 공간

def calculate_cva_angle(shoulder, ear):
    dx = ear[0] - shoulder[0]
    dy = ear[1] - shoulder[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(angle_rad * 180.0 / math.pi)
    return 90 - angle_deg

if run:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            h, w, _ = frame.shape

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 오른쪽 좌표
                shoulder_r = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
                ear_r = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))
                # 왼쪽 좌표
                shoulder_l = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                              int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
                ear_l = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))

                # CVA 평균
                cva_r = calculate_cva_angle(shoulder_r, ear_r)
                cva_l = calculate_cva_angle(shoulder_l, ear_l)
                cva_avg = (cva_r + cva_l) / 2

                # 화면 표시
                cv2.putText(frame, f"CVA Avg: {cva_avg:.1f} deg", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

             # 판정 기준 30도로 변경
                if cva_avg >= 30:
                    color = (0, 00, 255) 
                    status = "Forward Head"
                    text_pos = (30, 150) 
                else:
                    color = (0, 255, 0)
                    status = "Normal"
                    text_pos = (30, 150)  

                cv2.putText(frame, status, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.line(frame, shoulder_r, ear_r, color, 3)
                cv2.line(frame, shoulder_l, ear_l, color, 3)


                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
