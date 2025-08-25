"""
Detector de pose humana para ejercicios (Sentadilla y Peso Muerto) - Interfaz Web con Streamlit

Requisitos:
pip install opencv-python mediapipe pandas streamlit

Ejecución:
streamlit run detector_pose_web.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def landmark_to_point(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

class ExerciseDetector:
    def __init__(self):
        self.squat_count = 0
        self.squat_stage = None
        self.deadlift_count = 0
        self.deadlift_stage = None
        self.knee_angles_buffer = deque(maxlen=5)
        self.hip_angles_buffer = deque(maxlen=5)
        self.back_angles_buffer = deque(maxlen=5)
        self.log = []

    def smooth(self, buffer, value):
        buffer.append(value)
        return float(np.mean(buffer))

    def process(self, landmarks, w, h):
        try:
            ls, rs = landmarks[11], landmarks[12]
            lh, rh = landmarks[23], landmarks[24]
            lk, rk = landmarks[25], landmarks[26]
            la, ra = landmarks[27], landmarks[28]
        except:
            return None
        L_sh, R_sh = landmark_to_point(ls, w, h), landmark_to_point(rs, w, h)
        L_hip, R_hip = landmark_to_point(lh, w, h), landmark_to_point(rh, w, h)
        L_knee, R_knee = landmark_to_point(lk, w, h), landmark_to_point(rk, w, h)
        L_ank, R_ank = landmark_to_point(la, w, h), landmark_to_point(ra, w, h)
        knee_angle = (calc_angle(L_hip, L_knee, L_ank) + calc_angle(R_hip, R_knee, R_ank)) / 2
        hip_angle = (calc_angle(L_sh, L_hip, L_knee) + calc_angle(R_sh, R_hip, R_knee)) / 2
        back_angle = hip_angle
        knee_angle_s = self.smooth(self.knee_angles_buffer, knee_angle)
        hip_angle_s = self.smooth(self.hip_angles_buffer, hip_angle)
        back_angle_s = self.smooth(self.back_angles_buffer, back_angle)
        squat_feedback, deadlift_feedback = '', ''
        if knee_angle_s < 100:
            if self.squat_stage != 'down':
                self.squat_stage = 'down'
        if knee_angle_s > 160 and self.squat_stage == 'down':
            self.squat_stage = 'up'
            self.squat_count += 1
            squat_feedback = f"Sentadilla #{self.squat_count}"
        if knee_angle_s < 120 and back_angle_s < 120:
            squat_feedback += ' | Espalda recta'
        if hip_angle_s < 70:
            if self.deadlift_stage != 'down':
                self.deadlift_stage = 'down'
        if hip_angle_s > 160 and self.deadlift_stage == 'down':
            self.deadlift_stage = 'up'
            self.deadlift_count += 1
            deadlift_feedback = f"Peso muerto #{self.deadlift_count}"
        if back_angle_s < 40:
            deadlift_feedback += ' | Evita redondear espalda'
        self.log.append({
            'time': time.time(),
            'knee_angle': knee_angle_s,
            'hip_angle': hip_angle_s,
            'back_angle': back_angle_s,
            'squat_count': self.squat_count,
            'deadlift_count': self.deadlift_count
        })
        return {
            'knee_angle': knee_angle_s,
            'hip_angle': hip_angle_s,
            'back_angle': back_angle_s,
            'squat_count': self.squat_count,
            'deadlift_count': self.deadlift_count,
            'squat_feedback': squat_feedback.strip(' |'),
            'deadlift_feedback': deadlift_feedback.strip(' |')
        }

def main():
    st.title("Detector de Pose - Sentadilla y Peso Muerto")
    run = st.checkbox("Iniciar cámara")
    FRAME_WINDOW = st.image([])
    detector = ExerciseDetector()
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("No se detecta cámara")
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out = detector.process(results.pose_landmarks.landmark, w, h)
                if out:
                    cv2.putText(frame, f"Squats: {out['squat_count']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(frame, f"Deadlifts: {out['deadlift_count']}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

if __name__ == "__main__":
    main()
