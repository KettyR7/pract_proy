"""
Detector de pose humana para ejercicios: Sentadilla y Peso Muerto
Archivo: detector_pose_sentadilla_peso_muerto.py
Resumen:
- Usa MediaPipe Pose y OpenCV para capturar vídeo de la webcam
- Calcula ángulos clave (rodilla, cadera, espalda) y detecta repeticiones y errores comunes
- Muestra retroalimentación en pantalla y guarda un registro (CSV opcional)

Requisitos:
pip install opencv-python mediapipe pandas

Ejecución:
python detector_pose_sentadilla_peso_muerto.py

Controles:
- Presiona 'q' para salir
- Presiona 's' para alternar guardado de métricas en CSV

"""

import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from collections import deque

# ---------- Utilidades ----------
# Importación del módulo de utilidades de dibujo de mediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# calculo el ángulo formado por tres puntos en el plano, especificamente en el angulo ABC
def calc_angle(a, b, c):
    """Calcula el ángulo ABC (en grados) donde a,b,c son (x,y) puntos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    # proteger contra división por cero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # limitar dominio numérico
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # calcular el ángulo en grados
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# convierte un landmark(punto clave detectado por MediaPipe) en una coordenada de pixel dentro de la imagen original
def landmark_to_point(landmark, image_width, image_height):
    return (int(landmark.x * image_width), int(landmark.y * image_height))

# ---------- Detectar y contar repeticiones de ejercicios ----------
class ExerciseDetector:
    def __init__(self):
        # estados para contar reps
        self.squat_count = 0
        self.squat_stage = None  # 'down' o 'up'
        self.deadlift_count = 0
        self.deadlift_stage = None
        # buffers para suavizar ángulos
        self.knee_angles_buffer = deque(maxlen=5)
        self.hip_angles_buffer = deque(maxlen=5)
        self.back_angles_buffer = deque(maxlen=5)
        # registro de métricas
        self.log = []
        self.save_csv = False

# función útil para suavizar valores númericos, cuando se trabaja con datos que puedan tener ruido como la visión por computadora
# esto asegura que solo se mantengan los últimos N valores, calcula el promedio de los valores en buffer dando una estimación más estable
    def smooth(self, buffer, value):
        buffer.append(value)
        return float(np.mean(buffer))

#obtener las coordenadas de ciertos landmarks, permite detectar asimetrías o errores de postura
    def process(self, landmarks, img_w, img_h):
        # Obtener puntos claves (usar ambos lados y promediar cuando convenga)
        try:
            # Hombro, cadera, rodilla, tobillo: para ambos lados
            ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            le = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
            re = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        except Exception:
            return None

        # Convertir a coordenadas de píxeles
        L_sh = landmark_to_point(ls, img_w, img_h)
        R_sh = landmark_to_point(rs, img_w, img_h)
        L_hip = landmark_to_point(lh, img_w, img_h)
        R_hip = landmark_to_point(rh, img_w, img_h)
        L_knee = landmark_to_point(lk, img_w, img_h)
        R_knee = landmark_to_point(rk, img_w, img_h)
        L_ank = landmark_to_point(la, img_w, img_h)
        R_ank = landmark_to_point(ra, img_w, img_h)

        # Ángulos: rodilla (hip - knee - ankle), cadera (shoulder - hip - knee), espalda (shoulder - hip - knee?)
        # Para la espalda mejor usar: hombro - cadera - rodilla (ángulo torso-hip)
        left_knee_angle = calc_angle(L_hip, L_knee, L_ank)
        right_knee_angle = calc_angle(R_hip, R_knee, R_ank)
        knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        left_hip_angle = calc_angle(L_sh, L_hip, L_knee)
        right_hip_angle = calc_angle(R_sh, R_hip, R_knee)
        hip_angle = (left_hip_angle + right_hip_angle) / 2.0

        # Back (torso) angle: use shoulder-hip-ankle verticality approximate
        # Alternativa: ángulo entre vector hombro->cadera y cadera->rodilla ya calculado como hip_angle
        back_angle = hip_angle  # simplificación razonable para feedback

        # Suavizar
        knee_angle_s = self.smooth(self.knee_angles_buffer, knee_angle)
        hip_angle_s = self.smooth(self.hip_angles_buffer, hip_angle)
        back_angle_s = self.smooth(self.back_angles_buffer, back_angle)

        # Detección de sentadilla (squat)
        squat_feedback = ''
        # Umbrales: rodilla < 90-100 grados suele indicar bajada profunda; ajustar según necesidad
        squat_down_threshold = 100  # cuando el ángulo de la rodilla cae por debajo de este valor se considera "down"
        squat_up_threshold = 160

# si el ángulo de la rodilla es menor que el umbral de bajada se considera que la persona está en la fase baja de la sentadilla
        if knee_angle_s < squat_down_threshold:
            # posición baja
            if self.squat_stage != 'down':
                self.squat_stage = 'down'
# si el ángulo supera el umbral de subida y el estado anterior era down, se considera que la persona ha completado una repetición                
        if knee_angle_s > squat_up_threshold and self.squat_stage == 'down':
            # se actualiza el estado
            self.squat_stage = 'up'
            self.squat_count += 1 # se incrementa 1 repetición y se genera un mensaje
            squat_feedback = f'Rep sentadilla contabilizada: {self.squat_count}'

        # Feedback de técnica para sentadilla
        # Ejemplos: rodillas hacia adelante excesivo (no comprobamos profundidad del talón), espalda inclinada
        if knee_angle_s < 120 and back_angle_s < 120:
            squat_feedback += ' | Mantén la espalda recta'
        if knee_angle_s > 170:
            squat_feedback += ' | No estás bajando suficiente'

        # Detección de peso muerto (deadlift)
        deadlift_feedback = ''
        # Para deadlift: tronco se inclina (hip_angle pequeño) y en la subida vuelve vertical
        deadlift_down_threshold = 70   # cadera doblada por debajo indica bisagra
        deadlift_up_threshold = 160

        if hip_angle_s < deadlift_down_threshold:
            if self.deadlift_stage != 'down':
                self.deadlift_stage = 'down'
        if hip_angle_s > deadlift_up_threshold and self.deadlift_stage == 'down':
            self.deadlift_stage = 'up'
            self.deadlift_count += 1
            deadlift_feedback = f'Rep peso muerto contabilizada: {self.deadlift_count}'

        # Feedback técnica peso muerto
        if back_angle_s < 40:
            deadlift_feedback += ' | Cuidado: espalda muy redondeada'
        if knee_angle_s < 140:
            deadlift_feedback += ' | Rodillas demasiado dobladas, intenta más bisagra de cadera'

        # Registrar métricas
        timestamp = time.time()
        self.log.append({
            'time': timestamp,
            'knee_angle': float(knee_angle_s),
            'hip_angle': float(hip_angle_s),
            'back_angle': float(back_angle_s),
            'squat_count': int(self.squat_count),
            'deadlift_count': int(self.deadlift_count)
        })

        return {
            'knee_angle': knee_angle_s,
            'hip_angle': hip_angle_s,
            'back_angle': back_angle_s,
            'squat_count': self.squat_count,
            'deadlift_count': self.deadlift_count,
            'squat_feedback': squat_feedback.strip(' |'),
            'deadlift_feedback': deadlift_feedback.strip(' |'),
            'points': {
                'L_sh': L_sh, 'R_sh': R_sh, 'L_hip': L_hip, 'R_hip': R_hip,
                'L_knee': L_knee, 'R_knee': R_knee, 'L_ank': L_ank, 'R_ank': R_ank
            }
        }

    def save_log_csv(self, filename='pose_log.csv'):
        if len(self.log) == 0:
            return False
        df = pd.DataFrame(self.log)
        df.to_csv(filename, index=False)
        return True

# ---------- Aplicación principal ----------

def main():
    cap = cv2.VideoCapture(0)
    detector = ExerciseDetector()

    # Parámetros MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('No se pudo leer la cámara')
                break

            # Voltear horizontal para efecto espejo
            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape

            # Convertir a RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Procesar pose
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                )

                # Procesar lógica del ejercicio
                out = detector.process(results.pose_landmarks.landmark, image_width, image_height)
                if out is not None:
                    # Mostrar ángulos y conteos en la imagen
                    cv2.putText(image_bgr, f"Knee: {out['knee_angle']:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(image_bgr, f"Hip: {out['hip_angle']:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(image_bgr, f"Squats: {out['squat_count']}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.putText(image_bgr, f"Deadlifts: {out['deadlift_count']}", (10,135), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                    # Feedback
                    if out['squat_feedback']:
                        cv2.putText(image_bgr, out['squat_feedback'], (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2)
                    if out['deadlift_feedback']:
                        cv2.putText(image_bgr, out['deadlift_feedback'], (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2)

                    # Dibujar puntos personalizados (opcionales)
                    for name, pt in out['points'].items():
                        cv2.circle(image_bgr, pt, 4, (0,0,255), -1)

            # Mostrar FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(image_bgr, f"FPS: {int(fps)}", (image_width-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow('Pose Detector - Sentadilla / Peso Muerto', image_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                detector.save_csv = not detector.save_csv
                print('Guardar CSV:', detector.save_csv)

        # Al salir, opcionalmente guardar log
        if detector.save_csv:
            saved = detector.save_log_csv()
            if saved:
                print('Registro guardado en pose_log.csv')
            else:
                print('No hay datos para guardar')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
