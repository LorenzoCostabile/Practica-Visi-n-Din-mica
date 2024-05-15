import cv2
import numpy as np
import mediapipe as mp #holistic module

def draw_rectangle(image, points):
    # Convertir puntos de flotante a entero
    points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in points]

    # Ordenar los puntos para formar un rectángulo
    sorted_points = sorted(points, key=lambda x: x[1])  # Ordenar por coordenada y

    # Dibujar rectángulo
    cv2.rectangle(image, sorted_points[0], sorted_points[3], (0, 255, 0), 2)

    # Calcular el centro del rectángulo
    center_x = (sorted_points[0][0] + sorted_points[3][0]) // 2
    center_y = (sorted_points[0][1] + sorted_points[3][1]) // 2

    return image, (center_x, center_y)

def calculate_region_center(points):
    # Calcular los límites de la región
    xmin = min(points, key=lambda x: x[0])[0]
    ymin = min(points, key=lambda x: x[1])[1]
    xmax = max(points, key=lambda x: x[0])[0]
    ymax = max(points, key=lambda x: x[1])[1]

    # Calcular el centro de la región
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    return center_x, center_y

# Inicializar el objeto de detección de pose de Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Capturar video desde la cámara
video_path = 'video/Walking 1.54138969.mp4'
#video_path = 'bs_output/bs_MOG2_Walking 1.54138969.mp4'
cap = cv2.VideoCapture(video_path)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realizar la detección de la pose
        results = holistic.process(rgb_frame)
        # results contain face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        #face_landmarks
        #FACE_CONNECTIONS has been replace by FACEMESH_TESSELATION
        #mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        #                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=2), # key point
        #                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=2), ) # connection

        #right_hand_landmarks
        #mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                          mp_drawing.DrawingSpec(color=(180, 110, 10), thickness=1, circle_radius=2), # key point
        #                          mp_drawing.DrawingSpec(color=(180, 256, 121), thickness=1, circle_radius=2), ) # connection

        #left_hand_landmarks
        #mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                          mp_drawing.DrawingSpec(color=(180, 110, 10), thickness=1, circle_radius=2), # key point
        #                          mp_drawing.DrawingSpec(color=(180, 256, 121), thickness=1, circle_radius=2), ) # connection

        #pose_landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=2), # key point
                                  mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=2), ) # connection

        if results.pose_landmarks:
            # Obtener los landmarks de la pose
            pose_landmarks = results.pose_landmarks.landmark

            # Indices de los puntos correspondientes al torso
            shoulder_left = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            hip_left = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            hip_right = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            # Crear una copia de la imagen original para dibujar el rectángulo
            #image_with_rectangle = frame.copy()

            # Dibujar el rectángulo en la imagen
            #frame, center = draw_rectangle(frame, [shoulder_left, shoulder_right, hip_left, hip_right])

            torso_points = [(shoulder_left.x * frame.shape[1], shoulder_left.y * frame.shape[0]),
                (shoulder_right.x * frame.shape[1], shoulder_right.y * frame.shape[0]),
                (hip_left.x * frame.shape[1], hip_left.y * frame.shape[0]),
                (hip_right.x * frame.shape[1], hip_right.y * frame.shape[0])]
            
            centro_x, centro_y = calculate_region_center(torso_points)

            cv2.circle(frame, (int(centro_x), int(centro_y)), 5, (0, 0, 255), -1)
        
        cv2.imshow('Mediapipe detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

