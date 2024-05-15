import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import Utils

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

trajectory_len = 0
detect_interval = 5
trajectories = []
frame_idx = 0

known_x, known_y = [], []
estimated_x, estimated_y = [], []

video_path = 'video/Walking 1.54138969.mp4'
cap = cv.VideoCapture(video_path)

point2D = np.load("data/cdf/Walking 1_h36m_points2D.npy")

out = Utils.create_output_video("seguimiento_flujo_optico.mp4")

point_2d = point2D[:2, :]
print("point_2d.shape=\n", point_2d.shape)
chunk_points = np.array_split(point_2d, 111232 // 32, axis=1)
index_frame = 0
print("len(chunk_points)=\n", len(chunk_points))

# Take first frame and find corners in it
ret, old_frame = cap.read()
gray_prev = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame_points = chunk_points[index_frame]
prev_points = np.array([frame_points[0], frame_points[1]]).T.reshape(-1, 1, 2).astype(np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

index_frame += 1

# Iterar sobre cada frame del video
while cap.isOpened() and index_frame < len(chunk_points):
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    next_points, status, _ = cv.calcOpticalFlowPyrLK(gray_prev, gray, prev_points, None, **lk_params)

    # Dibujar los puntos 2D en el frame
    frame_points = chunk_points[index_frame]
    for j in range(0, len(frame_points[0])):
        x, y = int(frame_points[0][j]), int(frame_points[1][j])
        cv.circle(frame, (x, y), 5, (0, 255, 0), -1)  # verde

        known_x.append(x)
        known_y.append(y)

    # Filtrar solo los puntos con buen seguimiento
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # Dibujar los puntos en el fotograma
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2)
        cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        #Línea verde entre el punto antiguo y el nuevo
        cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        #img = cv.add(frame, mask)

        estimated_x.append(int(a))
        estimated_y.append(int(b))

    # Actualizar los puntos para el siguiente fotograma
    prev_points = np.array([frame_points[0], frame_points[1]]).T.reshape(-1, 1, 2).astype(np.float32)
    gray_prev = gray.copy()
    # Escribir el frame modificado en el video de salida
    out.write(frame)
    #cv.imshow('Frame', img)
    cv.imshow('Frame', frame)

    index_frame += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv.destroyAllWindows()
