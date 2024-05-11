import Utils
import code
import numpy as np
import cv2

file_path = 'poses/2D/Walking 1.54138969.cdf'

#Esta línea genera los fichero .npy a partir de cdf
#code.convert_cdf_to_matlab("Walking 1", "data/cdf/")
point2D_h = np.load("data/cdf/Walking 1_h36m_points2D_h.npy")
point2D = np.load("data/cdf/Walking 1_h36m_points2D.npy")
point3D = np.load("data/cdf/Walking 1_h36m_points3D.npy")

print("point2D_h.shape=\n", point2D_h.shape)
print("point2D_h.shape=\n", point2D_h[0])
print("point2D.shape=\n", point2D.shape)
print("point3D.shape=\n", point3D[0])

c1_params = code.get_cam_params(cam_id = "54138969", subject = 1)
print("c1_params=\n", c1_params)

#Cameras
ks, dcs, rs, ts = code.load_cams(1)
print("len(ks)=\n", len(ks))
project_cams = code.krts2proj_cameras(ks, rs, ts)

points2D = code.project_points_proj_cameras(point3D, project_cams)
print("points2D=\n", points2D.shape)

# Cargar el video
video_path = 'video/Walking 1.54138969.mp4'
cap = cv2.VideoCapture(video_path)

# Obtener la información del video (dimensiones, FPS, etc.)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Crear el objeto VideoWriter para guardar el video con los puntos dibujados
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Iterar sobre cada frame del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dibujar los puntos 2D en el frame
    for point in points2D:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Dibuja un círculo verde
    
    # Escribir el frame modificado en el video de salida
    out.write(frame)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()

