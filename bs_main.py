import cv2 as cv
import BackgroundSubtraction
import Utils

# Cargar el video
video_path = 'video/Walking 1.54138969.mp4'
cap = cv.VideoCapture(video_path)

output_video_MOG = Utils.create_output_video("bs_output/bs_MOG_Walking 1.54138969.mp4")
output_video_MOG2 = Utils.create_output_video("bs_output/bs_MOG2_Walking 1.54138969.mp4")
output_video_GMG = Utils.create_output_video("bs_output/bs_GMG_Walking 1.54138969.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    fgMask_MOG = BackgroundSubtraction.bs_MOG(frame) #MEZCLA GUASIANA
    fgMask_MOG2 = BackgroundSubtraction.bs_MOG2(frame) # mezcla gausiana mejorando la iluminación, detecta las sombras
    fgMask_GMG = BackgroundSubtraction.bs_GMG(frame)
    #fgMask_filtered = cv.medianBlur(fgMask, ksize=5)
    #fgMask_filtered = cv.morphologyEx(fgMask_filtered, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    # Mostrar la imagen
    cv.imshow("Frame", frame)
    cv.imshow("fgMask_MOG", fgMask_MOG)
    cv.imshow("fgMask_MOG2", fgMask_MOG2)
    cv.imshow("fgMask_GMG", fgMask_GMG)

    output_video_MOG.write(fgMask_MOG)
    output_video_MOG2.write(fgMask_MOG2)
    output_video_GMG.write(fgMask_GMG)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    cv.waitKey(1)

output_video_MOG.release()
output_video_MOG2.release()
output_video_GMG.release()
cap.release()
cv.destroyAllWindows()