import cv2 as cv

# Inicializar el objeto de substracción de fondo
backSub_MOG = cv.bgsegm.createBackgroundSubtractorMOG()
backSub_MOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)

#Estimación estadística de la imagen de fondo con la segmentación bayesiana por pixel
#120 primeros fotogramas por defecto para el modelado
#Gaussian Mixture-based Background/Foreground Segmentation
backSub_GMG = cv.bgsegm.createBackgroundSubtractorGMG()

#K-Nearest Neighbors
#backSub_KNN = cv.createBackgroundSubtractorKNN()

#Mezcla de Guasianas --> Apuntes: SubstraccionFondoII2023.pdf
def bs_MOG(frame):
    # Aplicar la substracción de fondo al fotograma
    #MÁscara del primer plano
    fgMask = backSub_MOG.apply(frame)
    return fgMask

def bs_MOG2(frame):
    # Aplicar la substracción de fondo al fotograma
    fgMask = backSub_MOG2.apply(frame)
    return fgMask

def bs_GMG(frame):
    # Aplicar la substracción de fondo al fotograma
    fgMask = backSub_GMG.apply(frame)
    return fgMask

#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#Diferencia entre imágenes consecutivas
def bs_diff_frames(actual_gray_frame, prev_gray_frame):
    # Calcular la diferencia absoluta entre el fotograma actual y el anterior
    diff = cv.absdiff(actual_gray_frame, prev_gray_frame)
    # Aplicar umbralización para obtener la máscara de movimiento
    _, fgMask = cv.threshold(diff, 127, 255, cv.THRESH_BINARY)
    #fgMask = cv.adaptiveThreshold(diff, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return fgMask