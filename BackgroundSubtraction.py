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