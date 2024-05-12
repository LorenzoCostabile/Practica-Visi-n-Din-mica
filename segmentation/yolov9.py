from ultralytics import YOLO
import cv2
import numpy as np

def load_model(model_path, clases = [0]):
    # Carga un modelo YOLOv9 desde un archivo de pesos preentrenado
    model = YOLO(model_path)
    model.classes = clases  # Configura las clases si es necesario
    return model

def process_image(model, numpy_image):
    # Carga una imagen y realiza la inferencia
    results = model.predict(source=numpy_image, imgsz=640, conf=0.6, save=False, classes=0)
    return results

def display_results(results):
    # Muestra los resultados de la inferencia
    for result in results:
        result.show()

def extraer_bbox(results):
    bbox = results[0].boxes.xyxy[0].cpu().numpy()
    return bbox

def obtener_mascaras(imagen, model):
    results = model.predict(source=imagen, imgsz=640, conf=0.6, save=False, classes=0)

    # Extraer todas las máscaras y sus respectivos scores
    all_masks = results[0].masks
    all_scores = results[0].boxes.conf

    # Identificar la máscara con el mayor score
    best_score_index = np.argmax(all_scores)
    best_mask = all_masks[best_score_index]

    # Convertir la máscara a un formato adecuado para cv2.fillPoly
    mask = np.array(best_mask.xy[0], dtype=np.int32)

    # Crear una imagen de fondo para la máscara
    mask_bg = np.zeros(imagen.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_bg, [mask], 1)

    return mask_bg

if __name__ == "__main__":
    # Ejemplo de cómo usar el módulo
    model = load_model('modelos\yolov9e-seg.pt')

    path_video = 'video\Walking 1.54138969.mp4'

    cap = cv2.VideoCapture(path_video)
    ret, frame = cap.read()
    cap.release()

    mask = obtener_mascaras(frame, model)

    # Convertir a 255
    mask = mask * 255

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


