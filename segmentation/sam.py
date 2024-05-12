import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor


def load_model_SAM(model_type, checkpoint_path, device=torch.device('cpu')):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_predictor = SamPredictor(sam)
    return mask_predictor

def borrar_fondo(path, mask_predictor):
    image_bgr = cv2.imread(path)

    return borrar_fondo_imagen(image_bgr, mask_predictor)


def obtener_mascara(imagen, mask_predictor, box=None):

    image_rgb = cv2.cvtColor(imagen.copy(), cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    
    if box is None:
        box = np.array([0,0,image_rgb.shape[1],image_rgb.shape[0]])
    
    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    #Save masks
    for i, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)
        #cv2.imwrite(IMAGE_PATH.replace('.jpg', f'_mask_{i}.jpg'), mask)

    # Get the best mask
    object_mask = masks[scores.argmax()]

    return object_mask

def borrar_fondo_imagen(imagen, mask_predictor, box=None):

    object_mask = obtener_mascara(imagen, mask_predictor, box)

    object_mask_8u = (object_mask * 255).astype(np.uint8)
    cv2.imshow('mascara', object_mask_8u)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Prepare an all-black background with the same size as the original image.
    image_rgb = cv2.cvtColor(imagen.copy(), cv2.COLOR_BGR2RGB)
    background = np.zeros_like(image_rgb)

    # Use the object mask to copy the object onto the all-black background.
    object_on_black = np.where(object_mask[:,:,None], image_rgb, background)

    # Convert the result back to BGR for saving or displaying with OpenCV.
    object_on_black_bgr = cv2.cvtColor(object_on_black, cv2.COLOR_RGB2BGR)

    return object_on_black_bgr

if __name__ == "__main__":

    device = torch.device('cpu')

    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "modelos\sam_vit_h_4b8939.pth"

    path_video = 'video\Walking 1.54138969.mp4'
    box = np.array([442.46, 270.22, 548.82, 639.57])

    cap = cv2.VideoCapture(path_video)
    ret, frame = cap.read()
    cap.release()

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)

    mask_predictor = SamPredictor(sam)
    
    mask = obtener_mascara(frame, mask_predictor, box)
    mask_8u = (mask * 255).astype(np.uint8)

    cv2.imshow('mask', mask_8u)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


