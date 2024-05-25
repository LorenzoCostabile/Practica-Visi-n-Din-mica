import numpy as np
import cv2 as cv

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

def create_video_capture(path_file_video):
    return cv.VideoCapture(path_file_video)

def create_output_video(output_path_file_video):
    return cv.VideoWriter(output_path_file_video, fourcc, 33.0, (1000, 1002))

def draw_original_points(frame, title, frame_points, color=(0 , 0, 255)):
    for j in range(0, len(frame_points[0])):
        x, y = int(frame_points[0][j]), int(frame_points[1][j])
        cv.circle(frame, (x, y), 5, color, -1)

    if title is not None:
        draw_title(frame, title)

    return frame

def draw_title(img, text, color=(255, 255, 255)):
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (10, 30)
    font_scale = 1
    thickness = 2
    cv.putText(img, text, org, font, font_scale, color, thickness, cv.LINE_AA)
    return