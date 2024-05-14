import numpy as np
import cv2 as cv

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

def create_video_capture(path_file_video):
    return cv.VideoCapture(path_file_video)

def create_output_video(output_path_file_video):
    return cv.VideoWriter(output_path_file_video, fourcc, 33.0, (1000, 1002))