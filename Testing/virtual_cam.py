import pyvirtualcam
from pyvirtualcam import PixelFormat
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3,640)  # ID 3 changes the width
cap.set(4,480)  # ID 4 changes the Height
cap.set(10,100) # ID 10 changes the brightness

cam_fmt = PixelFormat.BGR
with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=cam_fmt) as cam:
    pyvirtualcam.PixelFormat('24BG')
    print(f'Using virtual camera: {cam.device} with format : {cam.fmt}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        img = cap.read()
        success, frame[:] = img # webcam input
        cam.send(frame)
        cam.sleep_until_next_frame()