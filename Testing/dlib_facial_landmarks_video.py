import dlib
import cv2
import numpy as np
from dataclasses import dataclass


face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
cap.set(3,640)  # ID 3 changes the width
cap.set(4,480)  # ID 4 changes the Height
cap.set(10,100) # ID 10 changes the brightness

while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    faces = face_detector(img, 1)

    landmark_tuple = []
    for k, d in enumerate(faces):
        landmarks = landmark_detector(img, d)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
    cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)