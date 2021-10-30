import cv2
import numpy as np
import datetime

faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3,640)  # ID 3 changes the width
cap.set(4,480)  # ID 4 changes the Height
cap.set(10,100) # ID 10 changes the brightness

current_faces = 0
current_time = datetime.datetime.now()

fps = cap.get(cv2.CAP_PROP_FPS)
print("The FPS is ",fps)

while True:
     success, img = cap.read()
     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
     for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     cv2.imshow("WebCam",img)
     if len(faces)!=current_faces:
         current_time = datetime.datetime.now()
         print("number of Faces :",len(faces))
         print("Time : [",current_time.hour,":",current_time.minute,":",current_time.second,"]")
         current_faces=len(faces)
     if cv2.waitKey(1) & 0xFF ==ord('q'):
         break


cv2.waitKey(0)
