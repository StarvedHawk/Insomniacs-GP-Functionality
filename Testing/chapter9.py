import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/group2.jpg')
#img = cv2.resize(img,((int)(img.shape[1]/2),(int)(img.shape[0]/2)))
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray",imgGray)
faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

print("number of faces :",len(faces))

cv2.imshow("Result",img)
cv2.waitKey(0)
