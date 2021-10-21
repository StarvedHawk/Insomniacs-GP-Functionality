import cv2
import numpy as np
print("Package Imported")

# Importing images in python

#img = cv2.imread("Resources/Essentials.png")


#cv2.imshow("Output",img)
#cv2.waitKey(0)

# Importing and playing a video in python

# cap = cv2.VideoCapture("Resources/My greatest regret.mkv")

# while True:
#     success, img = cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break

# Using the webcam for video feed

cap = cv2.VideoCapture(0)
cap.set(3,640) # ID 3 changes the width
cap.set(4,480) # ID 4 changes the Height
cap.set(10,100) # ID 10 changes the brightness

while True:
     success, img = cap.read()
     # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     cv2.imshow("WebCam",img)
     if cv2.waitKey(1) & 0xFF ==ord('q'):
         break

# Basic functions for OpenCV

#img = cv2.imread("Resources/Photographer.jpg")# Reads the image file
#kernel = np.ones((5,5),np.uint8)

#imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# Changes the colorscale of the image
# in opencv RGB colorscale is known as BGR

#imgBlur = cv2.GaussianBlur(imgGray,(7,7),0) # has to be odd numbers
# the second part is the kernel size
# this function adds a blur to the img chosen

#imgCanny = cv2.Canny(img,200,200)# Detects the edges in an image
# the values in the params are the thresholds for detection

#imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)

#imgErosion = cv2.erode(imgDialation,kernel,iterations=1)

#cv2.imshow("Image",img)
#cv2.imshow("Gray Image",imgGray)# displays selected img
#cv2.imshow("Blurred Image",imgBlur)
#cv2.imshow("Canny Image",imgCanny)
#cv2.imshow("Dilation Image",imgDialation)
#cv2.imshow("Eroded Image",imgErosion)

#cv2.waitKey(0)  # Defines how long image is shown
# 0 means the image is displayed indefinitely

