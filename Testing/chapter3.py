import cv2
import numpy as np

img = cv2.imread("Resources/Essentials.png")

img = np.zeros((512,512,3),np.uint8)
print(img.shape)
#img[:]= 255,0,0#color whole img blue

cv2.line(img,(0,0),(250,350),(0,255,0),3)
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)#replace width of line with cv2.filled to fill rectangle
cv2.circle(img,(400,50),30,(255,255,0),5)
cv2.putText(img," OPEN CV ",(100,500),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,150),1)

cv2.imshow("Black Board",img)

cv2.waitKey(0)