import cv2
import numpy as np

#uses warp perspective to straighten images
width,height = 250,350#aspect ratio of cards
pts1 = np.float32([[320,194],[385,157],[390,287],[455,250]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)


img = cv2.imread("Resources/koiCards.jpg")
img = cv2.resize(img,((int)(img.shape[1]/2),(int)(img.shape[0]/2)))

imgOutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Cards",img)
cv2.imshow("Output",imgOutput)

cv2.waitKey(0)