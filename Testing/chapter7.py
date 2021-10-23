import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(val):
    pass

path = "Resources/Red_Glass (2).jpg"

img = cv2.imread(path)
img = cv2.resize(img,(200,300))

#TrackBars
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)                #Trackbar height = 60 : width adjustable
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)#Hue Min
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)#Hue Max
cv2.createTrackbar("Sat Min","TrackBars",183,255,empty)#Sat Min
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)#Sat Max
cv2.createTrackbar("Val Min","TrackBars",40,255,empty)#Val Min
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)#Val Max

# print(img)
#Checks if img is empty

while True:
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv2.getTrackbarPos("Val Min","TrackBars")
    v_max = cv2.getTrackbarPos("Val Max","TrackBars")
    #print(h_min,h_max,s_min,s_max,v_min,v_max)
    #prints all values
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    mask=cv2.inRange(imgHSV,lower,upper)
    imgResults=cv2.bitwise_and(img,img,mask=mask)

    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask",mask)
    # cv2.imshow("Masked Img",imgResults)


    #using the stacking function
    imgStack = stackImages(0.8,([img,imgHSV],[mask,imgResults]))
    cv2.imshow("All Images", imgStack)

    cv2.waitKey(1)