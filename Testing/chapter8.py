import cv2
import numpy as np

#functions
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

#gets all the countours in the image and their areas

def getCountours(img, imgCountour):
    countours, Hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgCountour, cnt, -1, (255, 255, 0), 4)
            perimeter = cv2.arcLength(cnt,True)
            print(perimeter)
            approx = cv2.approxPolyDP(cnt,0.02*perimeter,True)
            print(len(approx))
            object_corner = len(approx)
            x, y, w, h= cv2.boundingRect(approx)
            if object_corner == 3 : object_type = "Triangle"
            elif object_corner == 4 :
                aspect_Ratio = w / float(h)
                if aspect_Ratio > 0.95 and aspect_Ratio < 1.05: object_type= "Square"
                else : object_type = "Rectangle"
                object_type = "Square"
            elif object_corner > 4 : object_type = "Circle"
            else : object_type = "None"
            cv2.rectangle(imgCountour, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(imgCountour,object_type,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,0,0),2)

#code
path = 'Resources/shapes.png'
img = cv2.imread(path)
imgCountour = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)

getCountours(imgCanny,imgCountour)

imgBlank = np.zeros_like(img)

imgStack = stackImages(0.7,([img,imgGray,imgBlur],
                            [imgCanny,imgCountour,imgBlank]))

# cv2.imshow("Original",img)
# cv2.imshow("Gray",imgGray)
# cv2.imshow("Blur",imgBlur)
cv2.imshow("Stack",imgStack)

cv2.waitKey(0)