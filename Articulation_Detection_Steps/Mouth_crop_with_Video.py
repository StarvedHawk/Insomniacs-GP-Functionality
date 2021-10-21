import math
import cv2
import numpy as np
import dlib

mouth = np.zeros((8,2))

face_detector = dlib.get_frontal_face_detector()#detector
landmark_detector = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")#predictor

#mouth closed positive value
img = cv2.imread("../Resources/smile2.jpg")[:, :, ::-1]

#mouth open positive value
#img = cv2.imread("Resources/face.jpg")[:,:,::-1]
#img = cv2.resize(img,((int)(img.shape[1]/2),(int)(img.shape[0]/2)))

#mouth closed negative value
#img = cv2.imread("Resources/closed-Mouth.jpg")[:,:,::-1]
img = cv2.imread("C:/Users/faisa/PycharmProjects/CommonAssets/Lipread_assets/Extracts/Extracted/s2/bbaf3p/mouth_006.png")[:,:,::-1]


img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)

def crop_mouth(img,face_detector,landmark_detector):
    faces = face_detector(img, 1)
    top = -1
    left = -1
    right = -1
    bottom = -1
    for k, d in enumerate(faces):
        i = 0
        landmarks = landmark_detector(img, d)
        landmark_tuple = []
        for n in range(48,60):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            if top == -1 :
                top = x
            if left == -1:
                left = y
            if right == -1:
                right = y
            if bottom == -1:
                bottom = x

            if x < top:
                top = x
            if y < left:
                left = y
            if y > right:
                right = y
            if x > bottom:
                bottom = x

            landmark_tuple.append((x,y))
            #cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            #cv2.circle(img, (x, y), 2, (0, 100, 255), -1)


    Padding = 5

    top = top - Padding
    left = left - Padding
    right = right + Padding
    bottom = bottom + Padding

    # cv2.circle(img, (top, left), 1, (130, 130, 0), -1)
    # cv2.circle(img, (top, right), 1, (130, 130, 0), -1)
    # cv2.circle(img, (bottom, left), 1, (130, 130, 0), -1)
    # cv2.circle(img, (bottom, right), 1, (130, 130, 0), -1)

    #top = 405
    #left = 405
    #right = 377
    #bottom = 377

    MOUTH_WIDTH = 50
    MOUTH_HEIGHT = 100
    resize_bool = 0
    height_ratio = 1
    width_ratio = 1

    height = bottom-top
    width = right-left


    # print("height", height, "width", width)

    #cropped = img[left:right, top:bottom]

    if MOUTH_HEIGHT < height or MOUTH_WIDTH < width:
        current_mouth_height = height
        current_mouth_width = width
        total_height_pad = 0
        total_width_pad = 0
        #print("height ratio : ", height_ratio, "width ratio : ", width_ratio)
        if current_mouth_height % 10 > 0:
            total_height_pad += 10-current_mouth_height % 10
            current_mouth_height += total_height_pad
        if current_mouth_width % 5 > 0:
            total_width_pad += 5 - current_mouth_width % 5
            current_mouth_width += total_width_pad
        height_ratio = current_mouth_height / 10
        width_ratio = current_mouth_width / 5
        #print("height : ", current_mouth_height, "width : ", current_mouth_width)
        #print("height ratio : ", height_ratio, "width ratio : ", width_ratio)
        if height_ratio > width_ratio:
            total_width_pad = (height_ratio - width_ratio)*5
            current_mouth_width += total_width_pad
        if width_ratio > height_ratio:
            total_height_pad = (width_ratio - height_ratio) * 10
            current_mouth_height += total_height_pad
        height_ratio = current_mouth_height / 10
        width_ratio = current_mouth_width / 5
        #print("height : ", current_mouth_height, "width : ", current_mouth_width)
        #print("height ratio : ", height_ratio, "width ratio : ", width_ratio)

        total_width_pad = current_mouth_width - width
        total_height_pad = current_mouth_height - height
        top_height_pad = math.ceil(total_height_pad / 2)
        bot_height_pad = math.floor(total_height_pad / 2)
        bottom = bottom + bot_height_pad
        top = top - top_height_pad


        width_difference = MOUTH_WIDTH - width
        Left_width_pad = math.ceil(total_width_pad / 2)
        right_width_pad = math.floor(total_width_pad / 2)
        right = right + right_width_pad
        left = left - Left_width_pad

        height = bottom - top
        width = right - left
        resize_height = current_mouth_height / height_ratio * 10
        resize_width = current_mouth_width / width_ratio * 10
        resize_bool = 1

        # print("height : ", current_mouth_height, "width : ", current_mouth_width)
        # print("Real_height : ", height, "Real_width : ", width)
        # print("resize height : ", resize_height, "resize width : ", resize_width)
        # print("height ratio : ", height_ratio, "width ratio : ", width_ratio)

    if MOUTH_HEIGHT > height and resize_bool == 0:
        height_difference = MOUTH_HEIGHT - height
        top_height_pad = math.ceil(height_difference/2)
        bot_height_pad = math.floor(height_difference/2)
        bottom = bottom + bot_height_pad
        top = top - top_height_pad
        #print("top", top,"bottom", bottom)
    if MOUTH_WIDTH > width and resize_bool == 0:
        width_difference = MOUTH_WIDTH - width
        Left_width_pad = math.ceil(width_difference/2)
        right_width_pad = math.floor(width_difference/2)
        right = right + right_width_pad
        left = left - Left_width_pad
        #print("left", left, "right", right)

    height = bottom-top
    width = right-left

    # cv2.circle(img, (top, left), 1, (0, 0, 0), -1)
    # cv2.circle(img, (top, right), 1, (0, 0, 0), -1)
    # cv2.circle(img, (bottom, left), 1, (0, 0, 0), -1)
    # cv2.circle(img, (bottom, right), 1, (0, 0, 0), -1)
    # print("height",height,"width",width)
    # print("Crop Points", Mouth_crop_points)

    Adjusted = img[left:right, top:bottom]
    #print("height_ratio",height_ratio,"width_ratio",width_ratio)

    if resize_bool != 0:
        # print("Adjusted[1]",Adjusted.shape[1],"Adjusted[0]",Adjusted.shape[0])
        # print("Adjusted[1]", Adjusted.shape[1]/height_ratio*10, "Adjusted[0]", Adjusted.shape[0]/width_ratio*10)
        Aspect_param_1 = Adjusted.shape[1]/height_ratio*10
        Aspect_param_2 = Adjusted.shape[0]/width_ratio*10
        resized = cv2.resize(Adjusted,((int)(Aspect_param_1),(int)(Aspect_param_2)))
        #print("resized[1]", resized.shape[1], "resized[0]", resized.shape[0])
        height = resized.shape[1]
        width = resized.shape[0]
    if resize_bool != 0:
        return resized
    return Adjusted
cap = cv2.VideoCapture(r'C:\\Users\\faisa\\PycharmProjects\\CommonAssets\\Lipread_assets\\Extracts\\videos\\s1\\bbal8p.mpg')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    Adjusted = crop_mouth(frame,face_detector,landmark_detector)
    cv2.imshow("Adjusted", Adjusted)
    if ret == False:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imshow("Results",img)

cv2.waitKey(0)