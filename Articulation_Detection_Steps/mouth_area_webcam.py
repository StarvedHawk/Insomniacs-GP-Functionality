import cv2
import numpy as np
import dlib
import Articulation_Calc_Functions as ADC_F

mouth = np.zeros((8,2))

face_detector = dlib.get_frontal_face_detector()#detector
landmark_detector = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")#predictor

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, img = cap.read()
    if ret == True:
        faces = face_detector(img, 1)
        landmark_tuple = []
        for k, d in enumerate(faces):
           i = 0
           landmarks = landmark_detector(img, d)
           threshold_points = np.empty((2, 2))
           for n in range(48,60):
               x=landmarks.part(n).x
               y=landmarks.part(n).y
               landmark_tuple.append((x,y))
               cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
               cv2.circle(img, (x, y), 2, (0, 100, 255), -1)
               if n == 49:
                   threshold_points[0] = (x,y)
               if n == 60:
                   threshold_points[1] = (x,y)
           for n in range(60, 68):
              x = landmarks.part(n).x
              y = landmarks.part(n).y
              landmark_tuple.append((x, y))
              mouth[i,0] = x
              mouth[i,1]=y
              cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
              cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
              i=i+1

        total_area = ADC_F.calculate_area_of_mouth(mouth)
        threshold = 37
        if total_area<threshold:
            print("Mouth is closed!")
        else:
            print("Mouth is Open!")
        cv2.imshow("Results",img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break


cv2.waitKey(0)

#TODO: 1) apply this on the webcam input -  Done
#TODO: 2) record words per minute(WPM)
#TODO: 3) make sure no false positive words are recorded
#TODO: 4) check WPM against average WPM
#TODO: 5) record mouth state every x frames (eg frames%10)