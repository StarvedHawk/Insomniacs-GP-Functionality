import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")
img = cv2.imread("../Resources/face.jpg")[:, :, ::-1]
img = cv2.resize(img,((int)(img.shape[1]/2),(int)(img.shape[0]/2)))
img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)                         #Bug in opencv causes a color swap
                                                                    #so u need to swap colors twice
faces = face_detector(img, 1)

print(faces)
landmark_tuple = []
for k, d in enumerate(faces):
   landmarks = landmark_detector(img, d)
   for n in range(0, 68):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmark_tuple.append((x, y))
      cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
      cv2.circle(img, (x, y), 2, (255, 255, 0), -1)


cv2.imshow("Results",img)


cv2.waitKey(0)