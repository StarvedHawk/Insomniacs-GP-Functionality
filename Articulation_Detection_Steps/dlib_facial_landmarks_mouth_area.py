import cv2
import numpy as np
import dlib

def calculate_area_of_mouth(pt):

    # dividing the area into sections based on the number of points and shapes in the matrix of points for the mouth
    sections = [0,0,0,0]

    #Creating an array to hold the triangular coordinates
    pts_passed = np.empty((3,2))
    # passing the  coordinates of the triangular section
    pts_passed[0]=pt[0]
    pts_passed[1] = pt[1]
    pts_passed[2] = pt[7]

    #calling the function to calculate the area of a triangle
    sections[0]=area_of_triangle(pts_passed)

    #repeating previous steps
    pts_passed[0] = pt[0]
    pts_passed[1] = pt[1]
    pts_passed[2] = pt[7]

    sections[3] = area_of_triangle(pts_passed)

    # Creating an array to hold the Quadrangular coordinates
    pts_passed = np.empty((4, 2))
    # passing the  coordinates of the Quadrangular section
    pts_passed[0] = pt[1]
    pts_passed[1] = pt[2]
    pts_passed[2] = pt[6]
    pts_passed[3] = pt[7]
    # calling the function to calculate the area of a Quadrangle
    sections[1] = area_of_Quadrangle(pts_passed)

    # Creating an array to hold the Quadrangular coordinates
    pts_passed = np.empty((4, 2))
    # passing the  coordinates of the Quadrangular section
    pts_passed[0] = pt[2]
    pts_passed[1] = pt[3]
    pts_passed[2] = pt[5]
    pts_passed[3] = pt[6]
    # calling the function to calculate the area of a Quadrangle
    sections[2] = area_of_Quadrangle(pts_passed)
    print(sections)
    total_area = sections[0]+sections[1]+sections[2]+sections[3]
    print("Total Area :",total_area)
    return total_area

def area_of_triangle(pts):
    area = ((pts[0, 0] * (pts[1, 1] - pts[2, 1])) +
            (pts[1, 0] * (pts[2, 1] - pts[0, 1])) +
            (pts[2, 0] * (pts[0, 1] - pts[1, 1]))) / 2
    return area

def area_of_Quadrangle(pts):
    area = ( (pts[0,0]*pts[1,1]-pts[0,1]*pts[1,0]) +
             (pts[1,0]*pts[2,1]-pts[1,1]*pts[2,0]) +
             (pts[2,0]*pts[3,1]-pts[2,1]*pts[3,0]) +
             (pts[3,0]*pts[0,1]-pts[3,1]*pts[0,0])) / 2
    return area


mouth = np.zeros((8,2))

face_detector = dlib.get_frontal_face_detector()#detector
landmark_detector = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")#predictor

#mouth closed positive value
#img = cv2.imread("Resources/smile2.jpg")[:,:,::-1]

#mouth open positive value
img = cv2.imread("../Resources/face.jpg")[:, :, ::-1]
img = cv2.resize(img,((int)(img.shape[1]/2),(int)(img.shape[0]/2)))

#mouth closed negative value
#img = cv2.imread("Resources/closed-Mouth.jpg")[:,:,::-1]



img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)

faces = face_detector(img, 1)
landmark_tuple = []
for k, d in enumerate(faces):
   i = 0
   landmarks = landmark_detector(img, d)
   for n in range(48,60):
       x=landmarks.part(n).x
       y=landmarks.part(n).y
       landmark_tuple.append((x,y))
       cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
       cv2.circle(img, (x, y), 2, (0, 100, 255), -1)
   for n in range(60, 68):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmark_tuple.append((x, y))
      mouth[i,0] = x
      mouth[i,1]=y
      cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
      cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
      i=i+1


print(mouth)
total_area = calculate_area_of_mouth(mouth)
if total_area<20.0:
    print("Mouth is closed!")
else:
    print("Mouth is Open!")
print(img)
cv2.imshow("Results",img)


cv2.waitKey(0)

#TODO: 1) apply this on the webcam input
#TODO: 2) record words per minute(WPM)
#TODO: 3) make sure no false positive words are recorded
#TODO: 4) check WPM against average WPM
#TODO: 5) record mouth state every x frames (eg frames%10)