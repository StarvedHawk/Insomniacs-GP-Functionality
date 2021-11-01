from itertools import islice, cycle
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path
from sklearn.preprocessing import StandardScaler

#STATIC
DATA_POINT_LIMIT = 100 #Number of Data Points per window
WINDOW_SIZE = 50
NUMBER_OF_WINDOWS = DATA_POINT_LIMIT / WINDOW_SIZE
Wait_Length = 1
CAPTURE_SPAN = 5            #Number of frames between data point captures
eps = 0.02
MIN_SAMPLES = 4

#DYNAMIC
Gaze_points = np.zeros((DATA_POINT_LIMIT + 4,2))
Temp_Window = np.zeros((WINDOW_SIZE,2))
Window_One = np.zeros((WINDOW_SIZE,2))
Window_Two = np.zeros((WINDOW_SIZE,2))
Capture_Span_Iter = 0       #Iterator between Frames
Screen_Captured = True
gaze_point_iter = 0
First_Cluster = True       #First Cluster successful
repeat_collection = False

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

#Making the pyplot fig
fig = plt.figure()
fig.set_dpi(80)
fig.set_size_inches(9, 7.5)
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")


#Model
dbscan = DBSCAN(eps=eps,min_samples=MIN_SAMPLES)

#TEMP
#Screen Co-ords taken for testing
Screen = [[0.5132, 0.5131], [0.5468, 0.2821], [0.23140000000000005, 0.2167], [0.2136, 0.5]]
Boundaries = [[0, 0],[0, 1],[1, 0],[1, 1]]

print(Boundaries[0][0])

plt.scatter(Boundaries[0][0], Boundaries[0][1], s=10,marker='+', color="#ff0000")
plt.scatter(Boundaries[1][0], Boundaries[1][1], s=10,marker='+', color="#ff0000")
plt.scatter(Boundaries[2][0], Boundaries[2][1], s=10,marker='+', color="#ff0000")
plt.scatter(Boundaries[3][0], Boundaries[3][1], s=10,marker='+', color="#ff0000")

p = path.Path([(0.5132, 0.5131), (0.5468, 0.2821), (0.23140000000000005, 0.2167), (0.2136, 0.5)])

#polygon = plt.Polygon(Screen)
#n = [1,2,3,4]
#ax.add_patch(polygon)

while True:
    ret, frame = webcam.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    repeat_collection = False
    if gaze.pupils_located:
        if Screen_Captured:
            text = ""
            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"

            cv2.putText(frame, text, (30, 400), cv2.FONT_HERSHEY_DUPLEX, 0.7, (98, 98, 242), 2)

            Normalised_Eyes = gaze.Gaze_coords()
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, "Horizontal:  " + str(left_pupil), (30, 430), cv2.FONT_HERSHEY_DUPLEX, 0.5, (77, 77, 209), 1)
            cv2.putText(frame, "Vertical: " + str(right_pupil), (30, 465), cv2.FONT_HERSHEY_DUPLEX, 0.5, (77, 77, 209),1)

            #cv2.putText(frame, "Horizontal:  " + str(Normalised_Eyes[0]), (30, 430), cv2.FONT_HERSHEY_DUPLEX, 0.5,(77, 77, 209), 1)
            #cv2.putText(frame, "Vertical: " + str(Normalised_Eyes[1]), (30, 465), cv2.FONT_HERSHEY_DUPLEX, 0.5,(77, 77, 209), 1)

            #Function to save data_point
            if First_Cluster:           #Saving and creating cluster 1
                if Capture_Span_Iter == CAPTURE_SPAN - 1:
                    Gaze_points[gaze_point_iter] = gaze.Gaze_coords()
                    if gaze_point_iter < WINDOW_SIZE:
                        Window_One[gaze_point_iter] = Gaze_points[gaze_point_iter]
                    else:
                        Window_Two[gaze_point_iter-WINDOW_SIZE] = Gaze_points[gaze_point_iter]
                    if Gaze_points[gaze_point_iter][0] < 0 or Gaze_points[gaze_point_iter][0] > 1:
                        repeat_collection = True
                    if Gaze_points[gaze_point_iter][1] < 0 or Gaze_points[gaze_point_iter][1] > 1:
                        repeat_collection = True
                    if repeat_collection:
                        gaze_point_iter -= 1
                        Capture_Span_Iter -= 1
                        continue
                    if gaze_point_iter + 1 == DATA_POINT_LIMIT:
                        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
                        model = dbscan.fit(Gaze_points)
                        labels = model.labels_
                        sample_cores = np.zeros_like(labels, dtype=bool)
                        sample_cores[dbscan.core_sample_indices_] = True
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        label_iter = 0

                        print("Number of clusters : ", n_clusters)

                        for i in range(0, n_clusters):
                            clusters = Gaze_points[labels == i]  # assign cluster points to array
                            points = np.array(clusters)
                            bool_array = p.contains_points(points)
                            countTrue = np.count_nonzero(bool_array)
                            arrayLength = np.size(bool_array)
                            checkPercentage = (countTrue / arrayLength) * 100
                            if checkPercentage > 90:
                                print("Student is looking at screen with ", round(checkPercentage, 2), "% certainity")
                            else:
                                print("Student is looking off screen with ", round(100 - checkPercentage, 2),"% certainity")
                                break
                    gaze_point_iter = (gaze_point_iter + 1 ) % DATA_POINT_LIMIT
                Capture_Span_Iter = (Capture_Span_Iter + 1) % CAPTURE_SPAN
            else:
                if Capture_Span_Iter == CAPTURE_SPAN - 1:
                    Temp_Window[gaze_point_iter] = gaze.Gaze_coords()
                    if Temp_Window[gaze_point_iter][0] < 0 or Temp_Window[gaze_point_iter][0] > 1:
                        repeat_collection = True
                    if Temp_Window[gaze_point_iter][1] < 0 or Temp_Window[gaze_point_iter][1] > 1:
                        repeat_collection = True
                    if repeat_collection:
                        gaze_point_iter -= 1
                        Capture_Span_Iter -= 1
                        continue
                    if gaze_point_iter + 1 == WINDOW_SIZE:
                        print("Before :")
                        print(Window_One)
                        Window_One = Window_Two                     #Cycling the Windows
                        print("After :")
                        print(Window_One)
                        Window_Two = Temp_Window
                        Gaze_points = np.concatenate(Window_One, Window_Two)
                        #print(Gaze_points)


                        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
                        model = dbscan.fit(Gaze_points)
                        labels = model.labels_
                        sample_cores = np.zeros_like(labels, dtype=bool)
                        sample_cores[dbscan.core_sample_indices_] = True
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        label_iter = 0

                        for i in range(0,n_clusters):
                            clusters = Gaze_points[labels == i]  # assign cluster points to array
                            points = np.array(clusters)
                            bool_array = p.contains_points(points)
                            countTrue = np.count_nonzero(bool_array)
                            arrayLength = np.size(bool_array)
                            checkPercentage = (countTrue / arrayLength) * 100
                            if checkPercentage > 90:
                                print("Student is looking at screen with ",round(checkPercentage,2), "% certainity")
                            else:
                                print("Student is looking off screen with ",round(100 - checkPercentage,2), "% certainity")
                                break
                    gaze_point_iter = (gaze_point_iter + 1 ) % WINDOW_SIZE
                Capture_Span_Iter = (Capture_Span_Iter + 1) % CAPTURE_SPAN

        else:
            cv2.putText(frame, "Lost eyes", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
    cv2.imshow("Demo", frame)

    if cv2.waitKey(Wait_Length) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break