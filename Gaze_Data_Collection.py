import cv2
import numpy as np

from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path
from itertools import islice, cycle

from tqdm import tqdm
import sys
import cv2
import numpy as np
from sklearn import cluster, datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics

from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path
from sklearn.preprocessing import StandardScaler

#STATIC
DATA_POINT_LIMIT = 50 #400
#WINDOW_SIZE = 200
Wait_Length = 1
CAPTURE_SPAN = 5            #Number of frames between data point captures
eps = 0.22
MIN_SAMPLES = 4

#DYNAMIC
Gaze_points = np.zeros((DATA_POINT_LIMIT,2))
Capture_Span_Iter = 0       #Iterator between Frames
Screen_Captured = True
gaze_point_iter = 0
First_Cluster = False       #First Cluster successful
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
plt.xticks(())
plt.yticks(())

#Model
dbscan = DBSCAN(eps=eps,min_samples=MIN_SAMPLES)

#TEMP
#Screen Co-ords taken for testing
Screen = [[0.5132, 0.5131], [0.5468, 0.2821], [0.23140000000000005, 0.2167], [0.2136, 0.5]]

while True:
    _, frame = webcam.read()
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
            if Capture_Span_Iter == CAPTURE_SPAN - 1:
                Gaze_points[gaze_point_iter] = gaze.Gaze_coords()
                if Gaze_points[gaze_point_iter][0] < 0 or Gaze_points[gaze_point_iter][0] > 1:
                    repeat_collection = True
                if Gaze_points[gaze_point_iter][1] < 0 or Gaze_points[gaze_point_iter][1] > 1:
                    repeat_collection = True
                if repeat_collection:
                    gaze_point_iter -= 1
                    CAPTURE_SPAN -= 1
                    continue
                if gaze_point_iter + 1 == DATA_POINT_LIMIT:
                    print(Gaze_points)
                    Gaze_points = StandardScaler().fit_transform(Gaze_points)
                    dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
                    model = dbscan.fit(Gaze_points)
                    labels = model.labels_
                    sample_cores = np.zeros_like(labels, dtype=bool)
                    sample_cores[dbscan.core_sample_indices_] = True
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                    label_iter = 0

                    for _, label in tqdm(enumerate(labels), desc="Drawing_Clusters", total=len(labels)):
                        # print(label)
                        # print(labels[0])
                        # print(len(X))
                        # print(len(labels))
                        y_pred = dbscan.labels_.astype(np.int)
                        colors = np.array(list(islice
                                               (cycle
                                                (["#FE4A49",
                                                  "#2AB7CA",
                                                  "#A1C38C",
                                                  "#666699",
                                                  "#efe464",
                                                  "#b24c37",
                                                  "#432eb7",
                                                  "#3c8608",
                                                  "#d2c815",
                                                  "#d1aa99",
                                                  "#09d788",
                                                  "#3c028f",
                                                  "#27fa97",
                                                  "#9764c5",
                                                  "#09c8f1",
                                                  "#034d24",
                                                  "#84e332",
                                                  "#a7cd76",
                                                  "#6d73b6",
                                                  "#2F847C"]), n_clusters + 1)))
                        # add black color for outliers (if any)
                        colors = np.append(colors, ["#000000"])
                        print(plt.scatter(Gaze_points[:, 0], Gaze_points[:, 1], s=20, color=colors[y_pred]))
                        #print(plt.scatter(Gaze_points[:, 0], Gaze_points[:, 1], s=20, color="#000000"))
                        label_iter += 1
                        print(label_iter)
                    plt.show()
                    print(Gaze_points)
                gaze_point_iter = (gaze_point_iter + 1 ) % DATA_POINT_LIMIT
            Capture_Span_Iter = (Capture_Span_Iter + 1) % CAPTURE_SPAN

        else:
            cv2.putText(frame, "Lost eyes", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
    cv2.imshow("Demo", frame)

    if cv2.waitKey(Wait_Length) & 0xFF == ord('q'):
        break