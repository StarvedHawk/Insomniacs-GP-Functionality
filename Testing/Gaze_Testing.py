"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import numpy as np

from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


#Config Values
Screen_Captured = False
Log_Timer = 0
Wait_Length = 1

#Making the pyplot fig
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    if Screen_Captured:
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (30, 430), cv2.FONT_HERSHEY_DUPLEX, 0.5, (77, 77, 209), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (30, 465), cv2.FONT_HERSHEY_DUPLEX, 0.5, (77  , 77, 209), 1)
    else:
        if gaze.pupils_located:
            if 50 > Log_Timer >= 0:
                gaze.initialize_screen()
                cv2.putText(frame, "Look at Top Right", (180, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77  , 77, 209), 1)
            if 80 > Log_Timer >= 50:
                gaze.save_Top_Right()
                cv2.putText(frame, "Logging...", (180, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77  , 77, 209), 1)
            if 130 > Log_Timer >= 80:
                cv2.putText(frame, "Look at Bot Right", (180, 465), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if 150 > Log_Timer >= 130:
                gaze.save_Bot_Right()
                cv2.putText(frame, "Logging...", (180, 465), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if 200 > Log_Timer >= 150:
                cv2.putText(frame, "Look at bot Left", (30, 465), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if 250 > Log_Timer >= 200:
                gaze.save_Bot_Left()
                cv2.putText(frame, "Logging...", (30, 465), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if 300 > Log_Timer >= 250:
                cv2.putText(frame, "Look at Top Left", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if 350 > Log_Timer >= 300:
                gaze.save_Top_Left()
                cv2.putText(frame, "Logging...", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if 400 > Log_Timer >= 350:
                cv2.putText(frame, "Done", (200 , 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
            if Log_Timer == 400:
                Screen = gaze.Screen_coords()
                print(Screen)
                polygon = plt.Polygon(Screen)
                n = [1,2,3,4]
                xs,ys = zip(*Screen)
                print(xs)
                print(ys)
                for i, txt in enumerate(n):
                    ax.annotate(txt, (xs[i], ys[i]))
                plt.plot(xs,ys,'o',color='red')
                ax.add_patch(polygon)
                plt.show()
                Screen_Captured = True
            Log_Timer = Log_Timer + 1
        else:
            cv2.putText(frame, "Lost eyes", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(Wait_Length) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

