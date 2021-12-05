import time
import threading
from flask import Flask, request, url_for, redirect, render_template, Response
from itertools import islice, cycle
from tqdm import tqdm

import numpy as np
from sklearn.cluster import DBSCAN
from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path
from sklearn.preprocessing import StandardScaler
import math
import datetime
import numpy as np
import dlib
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import Articulation_Calc_Functions as ADC_F
import json
import sys
import requests
from requests.structures import CaseInsensitiveDict
import cv2
from video_capture_async.main.gfd.py.video.capture import VideoCaptureThreading

gaze = GazeTracking()
Student_ID = 6354279
Exam = "CSCI321"
Screen = [[0.5132, 0.5131], [0.5468, 0.2821], [0.23140000000000005, 0.2167], [0.2136, 0.5]]
p = path.Path([(0.5132, 0.5131), (0.5468, 0.2821), (0.23140000000000005, 0.2167), (0.2136, 0.5)])
Screen_Captured=False

#Create Flask Server
app = Flask(__name__,template_folder='Templates')

def gen(width=640, height=480):
    cap = VideoCaptureThreading(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.start()
    CurrentWorkingDirectory = os.getcwd()
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(
        r"C:\Users\faisa\PycharmProjects\Specula-Frontend\Insomniacs-GP-Functionality\Resources\shape_predictor_68_face_landmarks.dat")  # predictor
    MD_WORKSPACE_PATH = CurrentWorkingDirectory + '\Tensorflow\workspace'
    MD_SCRIPTS_PATH = CurrentWorkingDirectory + '\Tensorflow\scripts'
    MD_APIMODEL_PATH = MD_WORKSPACE_PATH + 'models'
    MD_ANNOTATION_PATH = MD_WORKSPACE_PATH + '/annotations'
    MD_IMAGE_PATH = MD_WORKSPACE_PATH + '/images'
    MD_MODEL_PATH = MD_WORKSPACE_PATH + '/models'
    MD_PRETRAINED_MODEL_PATH = MD_WORKSPACE_PATH + '/pre-trained-models'
    MD_CONFIG_PATH = MD_MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
    MD_CHECKPOINT_PATH = MD_MODEL_PATH + '/my_ssd_mobnet/'

    configs = config_util.get_configs_from_pipeline_file(MD_CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    category_index = label_map_util.create_category_index_from_labelmap(MD_ANNOTATION_PATH + '/label_map.pbtxt')

    eps = 0.02
    MIN_SAMPLES = 4

    dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)

    threadLock = threading.Lock()
    threads = []

    # Create new threads
    thread1 = Mouth_Detection(1, "Thread-alpha", cap, Student_ID, Exam, face_detector, landmark_detector,
                              detection_model, category_index)
    thread2 = Gaze_Detection(2, "Thread-beta", cap, Student_ID, Exam, face_detector, landmark_detector, dbscan, p)

    # thread3 = Main_Camera(3, "Thread-gamma", cap)

    # Start new Threads
    thread1.start()
    thread2.start()
    # thread3.start()

    # Add threads to thread list
    threads.append(thread1)
    threads.append(thread2)
    # threads.append(thread3)

    gaze = GazeTracking()
    while True:
        ret, frame = cap.read()
        cv2.waitKey(1) & 0xFF
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        cv2.imwrite('temp.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('temp.jpg', 'rb').read() + b'\r\n')
        os.remove('temp.jpg')

    # Wait for all threads to complete
    for t in threads:
        t.join()
    cap.stop()
    print("Exiting Main Thread")

    print("-------------")

def config():
    #Configuration file
    global Screen
    global Screen_Captured
    Temp_Screen =[[0,0,0,0]]
    Screen_Captured = False
    Log_Timer = 0
    Wait_Length = 1
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Horizontal")
    ax.set_ylabel("Vertical")
    ax.set_ylabel("Vertical")
    webcam = cv2.VideoCapture(0)
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        if Screen_Captured:
            break
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (30, 430), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        (77, 77, 209), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (30, 465), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        (77, 77, 209), 1)
        else:
            if gaze.pupils_located:
                if 50 > Log_Timer >= 0:
                    gaze.initialize_screen()
                    cv2.putText(frame, "Look at Top Right", (180, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
                if 80 > Log_Timer >= 50:
                    gaze.save_Top_Right()
                    cv2.putText(frame, "Logging...", (180, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
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
                    cv2.putText(frame, "Done", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
                if Log_Timer == 400:
                    Temp_Screen = gaze.Screen_coords()
                    polygon = plt.Polygon(Temp_Screen)
                    n = [1, 2, 3, 4]
                    xs, ys = zip(*Temp_Screen)
                    #print(xs)
                    #print(ys)
                    for i, txt in enumerate(n):
                        ax.annotate(txt, (xs[i], ys[i]))
                    plt.plot(xs, ys, 'o', color='red')
                    ax.add_patch(polygon)
                    #plt.show()
                    plt.savefig("Plt.jpg")
                    Screen_Captured = True
                Log_Timer = Log_Timer + 1
            else:
                cv2.putText(frame, "Lost eyes", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)

        cv2.imwrite('temp.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('temp.jpg', 'rb').read() + b'\r\n')
        os.remove('temp.jpg')
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('Plt.jpg', 'rb').read() + b'\r\n')
    os.remove('Plt.jpg')
    webcam.release()
    Screen_Captured = True
    update_Screen(Temp_Screen)
    return Temp_Screen

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #print(Screen_Captured)
        if Screen_Captured:
            return redirect(url_for('main'))
    return render_template('Configuration.html')

@app.route('/main')
def main():

    data = [sys.argv[1],sys.argv[2]]
    """Video streaming"""
    return render_template('StudentRoomDisplay.html',data=data)

@app.route('/stream_feed')
def stream_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    global Student_ID
    global Exam
    Student_ID = sys.argv[1]
    Exam = sys.argv[2]
    return Response(gen(640,480),
                mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(config(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

def update_Screen(Temp_Screen):
    global Screen
    global p
    Screen = Temp_Screen
    p = path.Path([Screen[0],Screen[1], Screen[2], Screen[3]])

class Mouth_Detection(threading.Thread):
    def __init__(self, threadID,
                 name,
                 cap,
                 Student_ID,
                 Exam,
                 MD_face_detector,
                 MD_landmark_detector,
                 MD_detection_model,
                 MD_category_index):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.cap = cap
        self.MD_face_detector = MD_face_detector
        self.MD_landmark_detector = MD_landmark_detector
        self.MD_detection_model = MD_detection_model
        self.MD_category_index = MD_category_index
        self.Student_ID=Student_ID
        self.Exam=Exam

    def run(self):
        print("Starting " + self.name + " at " + time.ctime(time.time()))
        t0 = time.time()
        win_name = "cap_" + str(self.threadID)
        # Constant
        # Config Numbers
        MD_LIST_SIZE = 20
        MD_THRESHOLD = 50
        MD_MAX_DANGER = 30
        MD_ERROR_RANGE = 100
        MD_CLOSE_THRESHOLD = 7
        MD_Danger_Maintainence = 1
        MD_DANGER_MOD_CLOSED_CONSTANT = 3  # AMOUNT DANGER VALUE IS DECREASED BY FOR A CONSTANT MOUTH AREA WHEN CLOSED
        MD_DANGER_MOD_OPEN_CONSTANT = 2  # AMOUNT DANGER VALUE IS DECREASED BY FOR A CONSTANT MOUTH AREA WHEN OPEN
        MD_DANGER_MOD_OPEN_DYNAMIC = 3  # AMOUNT DANGER VALUE IS INCREASED BY FOR A CHANGING MOUTH AREA WHEN OPEN
        MD_DANGER_MOD_PHNM_DETECTED = 10  # AMOUNT DANGER VALUE IS INCREASED BY FOR A POSSIBLE WORD DETECTION

        MD_DANGER_REVIEW = 1  # HOW MANY STEPS BACK YOU LOOK FOR CHANGES
        MD_DANGER_REVIEW = MD_MAX_DANGER - MD_DANGER_REVIEW  # ADJUST FOR ALGORITHM
        MD_PHONEME_THRESHOLD = 10

        CurrentWorkingDirectory = os.getcwd()

        # Dynamic Values
        MD_List_iterator = 0
        MD_Danger_Value = 0
        MD_Danger_Check = False
        MD_Close_time = 0
        MD_Open_time = 0
        MD_Face_Check = True
        # Wait_Length = 5
        MD_Phonemes_Detected = 0
        Mouth_CoolDown = 10
        Mouth_Time = 0

        # List that holds past LIST_SIZE number of Mouth_Areas
        MD_mouth_area_list = np.zeros((MD_LIST_SIZE, 2))
        # The Mouth Points
        MD_mouth = np.zeros((8, 2))


        while True:
            ret, frame = self.cap.read()
            cv2.waitKey(1) & 0xFF
            Current_Time_Stamp = datetime.datetime.now()
            faces = self.MD_face_detector(frame, 1)
            Face_Number = len(faces)
            if Face_Number < 2:
                if ret:
                    MD_landmark_tuple = []
                    MD_Face_Check = len(faces) != 0
                    for k, d in enumerate(faces):
                        i=0
                        MD_landmarks = self.MD_landmark_detector(frame, d)
                        MD_threshold_points = np.empty((2, 2))
                        for n in range(48, 60):
                            x = MD_landmarks.part(n).x
                            y = MD_landmarks.part(n).y
                            MD_landmark_tuple.append((x, y))
                        for n in range(60, 68):
                            x = MD_landmarks.part(n).x
                            y = MD_landmarks.part(n).y
                            MD_landmark_tuple.append((x, y))
                            MD_mouth[i, 0] = x
                            MD_mouth[i, 1] = y
                            i = i + 1
                    if MD_Face_Check:
                        MD_total_area = ADC_F.calculate_area_of_mouth(MD_mouth)
                        MD_previous_entry = MD_mouth_area_list[(MD_List_iterator + MD_DANGER_REVIEW) % MD_LIST_SIZE]
                        MD_previous_area = int(MD_previous_entry[0])
                        MD_Checking_Range = range(MD_previous_area - MD_ERROR_RANGE, MD_previous_area + MD_ERROR_RANGE)
                        if MD_total_area < MD_THRESHOLD:
                            if MD_Danger_Check:
                                if MD_Danger_Value != 0 and MD_total_area in MD_Checking_Range:
                                    MD_Danger_Value = MD_Danger_Value - MD_DANGER_MOD_CLOSED_CONSTANT
                                    if MD_Danger_Value < 0:
                                        MD_Danger_Value = 0
                            MD_Close_time = MD_Close_time + 1
                            MD_Open_time = 0
                            if MD_Close_time > MD_CLOSE_THRESHOLD:
                                MD_Danger_Value = 0
                            MD_mouth_area_list[MD_List_iterator] = (MD_total_area, 0)
                        else:
                            if MD_Danger_Check:
                                if MD_Danger_Value != MD_MAX_DANGER and MD_total_area not in MD_Checking_Range:
                                    MD_Danger_Value = MD_Danger_Value + MD_DANGER_MOD_OPEN_DYNAMIC
                                if MD_Danger_Value != 0 and MD_total_area in MD_Checking_Range:
                                    MD_Danger_Value = MD_Danger_Value - MD_DANGER_MOD_OPEN_CONSTANT
                                    if MD_Danger_Value < 0:
                                        MD_Danger_Value = 0
                            MD_Close_time = 0
                            MD_Open_time = MD_Open_time + 1
                            MD_mouth_area_list[MD_List_iterator] = (MD_total_area, 1)
                        if MD_Close_time < MD_CLOSE_THRESHOLD:
                            Adjusted = ADC_F.crop_mouth(frame, self.MD_face_detector, self.MD_landmark_detector)

                            try:
                                image_np = np.array(Adjusted)
                                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                                detections = detect_fn(input_tensor,self.MD_detection_model)
                                num_detections = int(detections.pop('num_detections'))
                                detections = {key: value[0, :num_detections].numpy()
                                              for key, value in detections.items()}
                                detections['num_detections'] = num_detections
                                # detection_classes should be ints.
                                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                                true = 1
                                label_id_offset = 1
                                image_np_with_detections = image_np.copy()
                                viz_utils.visualize_boxes_and_labels_on_image_array(
                                image_np_with_detections,
                                detections['detection_boxes'],
                                detections['detection_classes'] + label_id_offset,
                                detections['detection_scores'],
                                self.MD_category_index,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=5,
                                min_score_thresh=.5,
                                agnostic_mode=False)
                                label = ADC_F.get_label(
                                image_np_with_detections,
                                detections['detection_boxes'],
                                detections['detection_classes'] + label_id_offset,
                                detections['detection_scores'],
                                self.MD_category_index,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=5,
                                min_score_thresh=.5,
                                agnostic_mode=False)
                                if label != "Sil" and label != '':
                                    MD_Phonemes_Detected = MD_Phonemes_Detected + 1
                                MD_List_iterator = (MD_List_iterator + 1) % MD_LIST_SIZE
                                if MD_List_iterator == 19:
                                    MD_Danger_Check = True
                                    if MD_Phonemes_Detected > MD_PHONEME_THRESHOLD:
                                        MD_Danger_Value = MD_Danger_Value + MD_DANGER_MOD_PHNM_DETECTED
                                MD_Danger_Value = MD_Danger_Value - MD_Danger_Maintainence
                                if MD_Danger_Value > MD_MAX_DANGER:
                                    MD_Danger_Value = MD_MAX_DANGER
                                if MD_Danger_Value < 0:
                                    MD_Danger_Value = 0
                                if MD_Danger_Check and MD_Danger_Value > 25:
                                    if Mouth_Time == 0:
                                        Mouth_Time = Mouth_CoolDown
                                        post_to_server(Current_Time_Stamp, "Possible Speech: High Priority",
                                                       DangerLevel=MD_Danger_Value)
                                if MD_Danger_Check and 25 >= MD_Danger_Value > 15:
                                    if Mouth_Time == 0:
                                        Mouth_Time = Mouth_CoolDown
                                        post_to_server(Current_Time_Stamp, "Possible Speech: Medium Priority",
                                                       DangerLevel=MD_Danger_Value)
                                if MD_Danger_Check and 15 >= MD_Danger_Value > 10:
                                    if Mouth_Time == 0:
                                        Mouth_Time = Mouth_CoolDown
                                        post_to_server(Current_Time_Stamp, "Possible Speech: Low Priority",
                                                       DangerLevel=MD_Danger_Value)
                                if Mouth_Time>0:
                                    Mouth_Time -= 1
                            except:
                                print("Multiple Faces Detected")
        print('[i] Frames per second: {:.2f}, with_multi_threading'.format(self.n_frames / (time.time() - t0)))

class Gaze_Detection(threading.Thread):
    def __init__(self, threadID,
                 name,
                 cap,
                 Student_ID,
                 Exam,
                 face_detector,
                 landmark_detector,
                 dbscan,
                 pathScreen):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.cap = cap
        self.GD_face_detector = face_detector
        self.GD_landmark_detector = landmark_detector
        self.dbscan = dbscan
        self.p = pathScreen
        self.Student_ID=Student_ID
        self.Exam=Exam
    def run(self):

        print("Starting " + self.name + " at " + time.ctime(time.time()))
        t0 = time.time()
        win_name = "cap_" + str(self.threadID)
        #CONFIG
        DATA_POINT_LIMIT = 100  # Number of Data Points per window
        WINDOW_SIZE = 50
        NUMBER_OF_WINDOWS = DATA_POINT_LIMIT / WINDOW_SIZE
        Wait_Length = 1
        CAPTURE_SPAN = 2  # Number of frames between data point captures
        gaze = GazeTracking()
        Gaze_CoolDown = 10
        Other_CoolDown = 5
        dbscan = self.dbscan
        GD_MAX_DANGER = 30


        #DYNAMIC
        Gaze_points = np.zeros((DATA_POINT_LIMIT + 4, 2))
        Temp_Window = np.zeros((WINDOW_SIZE, 2))
        Window_One = np.zeros((WINDOW_SIZE, 2))
        Window_Two = np.zeros((WINDOW_SIZE, 2))
        Capture_Span_Iter = 0  # Iterator between Frames
        Screen_Captured = True
        gaze_point_iter = 0
        First_Cluster = True  # First Cluster successful
        repeat_collection = False
        Gaze_Time = 0
        Other_Time = 0
        GD_Danger_Value = 0

        while True:

            ret, frame = self.cap.read()
            cv2.waitKey(1) & 0xFF
            Current_Time_Stamp = datetime.datetime.now()
            GD_faces = self.GD_face_detector(frame, 1)
            GD_Face_Number = len(GD_faces)
            GD_Face_Check = len(GD_faces) != 0
            gaze.refresh(frame)
            frame = gaze.annotated_frame()
            repeat_collection = False
            if GD_Face_Check:
                if GD_Face_Number < 2:
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
                            cv2.putText(frame, "Horizontal:  " + str(left_pupil), (30, 430), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        (77, 77, 209), 1)
                            cv2.putText(frame, "Vertical: " + str(right_pupil), (30, 465), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        (77, 77, 209), 1)
                            if First_Cluster:  # Saving and creating cluster 1
                                if Capture_Span_Iter == CAPTURE_SPAN - 1:
                                    Gaze_points[gaze_point_iter] = gaze.Gaze_coords()
                                    if gaze_point_iter < WINDOW_SIZE:
                                        Window_One[gaze_point_iter] = Gaze_points[gaze_point_iter]
                                    else:
                                        Window_Two[gaze_point_iter - WINDOW_SIZE] = Gaze_points[gaze_point_iter]
                                    if Gaze_points[gaze_point_iter][0] < 0 or Gaze_points[gaze_point_iter][0] > 1:
                                        repeat_collection = True
                                    if Gaze_points[gaze_point_iter][1] < 0 or Gaze_points[gaze_point_iter][1] > 1:
                                        repeat_collection = True
                                    if repeat_collection:
                                        gaze_point_iter -= 1
                                        Capture_Span_Iter -= 1
                                        continue
                                    if gaze_point_iter + 1 == DATA_POINT_LIMIT:
                                        # print("-----------------------------------------------\n")
                                        model = dbscan.fit(Gaze_points)
                                        labels = model.labels_
                                        sample_cores = np.zeros_like(labels, dtype=bool)
                                        sample_cores[dbscan.core_sample_indices_] = True
                                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                        label_iter = 0
                                        Average_Percentage = 0
                                        # print("Number of clusters : ", n_clusters)
                                        for i in range(0, n_clusters):
                                            clusters = Gaze_points[labels == i]  # assign cluster points to array
                                            points = np.array(clusters)
                                            bool_array = p.contains_points(points)
                                            countTrue = np.count_nonzero(bool_array)
                                            arrayLength = np.size(bool_array)
                                            checkPercentage = (countTrue / arrayLength) * 100
                                            if checkPercentage > 90:
                                                GD_Danger_Value = round((100 - checkPercentage) / 100 * (GD_MAX_DANGER - 3))
                                                Average_Percentage += checkPercentage
                                            else:
                                                GD_Danger_Value = round((100 - checkPercentage) / 100 * (GD_MAX_DANGER - 3))
                                                Average_Percentage += checkPercentage
                                        if n_clusters > 0:
                                            Average_Percentage = Average_Percentage / n_clusters
                                            GD_Danger_Value = round((100 - Average_Percentage) / 100 * (GD_MAX_DANGER - 3))
                                            GD_msg = "Student is looking off screen ", round(100 - Average_Percentage,
                                                                                             2), "% of the time"
                                            if Gaze_Time == 0:
                                                Gaze_Time = Gaze_CoolDown
                                                post_to_server(Current_Time_Stamp, GD_msg, DangerLevel=GD_Danger_Value)
                                        # print("-----------------------------------------------\n")
                                    gaze_point_iter = (gaze_point_iter + 1) % DATA_POINT_LIMIT
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
                                        # print("-----------------------------------------------\n")
                                        Window_One = Window_Two  # Cycling the Windows
                                        Window_Two = Temp_Window
                                        Gaze_points = np.concatenate(Window_One, Window_Two)

                                        model = dbscan.fit(Gaze_points)
                                        labels = model.labels_
                                        sample_cores = np.zeros_like(labels, dtype=bool)
                                        sample_cores[dbscan.core_sample_indices_] = True
                                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                                        label_iter = 0
                                        Average_Percentage = 0
                                        for i in range(0, n_clusters):
                                            clusters = Gaze_points[labels == i]  # assign cluster points to array
                                            points = np.array(clusters)
                                            bool_array = p.contains_points(points)
                                            countTrue = np.count_nonzero(bool_array)
                                            arrayLength = np.size(bool_array)
                                            checkPercentage = (countTrue / arrayLength) * 100
                                            if checkPercentage > 90:
                                                # print("Student is looking at screen with ", round(checkPercentage, 2),"% certainity ", i)
                                                GD_Danger_Value = round((100 - checkPercentage) / 100 * (GD_MAX_DANGER - 3))
                                                Average_Percentage += checkPercentage
                                            else:
                                                # print("Student is looking off screen with ", round(100 - checkPercentage, 2),"% certainity", i)
                                                GD_Danger_Value = round((100 - checkPercentage) / 100 * (GD_MAX_DANGER - 3))
                                                Average_Percentage += checkPercentage
                                                # break
                                        if n_clusters > 0:
                                            Average_Percentage = Average_Percentage / n_clusters
                                            GD_Danger_Value = round((100 - Average_Percentage) / 100 * (GD_MAX_DANGER - 3))
                                            GD_msg = "Student is looking off screen ", round(100 - Average_Percentage,
                                                                                             2), "% of the time"
                                            if Gaze_Time == 0:
                                                Gaze_Time = Gaze_CoolDown
                                                post_to_server(Current_Time_Stamp, GD_msg, DangerLevel=GD_Danger_Value)

                                        # print("-----------------------------------------------\n")
                                    gaze_point_iter = (gaze_point_iter + 1) % WINDOW_SIZE
                            Capture_Span_Iter = (Capture_Span_Iter + 1) % CAPTURE_SPAN
                    else:
                        cv2.putText(frame, "Lost eyes", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)
                else:
                    if Other_Time == 0:
                        Other_Time = Other_CoolDown
                        post_to_server(Current_Time_Stamp, "Multiple Faces Detected", DangerLevel=30)
            else:
                if Other_Time == 0:
                    Other_Time = Other_CoolDown
                    post_to_server(Current_Time_Stamp, "No Faces Detected", DangerLevel=30)
            if Gaze_Time != 0:
                Gaze_Time = Gaze_Time - 1
            if Other_Time != 0:
                Other_Time = Other_Time - 1
            # cv2.imshow("Screen",frame)

class Main_Camera(threading.Thread):
    def __init__(self,
                 threadID,
                 name,
                 cap,):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.cap = cap
    def run(self):
        print("Starting " + self.name + " at " + time.ctime(time.time()))
        t0 = time.time()
        win_name = "cap_" + str(self.threadID)
        gaze = GazeTracking()
        while True:
            ret, frame = self.cap.read()
            cv2.waitKey(1) & 0xFF
            gaze.refresh(frame)
            frame = gaze.annotated_frame()
            cv2.imwrite('temp.jpg', frame)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('temp.jpg', 'rb').read() + b'\r\n')
            os.remove('temp.jpg')




@tf.function
def detect_fn(image,MD_detection_model):
    image, shapes = MD_detection_model.preprocess(image)
    prediction_dict = MD_detection_model.predict(image, shapes)
    MD_detections = MD_detection_model.postprocess(prediction_dict, shapes)
    return MD_detections

def post_to_server(timeStamp,textMessage,DangerLevel):

    global Student_ID
    global Exam


    url = "http://127.0.0.1:8000/api/TimeLine/"

    Danger_Text = str(DangerLevel)
    TS_Text = timeStamp.strftime('%Y-%m-%d %H:%M:%S')
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"

    data = {}
    data['student'] = Student_ID
    data['CurrentExam'] = Exam
    data['AItimeStamp'] = TS_Text
    data['AItextMessage'] = textMessage
    data['AIdangerLevel'] = Danger_Text

    print(data)

    # try:
    #     resp = requests.post(url, headers=headers, data=json.dumps(data))
    #     print(resp.status_code)
    #     Gaze_Time = Gaze_CoolDown
    #     Mouth_Time = Mouth_CoolDown
    # except Exception:
    #     print("Connection Error!")

if __name__ == '__main__':
    app.run()