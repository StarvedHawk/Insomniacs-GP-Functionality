#IMPORTS
from itertools import islice, cycle
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from Gaze_Tracking import GazeTracking
from matplotlib import pyplot as plt
from matplotlib import path
from sklearn.preprocessing import StandardScaler
import math
import cv2
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

#GAZE DETECTION CONFIG
#STATIC
DATA_POINT_LIMIT = 400 #Number of Data Points per window
WINDOW_SIZE = 200
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

plt.scatter(Boundaries[0][0], Boundaries[0][1], s=10,marker='+', color="#ff0000")
plt.scatter(Boundaries[1][0], Boundaries[1][1], s=10,marker='+', color="#ff0000")
plt.scatter(Boundaries[2][0], Boundaries[2][1], s=10,marker='+', color="#ff0000")
plt.scatter(Boundaries[3][0], Boundaries[3][1], s=10,marker='+', color="#ff0000")

p = path.Path([(0.5132, 0.5131), (0.5468, 0.2821), (0.23140000000000005, 0.2167), (0.2136, 0.5)])

#polygon = plt.Polygon(Screen)
#n = [1,2,3,4]
#ax.add_patch(polygon)


## ARTICULATION DETECTION CONFIG
#Config Numbers
MD_LIST_SIZE = 20
MD_THRESHOLD = 50
MD_MAX_DANGER = 30
MD_ERROR_RANGE = 100
MD_CLOSE_THRESHOLD = 7
MD_Danger_Maintainence = 1
MD_DANGER_MOD_CLOSED_CONSTANT = 3 #AMOUNT DANGER VALUE IS DECREASED BY FOR A CONSTANT MOUTH AREA WHEN CLOSED
MD_DANGER_MOD_OPEN_CONSTANT = 2 #AMOUNT DANGER VALUE IS DECREASED BY FOR A CONSTANT MOUTH AREA WHEN OPEN
MD_DANGER_MOD_OPEN_DYNAMIC = 3 #AMOUNT DANGER VALUE IS INCREASED BY FOR A CHANGING MOUTH AREA WHEN OPEN
MD_DANGER_MOD_PHNM_DETECTED = 10 #AMOUNT DANGER VALUE IS INCREASED BY FOR A POSSIBLE WORD DETECTION

MD_DANGER_REVIEW = 1 #HOW MANY STEPS BACK YOU LOOK FOR CHANGES
MD_DANGER_REVIEW = MD_MAX_DANGER - MD_DANGER_REVIEW #ADJUST FOR ALGORITHM
MD_PHONEME_THRESHOLD = 10

#Dynamic Values
MD_List_iterator = 0
MD_Danger_Value = 0
MD_Danger_Check = False
MD_Close_time = 0
MD_Open_time = 0
MD_Face_Check = True
#Wait_Length = 5
MD_Phonemes_Detected = 0

#List that holds past LIST_SIZE number of Mouth_Areas
MD_mouth_area_list = np.zeros((MD_LIST_SIZE, 2))
#The Mouth Points
MD_mouth = np.zeros((8, 2))

#Models
MD_face_detector = dlib.get_frontal_face_detector()  # detector
MD_landmark_detector = dlib.shape_predictor(r"Resources/shape_predictor_68_face_landmarks.dat")  # predictor

MD_CONFIG_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
MD_WORKSPACE_PATH = 'Tensorflow/workspace'
MD_SCRIPTS_PATH = 'Tensorflow/scripts'
MD_APIMODEL_PATH = 'Tensorflow/models'
MD_ANNOTATION_PATH = MD_WORKSPACE_PATH + '/annotations'
MD_IMAGE_PATH = MD_WORKSPACE_PATH + '/images'
MD_MODEL_PATH = MD_WORKSPACE_PATH + '/models'
MD_PRETRAINED_MODEL_PATH = MD_WORKSPACE_PATH + '/pre-trained-models'
MD_CONFIG_PATH = MD_MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
MD_CHECKPOINT_PATH = MD_MODEL_PATH + '/my_ssd_mobnet/'

# Load pipeline config and build a detection model
MD_configs = config_util.get_configs_from_pipeline_file(MD_CONFIG_PATH)
MD_detection_model = model_builder.build(model_config=MD_configs['model'], is_training=False)
MD_category_index = label_map_util.create_category_index_from_labelmap(MD_ANNOTATION_PATH + '/label_map.pbtxt')

# Restore checkpoint
MD_ckpt = tf.compat.v2.train.Checkpoint(model=MD_detection_model)
MD_ckpt.restore(os.path.join(MD_CHECKPOINT_PATH, 'ckpt-25')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = MD_detection_model.preprocess(image)
    prediction_dict = MD_detection_model.predict(image, shapes)
    MD_detections = MD_detection_model.postprocess(prediction_dict, shapes)
    return MD_detections

webcam = cv2.VideoCapture(0)

#Testing-For Phonemes( NEED TO IMPROVE ACCURACY )
#cap = cv2.VideoCapture(r'C:\\Users\\faisa\\PycharmProjects\\CommonAssets\\Lipread_assets\\Extracts\\videos\\s1\\bbal8p.mpg')

MD_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
MD_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = webcam.read()
    # GAZE TRACKING REAL-TIME
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
                        print("-----------------------------------------------\n")
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
                                print("Student is looking at screen with ", round(checkPercentage, 2), "% certainity",i)
                            else:
                                print("Student is looking off screen with ", round(100 - checkPercentage, 2),"% certainity",i)
                                #break
                            print("-----------------------------------------------\n")
                    gaze_point_iter = (gaze_point_iter + 1 ) % DATA_POINT_LIMIT
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
                        print("-----------------------------------------------\n")
                        Window_One = Window_Two                     #Cycling the Windows
                        Window_Two = Temp_Window
                        Gaze_points = np.concatenate(Window_One, Window_Two)

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
                                print("Student is looking at screen with ",round(checkPercentage,2), "% certainity ",i)
                            else:
                                print("Student is looking off screen with ",round(100 - checkPercentage,2), "% certainity",i)
                                #break
                        print("-----------------------------------------------\n")
                    gaze_point_iter = (gaze_point_iter + 1 ) % WINDOW_SIZE
            Capture_Span_Iter = (Capture_Span_Iter + 1) % CAPTURE_SPAN
        else:
            cv2.putText(frame, "Lost eyes", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (77, 77, 209), 1)

    #MOUTH DETECTION REAL-TIME
    print("-----------------------------------------------\n")
    if ret == True:
        MD_faces = MD_face_detector(frame, 1)
        MD_landmark_tuple = []
        MD_Face_Check = len(MD_faces) != 0
        for k, d in enumerate(MD_faces):
            i = 0
            MD_landmarks = MD_landmark_detector(frame, d)
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
                print("Mouth is closed!")
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
                print("Mouth is Open!")
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
                Adjusted = ADC_F.crop_mouth(frame, MD_face_detector, MD_landmark_detector)
                try:
                    image_np = np.array(Adjusted)
                    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                    detections = detect_fn(input_tensor)
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
                        MD_category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.5,
                        agnostic_mode=False)

                    label = ADC_F.get_label(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'] + label_id_offset,
                        detections['detection_scores'],
                        MD_category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.5,
                        agnostic_mode=False)
                    # cv2.imshow("object detection", cv2.resize(image_np_with_detections,
                    #                                         (image_np_with_detections.shape[1],
                    #                                         image_np_with_detections.shape[0])))
                    if label != "Sil" and label != '':
                        MD_Phonemes_Detected = MD_Phonemes_Detected + 1
                        print("Phoneme Detected : ", label)
                    # cv2.imshow("object detection", cv2.resize(image_np_with_detections,
                    #                                         (image_np_with_detections.shape[1],
                    #                                         image_np_with_detections.shape[0])))

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
                        print("DANGER LEVEL : ", MD_Danger_Value, " : HIGH")
                    if MD_Danger_Check and 25 >= MD_Danger_Value > 15:
                        print("DANGER LEVEL : ", MD_Danger_Value, " : MEDIUM")
                    if MD_Danger_Check and 15 >= MD_Danger_Value > 10:
                        print("DANGER LEVEL : ", MD_Danger_Value, " : LOW")
                except:
                    print("No Face")
        else:
            print("No Face Detected!!")
        print("-----------------------------------------------")
    else:
        break

    cv2.imshow("Demo", frame)

    if cv2.waitKey(Wait_Length) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break