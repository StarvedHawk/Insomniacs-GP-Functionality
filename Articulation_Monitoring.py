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

## MOUTH AREA
#Config Numbers
LIST_SIZE = 20
THRESHOLD = 50
MAX_DANGER = 30
ERROR_RANGE = 100
CLOSE_THRESHOLD = 7
DANGER_MOD_CLOSED_CONSTANT = 3 #AMOUNT DANGER VALUE IS DECREASED BY FOR A CONSTANT MOUTH AREA WHEN CLOSED
DANGER_MOD_OPEN_CONSTANT = 2 #AMOUNT DANGER VALUE IS DECREASED BY FOR A CONSTANT MOUTH AREA WHEN OPEN
DANGER_MOD_OPEN_DYNAMIC = 3 #AMOUNT DANGER VALUE IS INCREASED BY FOR A CHANGING MOUTH AREA WHEN OPEN
DANGER_MOD_PHNM_DETECTED = 10 #AMOUNT DANGER VALUE IS INCREASED BY FOR A POSSIBLE WORD DETECTION

DANGER_REVIEW = 1 #HOW MANY STEPS BACK YOU LOOK FOR CHANGES
DANGER_REVIEW = MAX_DANGER - DANGER_REVIEW #ADJUST FOR ALGORITHM
PHONEME_THRESHOLD = 10

#Dynamic Values
List_iterator = 0
Danger_Value = 0
Danger_Check = False
Close_time = 0
Open_time = 0
Face_Check = True
Wait_Length = 5
Phonemes_Detected = 0

#List that holds past LIST_SIZE number of Mouth_Areas
mouth_area_list = np.zeros((LIST_SIZE, 2))
#The Mouth Points
mouth = np.zeros((8, 2))

#Models
face_detector = dlib.get_frontal_face_detector()  # detector
landmark_detector = dlib.shape_predictor(r"Resources/shape_predictor_68_face_landmarks.dat")  # predictor

CONFIG_PATH = r"C:\Users\faisa\PycharmProjects\RealTimeObjectDetection\Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config"
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-25')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

cap = cv2.VideoCapture(0)

#Testing-For Phonemes( NEED TO IMPROVE ACCURACY )
#cap = cv2.VideoCapture(r'C:\\Users\\faisa\\PycharmProjects\\CommonAssets\\Lipread_assets\\Extracts\\videos\\s1\\bbal8p.mpg')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    print("-----------------------------------------------\n")
    if ret == True:
        faces = face_detector(frame, 1)
        landmark_tuple = []
        Face_Check = len(faces) != 0
        for k, d in enumerate(faces):
           i = 0
           landmarks = landmark_detector(frame, d)
           threshold_points = np.empty((2, 2))
           for n in range(48,60):
               x=landmarks.part(n).x
               y=landmarks.part(n).y
               landmark_tuple.append((x,y))
           for n in range(60, 68):
              x = landmarks.part(n).x
              y = landmarks.part(n).y
              landmark_tuple.append((x, y))
              mouth[i,0] = x
              mouth[i,1]=y
              i=i+1

        if Face_Check:
            total_area = ADC_F.calculate_area_of_mouth(mouth)
            previous_entry = mouth_area_list[(List_iterator + DANGER_REVIEW) % LIST_SIZE]
            previous_area = int(previous_entry[0])
            Checking_Range = range(previous_area-ERROR_RANGE,previous_area+ERROR_RANGE)
            print(Checking_Range)
            if total_area < THRESHOLD:
                print("Mouth is closed!")
                Wait_Length = 5
                if Danger_Check:
                    if Danger_Value != 0 and total_area in Checking_Range:
                        Danger_Value = Danger_Value - DANGER_MOD_CLOSED_CONSTANT
                        if Danger_Value < 0:
                            Danger_Value = 0
                Close_time = Close_time + 1
                Open_time = 0
                if Close_time > CLOSE_THRESHOLD:
                    Danger_Value = 0
                    Wait_Length = 10
                mouth_area_list[List_iterator] = (total_area, 0)
            else:
                print("Mouth is Open!")
                Wait_Length = 5
                if Danger_Check:
                    if Danger_Value != MAX_DANGER and total_area not in Checking_Range:
                        Danger_Value = Danger_Value + DANGER_MOD_OPEN_DYNAMIC
                    if Danger_Value != 0 and total_area in Checking_Range:
                        Danger_Value = Danger_Value - DANGER_MOD_OPEN_CONSTANT
                        if Danger_Value < 0:
                            Danger_Value = 0
                Close_time = 0
                Open_time = Open_time + 1
                mouth_area_list[List_iterator] = (total_area, 1)
            if Close_time < CLOSE_THRESHOLD:
                Adjusted = ADC_F.crop_mouth(frame, face_detector, landmark_detector)
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
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.5,
                        agnostic_mode=False)
                    label = ADC_F.get_label(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'] + label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.5,
                        agnostic_mode=False)
                    cv2.imshow("object detection", cv2.resize(image_np_with_detections, (
                        image_np_with_detections.shape[1], image_np_with_detections.shape[0])))
                    if label != "Sil" and label != '':
                        Phonemes_Detected = Phonemes_Detected + 1
                        print("Phoneme Detected : ", label)
                    cv2.imshow("object detection", cv2.resize(image_np_with_detections, (
                    image_np_with_detections.shape[1], image_np_with_detections.shape[0])))

                    List_iterator = (List_iterator + 1) % LIST_SIZE
                    if List_iterator == 19:
                        Danger_Check = True
                        if Phonemes_Detected > PHONEME_THRESHOLD:
                            Danger_Value = Danger_Value + DANGER_MOD_PHNM_DETECTED
                    if Danger_Value > MAX_DANGER:
                        Danger_Value = MAX_DANGER
                    if Danger_Value < 0:
                        Danger_Value = 0
                    if Danger_Check and Danger_Value > 25:
                        print("DANGER LEVEL : ", Danger_Value, " : HIGH")
                    if Danger_Check and 25 >= Danger_Value > 15:
                        print("DANGER LEVEL : ", Danger_Value, " : MEDIUM")
                    if Danger_Check and 15 >= Danger_Value > 10:
                        print("DANGER LEVEL : ", Danger_Value, " : LOW")
                except:
                    print("No Face")
        else:
            print("No Face Detected!!")
        print("-----------------------------------------------")
        if cv2.waitKey(Wait_Length) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    else:
        break

cv2.waitKey(0)

#TODO: 1) Create an Aspect Ratio code that allows to factor distance
#TODO: 2) Improve the range detection for the Danger Level
#TODO: 3) Improve the Model ( Either Label more images and retrain /
#TODO:                        Collect Images for vowels using the Camera and Retrain )
