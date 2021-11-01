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
Wait_Length = 5
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
    print("-----------------------------------------------\n")
    if ret == True:
        MD_faces = MD_face_detector(frame, 1)
        MD_landmark_tuple = []
        MD_Face_Check = len(MD_faces) != 0
        for k, d in enumerate(MD_faces):
           i = 0
           MD_landmarks = MD_landmark_detector(frame, d)
           MD_threshold_points = np.empty((2, 2))
           for n in range(48,60):
               x=MD_landmarks.part(n).x
               y=MD_landmarks.part(n).y
               MD_landmark_tuple.append((x, y))
           for n in range(60, 68):
              x = MD_landmarks.part(n).x
              y = MD_landmarks.part(n).y
              MD_landmark_tuple.append((x, y))
              MD_mouth[i, 0] = x
              MD_mouth[i, 1]=y
              i=i+1

        if MD_Face_Check:
            MD_total_area = ADC_F.calculate_area_of_mouth(MD_mouth)
            MD_previous_entry = MD_mouth_area_list[(MD_List_iterator + MD_DANGER_REVIEW) % MD_LIST_SIZE]
            MD_previous_area = int(MD_previous_entry[0])
            MD_Checking_Range = range(MD_previous_area - MD_ERROR_RANGE, MD_previous_area + MD_ERROR_RANGE)
            if MD_total_area < MD_THRESHOLD:
                print("Mouth is closed!")
                Wait_Length = 5
                if MD_Danger_Check:
                    if MD_Danger_Value != 0 and MD_total_area in MD_Checking_Range:
                        MD_Danger_Value = MD_Danger_Value - MD_DANGER_MOD_CLOSED_CONSTANT
                        if MD_Danger_Value < 0:
                            MD_Danger_Value = 0
                MD_Close_time = MD_Close_time + 1
                MD_Open_time = 0
                if MD_Close_time > MD_CLOSE_THRESHOLD:
                    MD_Danger_Value = 0
                    Wait_Length = 10
                MD_mouth_area_list[MD_List_iterator] = (MD_total_area, 0)
            else:
                print("Mouth is Open!")
                Wait_Length = 5
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
                    #cv2.imshow("object detection", cv2.resize(image_np_with_detections,
                     #                                         (image_np_with_detections.shape[1],
                      #                                         image_np_with_detections.shape[0])))
                    if label != "Sil" and label != '':
                        MD_Phonemes_Detected = MD_Phonemes_Detected + 1
                        print("Phoneme Detected : ", label)
                    #cv2.imshow("object detection", cv2.resize(image_np_with_detections,
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
        if cv2.waitKey(Wait_Length) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
    else:
        break

cv2.waitKey(0)

#TODO: 1) Create an Aspect Ratio code that allows to factor distance
#TODO: 2) Improve the range detection for the Danger Level
#TODO: 3) Improve the Model ( Either Label more images and retrain /
#TODO:                        Collect Images for vowels using the Camera and Retrain )
