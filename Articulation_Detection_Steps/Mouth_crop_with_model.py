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

mouth = np.zeros((8, 2))

face_detector = dlib.get_frontal_face_detector()  # detector
landmark_detector = dlib.shape_predictor(
    r"C:\Users\faisa\PycharmProjects\OpenCVPython\Resources\shape_predictor_68_face_landmarks.dat")  # predictor

CONFIG_PATH = r"C:\Users\faisa\PycharmProjects\RealTimeObjectDetection\Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config"
WORKSPACE_PATH = '../Tensorflow/workspace'
SCRIPTS_PATH = '../Tensorflow/scripts'
APIMODEL_PATH = '../Tensorflow/models'
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


def crop_mouth(img, face_detector, landmark_detector):
    faces = face_detector(img, 1)
    top = -1
    left = -1
    right = -1
    bottom = -1
    for k, d in enumerate(faces):
        i = 0
        landmarks = landmark_detector(img, d)
        landmark_tuple = []
        for n in range(48, 60):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if top == -1:
                top = x
            if left == -1:
                left = y
            if right == -1:
                right = y
            if bottom == -1:
                bottom = x

            if x < top:
                top = x
            if y < left:
                left = y
            if y > right:
                right = y
            if x > bottom:
                bottom = x

            landmark_tuple.append((x, y))
            # cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            # cv2.circle(img, (x, y), 2, (0, 100, 255), -1)

    Padding = 5

    top = top - Padding
    left = left - Padding
    right = right + Padding
    bottom = bottom + Padding

    # cv2.circle(img, (top, left), 1, (130, 130, 0), -1)
    # cv2.circle(img, (top, right), 1, (130, 130, 0), -1)
    # cv2.circle(img, (bottom, left), 1, (130, 130, 0), -1)
    # cv2.circle(img, (bottom, right), 1, (130, 130, 0), -1)

    # top = 405
    # left = 405
    # right = 377
    # bottom = 377

    MOUTH_WIDTH = 50
    MOUTH_HEIGHT = 100
    resize_bool = 0
    height_ratio = 1
    width_ratio = 1

    height = bottom - top
    width = right - left

    # print("height", height, "width", width)

    # cropped = img[left:right, top:bottom]

    if MOUTH_HEIGHT < height or MOUTH_WIDTH < width:
        current_mouth_height = height
        current_mouth_width = width
        total_height_pad = 0
        total_width_pad = 0
        # print("height ratio : ", height_ratio, "width ratio : ", width_ratio)
        if current_mouth_height % 10 > 0:
            total_height_pad += 10 - current_mouth_height % 10
            current_mouth_height += total_height_pad
        if current_mouth_width % 5 > 0:
            total_width_pad += 5 - current_mouth_width % 5
            current_mouth_width += total_width_pad
        height_ratio = current_mouth_height / 10
        width_ratio = current_mouth_width / 5
        # print("height : ", current_mouth_height, "width : ", current_mouth_width)
        # print("height ratio : ", height_ratio, "width ratio : ", width_ratio)
        if height_ratio > width_ratio:
            total_width_pad = (height_ratio - width_ratio) * 5
            current_mouth_width += total_width_pad
        if width_ratio > height_ratio:
            total_height_pad = (width_ratio - height_ratio) * 10
            current_mouth_height += total_height_pad
        height_ratio = current_mouth_height / 10
        width_ratio = current_mouth_width / 5
        # print("height : ", current_mouth_height, "width : ", current_mouth_width)
        # print("height ratio : ", height_ratio, "width ratio : ", width_ratio)

        total_width_pad = current_mouth_width - width
        total_height_pad = current_mouth_height - height
        top_height_pad = math.ceil(total_height_pad / 2)
        bot_height_pad = math.floor(total_height_pad / 2)
        bottom = bottom + bot_height_pad
        top = top - top_height_pad

        width_difference = MOUTH_WIDTH - width
        Left_width_pad = math.ceil(total_width_pad / 2)
        right_width_pad = math.floor(total_width_pad / 2)
        right = right + right_width_pad
        left = left - Left_width_pad

        height = bottom - top
        width = right - left
        resize_height = current_mouth_height / height_ratio * 10
        resize_width = current_mouth_width / width_ratio * 10
        resize_bool = 1

        # print("height : ", current_mouth_height, "width : ", current_mouth_width)
        # print("Real_height : ", height, "Real_width : ", width)
        # print("resize height : ", resize_height, "resize width : ", resize_width)
        # print("height ratio : ", height_ratio, "width ratio : ", width_ratio)

    if MOUTH_HEIGHT > height and resize_bool == 0:
        height_difference = MOUTH_HEIGHT - height
        top_height_pad = math.ceil(height_difference / 2)
        bot_height_pad = math.floor(height_difference / 2)
        bottom = bottom + bot_height_pad
        top = top - top_height_pad
        # print("top", top,"bottom", bottom)
    if MOUTH_WIDTH > width and resize_bool == 0:
        width_difference = MOUTH_WIDTH - width
        Left_width_pad = math.ceil(width_difference / 2)
        right_width_pad = math.floor(width_difference / 2)
        right = right + right_width_pad
        left = left - Left_width_pad
        # print("left", left, "right", right)

    height = bottom - top
    width = right - left

    # cv2.circle(img, (top, left), 1, (0, 0, 0), -1)
    # cv2.circle(img, (top, right), 1, (0, 0, 0), -1)
    # cv2.circle(img, (bottom, left), 1, (0, 0, 0), -1)
    # cv2.circle(img, (bottom, right), 1, (0, 0, 0), -1)
    # print("height",height,"width",width)
    # print("Crop Points", Mouth_crop_points)

    Adjusted = img[left:right, top:bottom]
    # print("height_ratio",height_ratio,"width_ratio",width_ratio)

    if resize_bool != 0:
        # print("Adjusted[1]",Adjusted.shape[1],"Adjusted[0]",Adjusted.shape[0])
        # print("Adjusted[1]", Adjusted.shape[1]/height_ratio*10, "Adjusted[0]", Adjusted.shape[0]/width_ratio*10)
        Aspect_param_1 = Adjusted.shape[1] / height_ratio * 10
        Aspect_param_2 = Adjusted.shape[0] / width_ratio * 10
        resized = cv2.resize(Adjusted, ((int)(Aspect_param_1), (int)(Aspect_param_2)))
        # print("resized[1]", resized.shape[1], "resized[0]", resized.shape[0])
        height = resized.shape[1]
        width = resized.shape[0]
    if resize_bool != 0:
        return resized
    return Adjusted


cap = cv2.VideoCapture(
    r'C:\\Users\\faisa\\PycharmProjects\\CommonAssets\\Lipread_assets\\Extracts\\videos\\s1\\bbal8p.mpg')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if ret == True:
        Adjusted = crop_mouth(frame, face_detector, landmark_detector)

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

        label, image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.2,
            agnostic_mode=False)
        cv2.waitKey(5)
        cv2.imshow("object detection", cv2.resize(image_np_with_detections, (
        image_np_with_detections.shape[1], image_np_with_detections.shape[0])))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.waitKey(0)