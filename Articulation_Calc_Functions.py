import math
import collections
import cv2
import numpy as np


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

    Padding = 5

    top = top - Padding
    left = left - Padding
    right = right + Padding
    bottom = bottom + Padding

    MOUTH_WIDTH = 50
    MOUTH_HEIGHT = 100
    resize_bool = 0
    height_ratio = 1
    width_ratio = 1

    height = bottom - top
    width = right - left


    if MOUTH_HEIGHT < height or MOUTH_WIDTH < width:
        current_mouth_height = height
        current_mouth_width = width
        total_height_pad = 0
        total_width_pad = 0
        if current_mouth_height % 10 > 0:
            total_height_pad += 10 - current_mouth_height % 10
            current_mouth_height += total_height_pad
        if current_mouth_width % 5 > 0:
            total_width_pad += 5 - current_mouth_width % 5
            current_mouth_width += total_width_pad
        height_ratio = current_mouth_height / 10
        width_ratio = current_mouth_width / 5
        if height_ratio > width_ratio:
            total_width_pad = (height_ratio - width_ratio) * 5
            current_mouth_width += total_width_pad
        if width_ratio > height_ratio:
            total_height_pad = (width_ratio - height_ratio) * 10
            current_mouth_height += total_height_pad
        height_ratio = current_mouth_height / 10
        width_ratio = current_mouth_width / 5

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

    if MOUTH_HEIGHT > height and resize_bool == 0:
        height_difference = MOUTH_HEIGHT - height
        top_height_pad = math.ceil(height_difference / 2)
        bot_height_pad = math.floor(height_difference / 2)
        bottom = bottom + bot_height_pad
        top = top - top_height_pad
    if MOUTH_WIDTH > width and resize_bool == 0:
        width_difference = MOUTH_WIDTH - width
        Left_width_pad = math.ceil(width_difference / 2)
        right_width_pad = math.floor(width_difference / 2)
        right = right + right_width_pad
        left = left - Left_width_pad
    height = bottom - top
    width = right - left
    Adjusted = img[left:right, top:bottom]

    if resize_bool != 0:

        Aspect_param_1 = Adjusted.shape[1] / height_ratio * 10
        Aspect_param_2 = Adjusted.shape[0] / width_ratio * 10
        resized = cv2.resize(Adjusted, ((int)(Aspect_param_1), (int)(Aspect_param_2)))

        height = resized.shape[1]
        width = resized.shape[0]
    if resize_bool != 0:
        return resized
    return Adjusted

def get_label(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):

  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.

  final_label = ""
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
            final_label = display_str

  return final_label

def calculate_area_of_mouth(pt):

    # dividing the area into sections based on the number of points and shapes in the matrix of points for the mouth
    sections = [0,0,0,0]

    #Creating an array to hold the triangular coordinates
    pts_passed = np.empty((3,2))
    # passing the  coordinates of the triangular section
    pts_passed[0] = pt[0]
    pts_passed[1] = pt[1]
    pts_passed[2] = pt[7]

    #calling the function to calculate the area of a triangle
    sections[0]=area_of_triangle(pts_passed)

    #repeating previous steps
    pts_passed[0] = pt[3]
    pts_passed[1] = pt[4]
    pts_passed[2] = pt[5]

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
    #print(sections)
    total_area = sections[0]+sections[1]+sections[2]+sections[3]
    #print("Total Area :",total_area)
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