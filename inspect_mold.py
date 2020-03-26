__author__ = 'martino lorusso'

import os
import random
import time
import math
from collections import defaultdict
import argparse
from copy import copy
import json

import numpy as np
import matplotlib.pyplot as plt

import cv2
import imutils
from skimage import io
from sklearn.cluster import KMeans

from processing.segmentation import Segmentation
from processing.image_processing import gray_and_blur, preprocess_img, define_kernel
from detection.roi_detection import Frame, Sand
from detection.casting_detection import Casting

from config import *
from misc.visualization import *


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the input image")
ap.add_argument("-s", "--steps", action='store_true', default=False,
                help="show intermediate processing steps")
ap.add_argument("-r", "--random", action='store_true', default=False,
                help="load a random image from input images path")
args = vars(ap.parse_args())


# LOADING AND PRE-PROCESSING STEP
if args["random"]:
    # Loading a random image
    image = load_a_random_image(IPCAM_PATH, img_type='png')
else:
    # Loading the chosen image
    image = cv2.imread(IPCAM_PATH + args["image"])

# Pre-processing
cropped = preprocess_img(image, calib_cam_path=CALIBRATION_CAM_PATH, rotation_angle=-3.5,
                         cropping_params=CROP_PARAMS)
if args["steps"]:
    view_image(cropped, img_name="Pre-processed image")


# SEGMENTATION STEP
# Denoising according to the config file parameters
params = FRAME_PARAMS
first = cv2.fastNlMeansDenoising(cropped.copy(), h=params['h'],
                                 templateWindowSize=params['template_window_size'],
                                 searchWindowSize=params['search_window_size'])
# Converting to grayscale and smoothing by blurring
blurred = gray_and_blur(first, blur=True, ksize=7)
# Init a Segmentation class object
sgm = Segmentation(SEGMENTATION_PARAMS)
# Adjusting brightness/contrast and thresholding based on the config file settings
thresh = sgm.scale_abs_convert_thresh(blurred)
# Applying a sequence of morphological transformations to simplify the thresholded image
# in order to better detect contours
transf_seq = TRANSFORMATION_SEQUENCE
transformed = sgm.transform_morphology(thresh, transf_seq, SEGMENTATION_PARAMS['morph_ops_params'])
if args["steps"]:
    print(f"Sequence of transformations: {transf_seq}")
    view_image(
        np.hstack([thresh, transformed]),
        img_name=f"Thresholding | {transf_seq[0].capitalize()} and {transf_seq[1].capitalize()}")


# FRAME ROI DETECTION STEP
frame = Frame(FRAME_PARAMS)
lines = frame.detect_frame_lines(cropped.copy(), params=FRAME_PARAMS, num_frame_lines=10)
if args["steps"]:
    print(f"Number of lines detected: {len(lines)}")

blank = np.zeros((cropped.shape[0], cropped.shape[1], 3), dtype=np.uint8)
line_src = frame.draw_hough_lines(lines, blank, cropped)
_, out = cv2.threshold(line_src, 50, 255, cv2.THRESH_BINARY)
bound_boxes = frame.get_bounding_boxes_from_contours(out)
drawn_boxes, sel_boxes = frame.select_n_largest_boxes(bound_boxes, out, num_boxes=1)
if args["steps"]:
    view_image(drawn_boxes, img_name="Computed inner frame bounding box")

frame_box, diff = frame.set_inner_frame_rect(sel_boxes, FRAME_PARAMS['mean_box_coords'])
inner_frame = frame.get_rect_coords_from_box(frame_box)
outer_frame = frame.compute_outer_frame_coords(inner_frame)
frame_area_coords = [inner_frame, outer_frame]
frame_area_coords = [item for items in frame_area_coords for item in items]
frame_roi = frame.get_frame_roi(transformed.copy(), frame_area_coords)
if args["steps"]:
    view_image(frame_roi, img_name="Frame ROI mask")


# SAND ROI DETECTION STEP
top_left, bottom_right = frame.get_rect_coords_from_box(frame_box)
sand_area_coords = (top_left, bottom_right)
snd = Sand(sand_area_coords)
sand_roi = snd.get_sand_roi(transformed.copy())
if args["steps"]:
    view_image(sand_roi, img_name="Sand ROI mask")


# CASTING DETECTION
cst = Casting()
mold_cast_area = cst.compute_mold_casting_area(cropped, sand_roi, frame_roi,
                                               size_thresh_in_px=SIZE_THRESHOLD_IN_PX)
num_issues, frame_clean = cst.detect_mold_issues(mold_cast_area,
                                               mold_casting_threshold=MOLD_CASTING_THRESHOLD)
mold_state, warn_color, frame_clean = cst.check_mold_state(num_issues, frame_clean)

inspected = cst.update_mold_state(INSPECTED, mold_state, warn_color, mold_cast_area, frame_clean)
if args["steps"]:
    print(f"Mold information update: {inspected}")

with open('./output/data.txt', 'w') as outfile:
    json.dump(inspected, outfile)

# Check if json writing was successful
# with open('data.txt') as json_file:
#     data = json.load(json_file)
#     print(data)

print(f"Saving image . . .")
filename = './output/output_image.png'
cv2.imwrite(filename, cropped)
