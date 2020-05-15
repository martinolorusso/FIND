#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This python script performs the inspection of a mold image to detect potential issues occurred
    during the production. In particular, the type of anomaly investigated is a casting surface
    significantly exceeding the pouring cup, and possibly affecting also the mold chassis.
    This is the most critical scenario since it may cause the plant blocking in the worst case.
    The metal shapes occurring in such risky condition are called "burrs". So, this algorithm is
    designed to search for potential burrs by computing the casting surface area on the mold
    floor and spotting the presence of poured metal on the upper frame.

    To run this script n times, you can execute these lines from bash shell:

    for i in `seq <n>`; do python app.py <*args>; done

    """

__author__ = 'martino lorusso'


# Import the necessary packages
import argparse
import json
from json2html import json2html
from flask import Flask, request, render_template

import numpy as np
import matplotlib.pyplot as plt
import cv2

from processing.segmentation import Segmentation
from processing.image_processing import gray_and_blur, preprocess_img
from detection.roi_detection import Frame, Sand
from detection.casting_detection import Casting

from config import *
from misc.visualization import *


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the input image")
ap.add_argument("-s", "--steps", action='store_true', default=False,
                help="show intermediate processing steps")
ap.add_argument("-r", "--random", action='store_true', default=False,
                help="load a random image from input images path")
args = vars(ap.parse_args())


# ------ LOADING AND PRE-PROCESSING ------
# Load an image and apply the first basic transformations preceding the segmentation step

# Load a random image if required. On the contrary, load the chosen image
if args["random"]:
    image = load_a_random_image(os.getcwd() + IPCAM_PATH, img_type='png')
else:
    image = cv2.imread(os.getcwd() + IPCAM_PATH + args["image"])
print(f"Processing image . . .")
# Pre-process the loaded image
cropped = preprocess_img(image, cam_calib_path=os.getcwd() + IPCAM_CALIBRATION_PATH,
                         rotation_angle=-3.5, cropping_params=CROP_PARAMS)

# Show the pre-processing step output image if required
if args["steps"]:
    view_image(cropped, img_name="Pre-processed image")


# ------ SEGMENTATION ------
# Apply the image segmentation for separating significant elements (casting surfaces)
# from negligible ones (sandy background, upper metal frame, outer image background)
# with respect to the casting surfaces area computation

# De-noise the pre-processed image according to the config file parameters
first = cv2.fastNlMeansDenoising(cropped.copy(), h=FRAME_PARAMS['h'],
                                 templateWindowSize=FRAME_PARAMS['template_window_size'],
                                 searchWindowSize=FRAME_PARAMS['search_window_size'])
# Convert the de-noised image to grayscale and smooth it by blurring
blurred = gray_and_blur(first, blur=True, ksize=7)
# Init a Segmentation class object with the config.py segmentation parameters
sgm = Segmentation(SEGMENTATION_PARAMS)
# Adjust brightness/contrast and threshold the blurred image based on the configuration settings
thresh = sgm.scale_abs_convert_thresh(blurred)
# Apply a sequence of morphological transformations to simplify the thresholded image
# in order to better detect contours
transformed = sgm.transform_morphology(
    thresh, TRANSFORMATION_SEQ, SEGMENTATION_PARAMS['morph_ops_params'])

# Show the segmentation step output image if required
if args["steps"]:
    print(f"Sequence of transformations: {TRANSFORMATION_SEQ}")
    view_image(
        np.hstack([thresh, transformed]),
        img_name=f"Thresholding | {TRANSFORMATION_SEQ[0].capitalize()} and \
        {TRANSFORMATION_SEQ[1].capitalize()}")


# ------ FRAME ROI DETECTION ------
# Detect the upper metal frame region of interest to perform the casting analysis later on

# Init a Frame class object with the config.py frame parameters
frame = Frame(FRAME_PARAMS)
# Detect the frame lines with the parameters passed through the configuration file
lines = frame.detect_frame_lines(cropped.copy(), params=FRAME_PARAMS, num_frame_lines=10)

# Print the number of detected lines if required
if args["steps"]:
    print(f"Number of lines detected: {len(lines)}")

# Init a new blank, that is black, image with the same shape as pre-processed's one
blank = np.zeros((cropped.shape[0], cropped.shape[1], 3), dtype=np.uint8)
# Draw the detected lines on the blank image and get a masked image for contour detection
line_src = frame.draw_hough_lines(lines, blank, cropped)
_, out = cv2.threshold(line_src, 50, 255, cv2.THRESH_BINARY)
# Compute the bounding boxes from the approximated detected contours and extract the box
# corresponding to the inner metal frame contour
bound_boxes = frame.get_bounding_boxes_from_contours(out)
drawn_boxes, sel_boxes = frame.select_n_largest_boxes(bound_boxes, out, num_boxes=1)

# Show the boxes detection output image if required
if args["steps"]:
    view_image(drawn_boxes, img_name="Computed inner frame bounding box")

# Compute the inner and outer frame rectangle coordinates based on frame parameters, and starting
# from the inner frame bounding box
frame_box, diff = frame.set_inner_frame_rect(sel_boxes, FRAME_PARAMS['mean_box_coords'])
inner_frame = frame.get_rect_coords_from_box(frame_box)
outer_frame = frame.compute_outer_frame_coords(inner_frame)
frame_area_coords = [inner_frame, outer_frame]
frame_area_coords = [item for items in frame_area_coords for item in items]
# Extract the metal frame region of interest starting from the previously segmented image
frame_roi = frame.get_frame_roi(transformed.copy(), frame_area_coords)

# Show the frame roi detection step output image if required
if args["steps"]:
    view_image(frame_roi, img_name="Frame ROI mask")


# ------ SAND ROI DETECTION ------
# Detect the sandy region of interest to perform the casting analysis later on

# Get the top left and bottom right vertices coordinates of the sand roi rectangle
# starting from the inner frame bounding box detected in the frame roi detection step
top_left, bottom_right = frame.get_rect_coords_from_box(frame_box)
sand_area_coords = (top_left, bottom_right)
# Init a Sand class object with the config.py frame parameters
snd = Sand(sand_area_coords)
# Extract the sand roi starting from the previously segmented image
sand_roi = snd.get_sand_roi(transformed.copy())

# Show the sand roi detection step output image if required
if args["steps"]:
    view_image(sand_roi, img_name="Sand ROI mask")


# ------ CASTING DETECTION ------
# Compute the casting surfaces total area plus other information to determine the mold condition

# Init a Casting class object
cst = Casting()
# Compute the casting surfaces total area both for the sand and the frame regions. A threshold can
# be used to remove the smaller areas by setting the size_thresh_in_px parameter by config.py
# Show the casting total area if required
if args["steps"]:
    SHOW_IMG = True

mold_cast_area = cst.compute_mold_casting_area(cropped, sand_roi, frame_roi, show=SHOW_IMG,
                                               size_thresh_in_px=SIZE_THRESHOLD_IN_PX)


# ------ MOLD STATE CHECK ------
# Use the collected values to check the inspected mold condition

# Compare the computed area to a threshold set in the configuration file. Check whether the frame
# is free of molten metal remains (frame roi area = 0), then get the number of issues detected
num_issues, frame_clean = cst.detect_mold_issues(mold_cast_area,
                                                 mold_casting_threshold=MOLD_CASTING_THRESHOLD)
# Impute the inspected mold condition based on the collected information
mold_state, warn_color, frame_clean = cst.check_mold_state(num_issues, frame_clean)


# ------ MOLD INFORMATION UPDATE ------
# Update the inspected mold information

# Update the mold information through a dictionary
inspected = cst.update_mold_state(INSPECTED, mold_state, warn_color, mold_cast_area, frame_clean)
# Save the mold information in a json file
with open('./output/data.txt', 'w') as outfile:
    json.dump(inspected, outfile)
# Uncomment these lines to check if json writing was successful
# with open('./output/data.txt') as json_file:
#     data = json.load(json_file)
#     print(data)

# Print the mold information dictionary update if required
if args["steps"]:
    print(f"Mold information update: {inspected}")

# Save a copy of the pre-processed step output image for checking
print(f"Saving image and information . . .")
filename = './output/output_image.png'
cv2.imwrite(filename, cropped)


# ------ FLASK ------
app = Flask(__name__, static_folder="output")

@app.route('/')
def index():
    with open('./output/data.txt') as json_file:
        data_dict = json.load(json_file)
        remap = {'mold_state': 'Mold state', 'warning_color': 'Warning color',
                 'mold_tot_area': 'Mold casting area [px]',
                 'sand_roi_tot_area': 'Sandy ROI casting area [px]',
                 'frame_roi_tot_area': 'Frame ROI casting area [px]',
                 'frame_clean': 'Frame clean from residues'}
        data_dict = dict((remap[key], value) for (key, value) in data_dict.items())
    return render_template('index.html', data=data_dict)

@app.after_request
def add_header(response):
    """Add headers to cache the rendered page for 10 minutes"""
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
