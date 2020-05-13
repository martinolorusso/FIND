#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is a configuration file used to set the parameters requested by the classes and methods of
    the application"""

__author__ = 'martino lorusso'


# Loading paths
IPCAM_PATH = '/input_images/set/set_cropped/'
IPCAM_CALIBRATION_PATH = '/input_images/camera_calibration/'

# Pre-processing settings
# See classes and methods documentation of pre-processing step for reference
CROP_PARAMS = {'raw_crop_up': 164, 'raw_crop_down': 704, 'raw_crop_left': 4, 'raw_crop_right': 964,
               'bg_crop_up': 104, 'bg_crop_down': 454, 'bg_crop_left': 333, 'bg_crop_right': 663}
ROTATION_ANGLE = -3.5
BLUR_KERNEL = (5, 5)

# Segmentation settings
# See classes and methods documentation of segmentation step for reference
TRANSFORMATION_SEQ = ['erode', 'dilate']
SEGMENTATION_PARAMS = dict(morph_ops_params = {
    'erode': {'kernel': {'shape': 'default', 'ksize': 3}, 'iter': 1},
    'dilate': {'kernel': {'shape': 'default', 'ksize': 5}, 'iter': 2},
    'opening': {'kernel': {'shape': 'ellipse', 'ksize': 3}, 'iter': 1},
    'closing': {'kernel': {'shape': 'ellipse', 'ksize': 3}, 'iter': 1}
}, alpha=103, beta=1, threshold=130)

# Frame detection settings
# See classes and methods documentation of frame ROI detection step for reference
FRAME_PARAMS = dict(h=10, template_window_size=7, search_window_size=21,
                    d=18, sigma_color=30, sigma_space=30,
                    block_size=5, C=4,
                    angle_res=15, min_line_length=77,
                    mean_box_coords=(42, 40, 250, 276), mean_ratio=0.9000,
                    frame_thickness=[(18, 30), (18, 25)])

# Casting detection parameters
# See classes and methods documentation of sand ROI detection step for reference
MOLD_CASTING_THRESHOLD = 1500
SIZE_THRESHOLD_IN_PX = 20
INSPECTED = {'mold_state': 'Mold is ok', 'warning_color': 'green', 'mold_tot_area': 0,
             'sand_roi_tot_area': 0, 'frame_roi_tot_area': 0, 'frame_clean': True}
