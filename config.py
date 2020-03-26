"""
Configuration file to set the processing pipeline parameters
"""

# Loading paths
# Change theese two paths to your own local paths 
IPCAM_PATH = '/home/martino/git/FIND/input_images/test_set_01/'
CALIBRATION_CAM_PATH = '/home/martino/git/FIND/input_images/camera_calibration/'

# Pre-processing settings
CROP_PARAMS = {'raw_crop_up': 164, 'raw_crop_down': 704, 'raw_crop_left': 4, 'raw_crop_right': 964,
               'bg_crop_up': 104, 'bg_crop_down': 454, 'bg_crop_left': 333, 'bg_crop_right': 663}
ROTATION_ANGLE = -3.5
BLUR_KERNEL = (5, 5)

# Frame detection settings
FRAME_PARAMS_OLD = {"v_rho_min": 35, "v_rho_max": 295, "v_theta_min": 0, "v_theta_max": 3,
                    "h_rho_min": 35, "h_rho_max": 335, "h_theta_min": 87, "h_theta_max": 93}

FRAME_PARAMS = dict(h=10, template_window_size=7, search_window_size=21,
                    d=18, sigma_color=30, sigma_space=30,
                    block_size=5, C=4,
                    angle_res=15, min_line_length=77,
                    mean_box_coords=(42, 40, 250, 276), mean_ratio=0.9000,
                    frame_thickness=[(18, 30), (18, 25)])

# Segmentation settings
TRANSFORMATION_SEQUENCE = ['erode', 'dilate']

SEGMENTATION_PARAMS = dict(morph_ops_params = {
    'erode': {'kernel': {'shape': 'default', 'ksize': 3}, 'iter': 1},
    'dilate': {'kernel': {'shape': 'default', 'ksize': 5}, 'iter': 2},
    'opening': {'kernel': {'shape': 'ellipse', 'ksize': 3}, 'iter': 1},
    'closing': {'kernel': {'shape': 'ellipse', 'ksize': 3}, 'iter': 1}
}, alpha=103, beta=1, threshold=130)

# Casting detection parameters
MOLD_CASTING_THRESHOLD = 1500
SIZE_THRESHOLD_IN_PX = 20
INSPECTED = {'mold_state': 'Mold is ok', 'warning_color': 'green', 'mold_tot_area': 0,
             'sand_roi_tot_area': 0, 'frame_roi_tot_area': 0, 'frame_clean': True}

