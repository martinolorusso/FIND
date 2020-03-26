__author__ = 'martino lorusso'

import numpy as np
import cv2
import imutils


def preprocess_img(src, calib_cam_path, rotation_angle, cropping_params):
    """Add description"""
    # Loading the camera calibration parameters
    mtx = np.load(calib_cam_path + 'mtx.npy')
    dist = np.load(calib_cam_path + 'dist.npy')
    newcameramtx = np.load(calib_cam_path + 'newcameramtx.npy')
    # Undistorting the image through the calibration parameters
    undist = cv2.undistort(src.copy(), mtx, dist, None, newcameramtx)
    # Rotating the image by a rotation angle
    rotated = imutils.rotate_bound(undist, rotation_angle)
    # Cropping the image so to remove some annoying background
    up, down = cropping_params['bg_crop_up'], cropping_params['bg_crop_down']
    left, right = cropping_params['bg_crop_left'], cropping_params['bg_crop_right']
    cropped = rotated[up:down, left:right]

    return cropped

def gray_and_blur(src, blur=False, ksize=5):
    """Add description"""
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, ksize=(ksize, ksize), sigmaX=0)

    return gray

def equalize_hist(src, adaptive=True, cliplimit=2.0):
    """Add description"""
    if src.ndim == 3:
        raise ValueError("Warning: Source image must be grayscale!")
    if adaptive:
        clahe = cv2.createCLAHE(clipLimit=cliplimit)
        equalized = clahe.apply(src)
    else:
        equalized = cv2.equalizeHist(src)

    return equalized

def define_kernel(shape, ksize):
    """Add description"""
    if shape == 'rectangle':
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(ksize,ksize))
    elif shape == 'ellipse':
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(ksize, ksize))
    elif shape == 'cross':
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(ksize, ksize))
    elif shape == 'default':
        kernel = (ksize, ksize)
    else:
        return f"Unknown shape! Options are 'rectangle', 'ellipse', 'cross', 'default'."

    return kernel
