#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This python script contains some useful functions for basic image processing operations"""

__author__ = 'martino lorusso'


import numpy as np
import cv2
import imutils


def preprocess_img(src, cam_calib_path, rotation_angle, cropping_params):
    """Apply a sequence of step for the initial processing of a source image.
        See OpenCV documentation for reference


    Parameters
    ----------
    src : ndarray
        3D array of `int` type representing the source image
    cam_calib_path : str
        The path to the output files of camera calibration procedure
    rotation_angle : float
        The angle by which to rotate the undistorted image
    cropping_params : dict
        A dictionary containing the coordinates for cropping the undistorted and rotated image

    Returns
    -------
    cropped : ndarray
        3D array of `int` type representing the pre-processed image

    """
    # Load the camera calibration parameters from the camera calibration path
    mtx = np.load(cam_calib_path + 'mtx.npy')
    dist = np.load(cam_calib_path + 'dist.npy')
    newcameramtx = np.load(cam_calib_path + 'newcameramtx.npy')
    # Undistort the source image through the calibration parameters
    undist = cv2.undistort(src.copy(), mtx, dist, None, newcameramtx)
    # Rotate the undistorted image by a rotation angle
    rotated = imutils.rotate_bound(undist, rotation_angle)
    # Crop the rotated image using cropping parameters so to remove some annoying background
    up, down = cropping_params['bg_crop_up'], cropping_params['bg_crop_down']
    left, right = cropping_params['bg_crop_left'], cropping_params['bg_crop_right']
    cropped = rotated[up:down, left:right]

    return cropped

def gray_and_blur(src, blur=False, ksize=5):
    """Convert source image to grayscale and possibly apply a blur operation

    Parameters
    ----------
    src : ndarray
        3D array of `int` type representing the source image
    blur : bool
        Allow to blur the grayscale image (default False)
    ksize : int
        Set the size of kernel for blurring (default 5)

    Returns
    -------
    gray : ndarray
        2D array of `int` type representing the grayscale image

    """

    # Convert image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if blur:
        # Blur gray image using a Gaussian method
        gray = cv2.GaussianBlur(gray, ksize=(ksize, ksize), sigmaX=0)

    return gray

def equalize_hist(src, adaptive=True, cliplimit=2.0):
    """Equalize the histogram of a grayscale image. It improves the contrast in an image
    in order to stretch out the intensity range

    Parameters
    ----------
    src : ndarray
        2D array of `int` type representing a grayscale image
    adaptive : bool
        Use an adaptive histogram equalization (default True) --> OpenCV's CLAHE doc. for ref.
    cliplimit : float
        Set the magnitude of CLAHE (default 2.0)

    Returns
    -------
    equalized : ndarray
        2D array of `int` type representing an image with a better contrast

    """
    # Check if the image is grayscale
    if src.ndim == 3:
        raise ValueError("Warning: Source image must be grayscale!")
    # Choose the equalization method
    if adaptive:
        clahe = cv2.createCLAHE(clipLimit=cliplimit)
        equalized = clahe.apply(src)
    else:
        equalized = cv2.equalizeHist(src)

    return equalized

def define_kernel(shape, ksize):
    """Define the kernel shape and size for different kinds of processing operations

    Parameters
    ----------
    shape : str
        The shape of kernel --> OpenCV's getStructingElement() method doc. for ref.
    size : int
        The size of the kernel

    Returns
    -------
    kernel : ndarray (or tuple for 'default' shape)
        The defined kernel

    """
    if shape == 'rectangle':
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(ksize,ksize))
    elif shape == 'ellipse':
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(ksize, ksize))
    elif shape == 'cross':
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(ksize, ksize))
    # This shape returns a tuple representing a square matrix of `ksize` dimension
    elif shape == 'default':
        kernel = (ksize, ksize)
    else:
        return f"Unknown shape! Options are 'rectangle', 'ellipse', 'cross', 'default'."

    return kernel
