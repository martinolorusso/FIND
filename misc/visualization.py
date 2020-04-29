#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This python script contains some useful functions for image visualization"""

__author__ = 'martino lorusso'


import os
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils


def load_a_random_image(img_path, img_type='jpg', show=False, resize=False, width=600):
    """Load a random image from the defined path, possibly resizing it

    img_path : str
        The path to the image to be loaded
    img_type : str
        The image type (default jpg)
    show : bool
        Allow to show the image (default False)
    resize : bool
        Allow to resize the image (default False)
    width : int
        If resizing, set the image new width (default 600) where the height is set automatically
        by keeping the same ratio as of the original image

    """
    # Get a list with the names of image files from the image path
    img_list = os.listdir(img_path)
    just_img_list = []
    for img_name in img_list:
        if img_name.endswith('.' + img_type):
            just_img_list.append(img_name)
    # Extract an image file name from the list
    img = just_img_list[random.randrange(len(just_img_list))]
    print(f"Loading image {img} . . .")
    # Load the chosen image
    image = cv2.imread(img_path + img)
    # Possibly resize the image
    if resize:
        image = imutils.resize(image, width=width)
    # Possibly show the image a new window
    if show:
        view_image(image, img_name="Chosen image", resize=False)

    return image

def view_image(image, img_name='Display', resize=False, width=600):
    """Visualize the loaded image, possibly resizing it

    Parameters
    ----------
    image : ndarray
        2D or 3D array of `int` type representing the image to show
    img_name : str
        The title of the window where the image will be displayed
    resize : bool
        Allow to resize the image (default False)
    width : int
        If resizing, set the image new width (default 600) where the height is set automatically
        by keeping the same ratio as of the original image

    Returns
    -------
    None

    """
    # Possibly resize the image
    if resize:
        image = imutils.resize(image, width=width)
    cv2.imshow(img_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def view_image_plt(image, figsize=None, img_name='Display',
                   color=False, cmap=None, resize=False, width=600):
    """Visualize the loaded image with matplotlib, possibly resizing it.
    See Matplotlib documentation for reference

    Parameters
    ----------
    image : ndarray
        2D or 3D array of `int` type representing the image to show
    figsize : tuple
        The size of the window where the image will be displayed, for instance (12,8)
    color : bool
        Allow to convert the image from grayscale color space to RGB color space (default False)
    cmap : bool
        Allow to set a colormap
    img_name : str
        The title of the window where the image will be displayed
    resize : bool
        Allow to resize the image (default False)
    width : int
        If resizing, set the image new width (default 600) where the height is set automatically
        by keeping the same ratio as of the original image

    Returns
    -------
    None

    """
    # Convert grayscale image to RGB color space
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Possibly resize the image
    if resize:
        image = imutils.resize(image, width=width)
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(img_name)
    plt.show()

def show_hist(hist, title, color):
    """Shows an histogram using matplotlib plot function. See Matplotlib documentation for reference

    Parameters
    ----------
    hist : array
        A 1D array containing the histogram values
    title : str
        The title of the window where the image will be displayed
    color : str
        The color of the histogram plot

    Returns
    -------
    None

    """
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)
    plt.show()

def reshape_img_for_outcome_visualization(src, px=40):
    """Reshape the source image for allowing to display annotations on it

    Parameters
    ----------
    src : ndarray
        3D array of `int` type representing the image to reshape
    px : int
        The size in pixels of which to extend the top edge of the image (default 40)

    Returns
    -------
    res : ndarray
        3D array of `int` type representing the image to reshape

    """
    # Increase the height of source image by px pixels
    res = np.zeros((src.shape[0] + px, src.shape[1], 3), dtype=np.uint8)
    res[px:res.shape[0], :res.shape[1]] = src

    return res
