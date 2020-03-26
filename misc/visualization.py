__author__ = 'martino lorusso'

import os
import random

import numpy as np
import matplotlib.pyplot as plt

import cv2
import imutils


def load_a_random_image(img_path, img_type='jpg', show=False, resize=False, width=600):
    """Load a random image from the defined path, possibly resizing it."""
    img_list = os.listdir(img_path)
    just_img_list = []
    for img_name in img_list:
        if img_name.endswith('.' + img_type):
            just_img_list.append(img_name)

    img = just_img_list[random.randrange(len(just_img_list))]
    print(f"Loading image {img} . . .")
    image = cv2.imread(img_path + img)
    if resize:
        image = imutils.resize(image, width=width)
    if show:
        view_image(image, img_name="Chosen image", resize=False)

    return image

def view_image(image, img_name='Display', resize=False, width=600):
    """Visualize the loaded image, possibly resizing it."""
    if resize:
        image = imutils.resize(image, width=width)
    cv2.imshow(img_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def view_image_plt(image, figsize=None, img_name='Display',
                   color=False, cmap=None, resize=False, width=600):
    """Visualize the loaded image in ipynb, possibly resizing it."""
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = imutils.resize(image, width=width)
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(img_name)
    plt.show()

def show_hist(hist, title, color):
    """Shows the histogram using matplotlib capabilities."""
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)
    plt.show()

def reshape_img_for_outcome_visualization(src):
    res = np.zeros((src.shape[0] + 40, src.shape[1], 3), dtype=np.uint8)
    res[40:res.shape[0], :res.shape[1]] = src

    return res