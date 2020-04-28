#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This python script contains the class and methods for image segmentation"""

__author__ = 'martino lorusso'

# Import the necessary packages
import cv2
from processing.image_processing import define_kernel


class Segmentation:
    """
    A class for representing am image segmentation process

    Allow to define the sequence of operations along with the needed parameters for segmenting an
    image, that is separating significant elements from negligible ones.
    See OpenCV documentation for reference

    Attributes
    ----------
    sgm_params : dict
        A dictionary containing the segmentation parameters

    Methods
    -------
    scale_abs_convert_thresh(src)
        Adjust the brightness/contrast of an image and apply a simple threshold
    apply_morph_op(src, morph_op, morph_op_params)
        Apply an operation to the source image among a set of morphology operations

    """

    def __init__(self, sgm_params):
        self.alpha = sgm_params['alpha']
        self.beta = sgm_params['beta']
        self.threshold = sgm_params['threshold']
        self.morph_params = sgm_params['morph_ops_params']

    def scale_abs_convert_thresh(self, src):
        """Adjust the brightness/contrast of an image and apply a simple threshold

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image

        Returns
        -------
        thresh : ndarray
            3D array of `int` type representing the contrasted and thresholded image

        """
        # Parameters alpha and beta control brightness/contrast values --> OpenCV doc. for ref.
        alpha = self.alpha / 100
        beta = self.beta - 100
        res = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)
        # Apply a simple threshold
        ret, thresh = cv2.threshold(res, self.threshold, 255, cv2.THRESH_BINARY)

        return thresh

    def apply_morph_op(self, src, morph_op, morph_op_params):
        """Apply an operation to the source image among a set of morphological operations

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image
        morph_op : str
            The morphological operation to be applied
        morph_op_params : dict
            A dictionary of parameters to pass to the chosen morphological operation

        Returns
        -------
        image : ndarray
            3D array of `int` type representing the transformed image

        """
        # Define the kernel shape and size --> OpenCV doc. for ref.
        kernel = define_kernel(shape=morph_op_params[morph_op]['kernel']['shape'],
                               ksize=morph_op_params[morph_op]['kernel']['ksize'])
        if morph_op == 'erode':
            image = cv2.erode(src,
                              kernel=kernel,
                              iterations=morph_op_params['erode']['iter'])
        elif morph_op == 'dilate':
            image = cv2.dilate(src,
                              kernel=kernel,
                              iterations=morph_op_params['dilate']['iter'])
        elif morph_op == 'opening':
            image = cv2.morphologyEx(src,
                                     op=cv2.MORPH_OPEN,
                                     kernel=kernel,
                                     iterations=morph_op_params['opening']['iter'])
        elif morph_op == 'closing':
            image = cv2.morphologyEx(src,
                                     op=cv2.MORPH_CLOSE,
                                     kernel=kernel,
                                     iterations=morph_op_params['closing']['iter'])
        else:
            return f"Unknown operation. Options are:" \
                   f"'erode', 'dilate', 'opening', 'closing'."

        return image

    def transform_morphology(self, op_src, morph_ops_sequence, morph_ops_params):
        """Apply a sequence of morphological operations to the source image

        Parameters
        ----------
        op_src : ndarray
            3D array of `int` type representing the source image
        morph_ops_sequence : list
            A list of operations (as strings) to be applied sequentially
        morph_ops_params : dict
            A dictionary of parameters for each operation in the sequence

        Returns
        -------


        """
        # Loop over the transformations list
        for morph_op in morph_ops_sequence:
            # Apply the morphological operation in the sequence
            transformed = self.apply_morph_op(op_src, morph_op, morph_ops_params)
            op_src = transformed.copy()

        return op_src
