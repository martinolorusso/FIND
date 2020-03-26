__author__ = 'martino lorusso'

import numpy as np
import cv2
import imutils
from processing.image_processing import define_kernel
from misc import visualization as vis


class Segmentation:

    def __init__(self, sgm_params):
        self.alpha = sgm_params['alpha']
        self.beta = sgm_params['beta']
        self.threshold = sgm_params['threshold']
        self.morph_params = sgm_params['morph_ops_params']

    def scale_abs_convert_thresh(self, src):
        """Adjust the brightness/contrast of an image, through the alpha and beta
        parameters, then apply a thresholding operation."""
        alpha = self.alpha / 100
        beta = self.beta - 100
        res = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)
        ret, thresh = cv2.threshold(res, self.threshold, 255, cv2.THRESH_BINARY)

        return thresh

    def apply_morph_op(self, src, morph_op, morph_op_params):
        """Add description"""
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
        """Add description"""
        for morph_op in morph_ops_sequence:
            transformed = self.apply_morph_op(op_src, morph_op, morph_ops_params)
            op_src = transformed.copy()

        return op_src
