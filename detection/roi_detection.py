__author__ = 'martino lorusso'

import os
import random
import time
import math
from collections import defaultdict
import argparse
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

import cv2
import imutils
from skimage import io
from sklearn.cluster import KMeans

from misc import visualization as vis
from processing.segmentation import Segmentation as Seg
from processing.image_processing import gray_and_blur, equalize_hist, define_kernel


class Frame:

    def __init__(self, frame_params):
        self.frame_params = frame_params

    def detect_frame_lines(self, src, params, num_frame_lines=10):
        """Add description"""
        preproc = src.copy()
        den = cv2.fastNlMeansDenoising(preproc, h=params['h'],
                                       templateWindowSize=params['template_window_size'],
                                       searchWindowSize=params['search_window_size'])
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, params['d'],
                                   params['sigma_color'],
                                   params['sigma_space'])
        # ada_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        ada_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           blockSize=params['block_size'],
                                           C=params['C'])
        ada_thresh = cv2.erode(ada_thresh, kernel=(3, 3), iterations=1)
        ada_thresh = cv2.dilate(ada_thresh, kernel=(5, 5), iterations=1)
        img_to_detect_lines = ada_thresh.copy()

        # Setting angle resolution and minimum line length for line detection
        angle_res = params['angle_res']
        min_line_length = params['min_line_length']
        # Applying HoughLines method
        lines = cv2.HoughLines(img_to_detect_lines, 1, np.pi * angle_res / 180, min_line_length)
        if lines is not None:
            # Select the first num_frame_lines detected lines
            if len(lines) >= num_frame_lines:
                lines = lines[:num_frame_lines]

        return lines

    def get_bounding_boxes_from_contours(self, src):
        """Add description"""
        cnt_img = src.copy()
        if src.ndim == 3:
            cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(cnt_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Removing parent contour that corresponds to outer frame bounding box
        contours = [contours[j] for j in range(len(contours)) if hierarchy[0][j][3] >= 0]
        # Initializing contour approximating polygons' and bounding boxes' lists
        contours_poly = [None] * len(contours)
        bound_boxes = [None] * len(contours)

        for i, cnt in enumerate(contours):
            # Approximating the polygonal curves of the contour
            perimeter = cv2.arcLength(cnt, True)
            # Setting approximation precision to 1.0% of perimeter
            contours_poly[i] = cv2.approxPolyDP(cnt, 0.010 * perimeter, True)
            # Computing a bounding rectangle from the approximated contour
            bound_boxes[i] = cv2.boundingRect(contours_poly[i])

        return bound_boxes

    def select_n_largest_boxes(self, boxes, dst, num_boxes=4):
        """Add description"""
        drawn_boxes = dst.copy()
        box_areas = []
        for i in range(len(boxes)):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            box_area = int(boxes[i][2]) * int(boxes[i][3])
            box_areas.append(box_area)

        area_idx = [idx for idx, area in enumerate(box_areas)]
        areas = zip(area_idx, box_areas)
        sorted_areas = sorted(areas, key=lambda x: x[1], reverse=True)
        largest_n_areas = sorted_areas[:num_boxes]
        selected_boxes = [boxes[area[0]] for area in largest_n_areas][0]

        for area in largest_n_areas:
            i = area[0]
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.rectangle(drawn_boxes, (int(boxes[i][0]), int(boxes[i][1])),
                          (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])),
                          color, 3)

        return drawn_boxes, selected_boxes

    def set_mean_box_coords_from_box_collection(self, box_collection):
        """Add description"""
        mean_box = tuple(np.mean([box[0] for box in box_collection], axis=0, dtype='int'))

        return mean_box

    def get_rectangle_sides_ratio(self, box_coords):
        """Add description"""
        width = box_coords[2]
        height = box_coords[3]
        box_ratio = round(width / height, 4)

        return box_ratio

    def compute_percent_change(self, x, x_ref):
        """Add description"""
        pct_change = round(abs((x - x_ref) / x_ref) * 100, 2)

        return pct_change

    def get_rect_coords_from_box(self, box_coords):
        """Add description"""
        # Upper left vertex in opencv cv2.rectangle
        rect_start_point = (box_coords[0], box_coords[1])
        # Lower right vertex in opencv cv2.rectangle
        rect_end_point = (int(box_coords[0] + box_coords[2]), int(box_coords[1] + box_coords[3]))
        rect_from_box = [rect_start_point, rect_end_point]

        return rect_from_box

    def set_inner_frame_rect(self, box_coords, ref_box_coords, diff_thresh=2):
        """Add description"""
        box_ratio = self.get_rectangle_sides_ratio(box_coords)
        ref_box_ratio = self.get_rectangle_sides_ratio(ref_box_coords)
        diff_from_ratio = self.compute_percent_change(box_ratio, ref_box_ratio)

        if diff_from_ratio < diff_thresh:
            coords = box_coords
        else:
            coords = ref_box_coords

        return coords, diff_from_ratio

    def compute_outer_frame_coords(self, inner_frame_rect):
        """Add description"""
        pt1 = inner_frame_rect[0]
        pt2 = inner_frame_rect[1]
        top_left = self.frame_params['frame_thickness'][0]
        bottom_right = self.frame_params['frame_thickness'][1]
        pt1_out = tuple(map(lambda x, y: x - y, pt1, top_left))
        pt2_out = tuple(map(lambda x, y: x + y, pt2, bottom_right))
        outer_frame_rect = [pt1_out, pt2_out]

        return outer_frame_rect

    def get_frame_roi(self, src, frame_coords, remove_holes=True):
        """Add description"""
        # frame_coords is made up of [pt1, pt2, pt1_out, pt2_out]
        in_top_left, in_bottom_right = frame_coords[0], frame_coords[1]
        out_top_left, out_bottom_right = frame_coords[2], frame_coords[3]
        blank = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(blank, out_top_left, out_bottom_right, (255, 255, 255), -1)
        cv2.rectangle(blank, in_top_left, in_bottom_right, (0, 0, 0), -1)
        if remove_holes:
            blank = self.remove_frame_holes_from_mask(blank, in_top_left, in_bottom_right,
                                                  out_top_left, out_bottom_right)
        blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        frame_roi = cv2.bitwise_and(src, src, mask=blank)

        return frame_roi

    def draw_hough_lines(self, lines, dst, src,
                         show_val=False, show_lines=False, show_single=False):
        """Draw the lines detected through the Hough Line Transform algo."""
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                if show_val:
                    print("rho: ", rho, "\ttheta: ", theta)
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                if show_single:
                    cv2.imshow("Detected Lines - Standard Hough Line Transform",
                               np.hstack([src, dst]))
                if cv2.waitKey() & 0xFF == ord("q"):
                    break

            if show_lines:
                cv2.imshow("Frame", dst)
                cv2.waitKey()
                cv2.destroyAllWindows()

        return dst

    def remove_frame_holes_from_mask(self, src,
                                     in_top_left, in_bottom_right, out_top_left, out_bottom_right):
        dst = src.copy()
        pt1_x, pt1_y = (in_bottom_right[0]-in_top_left[0])//2, (in_top_left[1]-out_top_left[1])//2
        pt2_x, pt2_y = pt1_x, (out_bottom_right[1]-in_bottom_right[1])//2
        radius = src.shape[1] // 8
        cv2.circle(dst, (pt1_x, pt1_y), radius, (0, 0, 0), -1)
        cv2.circle(dst, (pt2_x, pt2_y), radius, (0, 0, 0), -1)

        return dst


class Sand:

    def __init__(self, sand_area_coords):
        self.top_left = sand_area_coords[0]
        self.bottom_right = sand_area_coords[1]

    def get_sand_roi(self, src):
        """Add description"""
        # sand_area_coords is made up of [pt1, pt2]
        top_left = self.top_left
        bottom_right = self.bottom_right
        blank = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(blank, top_left, bottom_right, (255, 255, 255), -1)
        mask = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        sand_roi = cv2.bitwise_and(src, src, mask=mask)

        return sand_roi