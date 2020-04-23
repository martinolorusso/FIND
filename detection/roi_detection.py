#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This python script contains the classes and methods for detecting the regions of interest (ROI)
    the analysis will be focused on."""

__author__ = 'martino lorusso'

# Import the necessary packages
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2


class Frame:
    """
    A class for representing the upper metal frame of a mold and implementing the image processing
    techniques to analyse it. Please, see OpenCV documentation for reference

    Attributes
    ----------
    frame_params : dict
        A dictionary of parameters to initialize a Frame object. Parameters are passed to class
        methods. The full list is available and editable in config.py

    Methods
    -------
    detect_frame_lines(src, params, num_frame_lines=10)
        Detect the frame image main lines
    get_bounding_boxes_from_contours(src)
        Compute the bounding boxes from contours
    select_n_largest_boxes(boxes, dst, num_boxes=4)
        Select the largest n boxes in a list of bounding boxes
    set_mean_box_coords_from_box_collection(box_collection)
        Compute a bounding box mean coordinates given a collection of boxes
    get_rectangle_sides_ratio(box_coords)
        Given a box, compute the ratio of its sides
    compute_percent_change(x, x_ref)
        Compute the percentage change between two numeric values
    get_rect_coords_from_box(box_coords)
        Compute top left and bottom right rectangle coordinates from a bounding box
    set_inner_frame_rect(box_coords, ref_box_coords, diff_thresh=2)
        Set the bounding box coordinates of the inner frame comparing the computed coordinates
        to a reference bounding box coordinates
    compute_outer_frame_coords(inner_frame_rect)
        Determine the outer frame rectangle coordinates given the frame thickness
    get_frame_roi(src, frame_coords, remove_holes=True)
        Extract the frame region of interest from the computed coordinates
    draw_hough_lines(lines, dst, src, show_val=False, show_lines=False, show_single=False)
        Draw the detected frame lines
    remove_frame_holes_from_mask(src, in_top_left, in_bottom_right, out_top_left, out_bottom_right)
        Remove the frame holes from the frame mask

    """

    def __init__(self, frame_params):
        self.frame_params = frame_params

    def detect_frame_lines(self, src, params, num_frame_lines=10):
        """Detect the frame image main lines

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image
        params : dict
            The dictionary containing the parameters requested by the image processing pipeline
            functions. Refer to OpenCV documentation for a thorough description of each one.
            Parameters need to be stored in config.py
        num_frame_lines : int
            The number of lines to be extracted among all the lines that are detected in order
            of likelihood (default 10). Look at OpenCV's HoughLines algo documentation for reference

        Returns
        -------
        lines : list
            A nested list of lines with length <= num_frame_lines parameter

        """
        preproc = src.copy()
        # Perform image denoising using Non-local Means Denoising algorithm --> OpenCV doc. for ref.
        den = cv2.fastNlMeansDenoising(preproc, h=params['h'],
                                       templateWindowSize=params['template_window_size'],
                                       searchWindowSize=params['search_window_size'])
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
        # Apply a bilateral filter to grayscale image. Bilateral filter can reduce unwanted noise
        # very well while keeping edges fairly sharp
        gray = cv2.bilateralFilter(gray, params['d'],
                                   params['sigma_color'],
                                   params['sigma_space'])
        # Apply an adaptive threshold
        # Switch comments of next two lines to choose between Gaussian_C and Mean_C method
        # ada_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        ada_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           blockSize=params['block_size'],
                                           C=params['C'])
        # Use the morphological erode and dilate operations with specified kernel sizes
        ada_thresh = cv2.erode(ada_thresh, kernel=(3, 3), iterations=1)
        ada_thresh = cv2.dilate(ada_thresh, kernel=(5, 5), iterations=1)
        img_to_detect_lines = ada_thresh.copy()

        # Set angle resolution and minimum line length for line detection
        angle_res = params['angle_res']
        min_line_length = params['min_line_length']
        # Apply HoughLines method to detect lines --> OpenCV doc. for ref.
        lines = cv2.HoughLines(img_to_detect_lines, 1, np.pi * angle_res / 180, min_line_length)
        if lines is not None:
            # Select the first num_frame_lines detected lines
            if len(lines) >= num_frame_lines:
                lines = lines[:num_frame_lines]

        return lines

    def get_bounding_boxes_from_contours(self, src):
        """Compute the bounding boxes from contours

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image

        Returns
        -------
        bound_boxes : list
            A nested list of the minimal bounding boxes for the specified contours

        """
        cnt_img = src.copy()
        # If source is a color image, convert it to grayscale
        if src.ndim == 3:
            cnt_img = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2GRAY)
        # Retrieve contours from the image using the algorithm [Suzuki85] --> OpenCV doc. for ref.
        contours, hierarchy = cv2.findContours(cnt_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Remove parent contour, drawn from hierarchy, that corresponds to outer frame bounding box
        contours = [contours[j] for j in range(len(contours)) if hierarchy[0][j][3] >= 0]
        # Initialize contour approximating polygons' and bounding boxes' lists
        contours_poly = [None] * len(contours)
        bound_boxes = [None] * len(contours)

        # Loop over the contours list
        for i, cnt in enumerate(contours):
            # Approximate the polygonal curves of the contour
            perimeter = cv2.arcLength(cnt, True)
            # Set approximation precision to 1.0% of perimeter
            contours_poly[i] = cv2.approxPolyDP(cnt, 0.010 * perimeter, True)
            # Compute a bounding rectangle from the approximated contour, as the tuple (x, y, w, h)
            # letting (x, y) be the top-left coordinate of the box rectangle and (w, h) be its
            # width and height
            bound_boxes[i] = cv2.boundingRect(contours_poly[i])

        return bound_boxes

    def select_n_largest_boxes(self, boxes, dst, num_boxes=4):
        """Select the n largest boxes in a list of bounding boxes

        Parameters
        ----------
        boxes : list
            A list of bounding boxes, each one defined by coordinates (x, y, w, h)
        dst : ndarray
            3D array of `int` type representing the destination image for drawing boxes
        num_boxes : int
            The number of boxes to be selected from all detected boxes (default 4)

        Returns
        -------
        drawn_boxes : ndarray
            3D array of `int` type representing the drawn boxes image
        selected_boxes : list
            A list of selected bounding boxes

        """
        drawn_boxes = dst.copy()

        box_areas = []
        # Loop over the boxes list
        for i in range(len(boxes)):
            # Compute the area of each box and then append it to box_areas list
            box_area = int(boxes[i][2]) * int(boxes[i][3])
            box_areas.append(box_area)
        # Sort the boxes by area size, from largest to smallest
        area_idx = [idx for idx, area in enumerate(box_areas)]
        areas = zip(area_idx, box_areas)
        sorted_areas = sorted(areas, key=lambda x: x[1], reverse=True)
        # Extract the n boxes with the largest area as tuples (x, y, w, h), letting (x, y) be the
        # top-left coordinate of the box rectangle and (w, h) be its width and height
        largest_n_areas = sorted_areas[:num_boxes]
        selected_boxes = [boxes[area[0]] for area in largest_n_areas][0]

        # Loop over the largest boxes list
        for area in largest_n_areas:
            # Draw box rectangles on the dst image
            i = area[0]
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.rectangle(drawn_boxes, (int(boxes[i][0]), int(boxes[i][1])),
                          (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])),
                          color, 3)

        return drawn_boxes, selected_boxes

    def set_mean_box_coords_from_box_collection(self, box_collection):
        """Compute a bounding box mean coordinates given a collection of boxes

        Parameters
        ----------
        box_collection : list
            A list of bounding boxes, each one defined by coordinates (x, y, w, h)

        Returns
        -------
        mean_box : tuple
            The bounding box computed by averaging the respective coordinates

        """
        # Get the mean bounding box by computing the average coordinates in a given collection
        # of boxes. Such box may be set as the default detected box for the most difficult molds
        mean_box = tuple(np.mean([box[0] for box in box_collection], axis=0, dtype='int'))

        return mean_box

    def get_rectangle_sides_ratio(self, box_coords):
        """Given a box, compute the ratio of its sides

        Parameters
        ----------
        box_coords : tuple
            A tuple representing a box coordinates (x, y, w, h)

        Returns
        -------
        box_ratio : float
            The ratio between the width (w) and height (h) of the box rectangle

        """
        width = box_coords[2]
        height = box_coords[3]
        box_ratio = round(width / height, 4)

        return box_ratio

    def compute_percent_change(self, x, x_ref):
        """Compute the percentage change between two numeric values

        Parameters
        ----------
        x : float
            New value
        x_ref : float
            Old or reference value

        Returns
        -------
        pct_change : float
            The percentage increase or decrease of the new value compared to the old value

        """
        pct_change = round(abs((x - x_ref) / x_ref) * 100, 2)

        return pct_change

    def get_rect_coords_from_box(self, box_coords):
        """Compute top left and bottom right rectangle coordinates from a bounding box coordinates

        Parameters
        ----------
        box_coords : tuple
            A tuple representing a box coordinates (x, y, w, h)

        Returns
        -------
        rect_from_box : list
            A list of top -left and bottom-right coordinates tuples of the corresponding rectangle

        """
        # Upper left vertex in OpenCV cv2.rectangle() method
        rect_start_point = (box_coords[0], box_coords[1])
        # Lower right vertex in OpenCV cv2.rectangle() method
        rect_end_point = (int(box_coords[0] + box_coords[2]), int(box_coords[1] + box_coords[3]))
        rect_from_box = [rect_start_point, rect_end_point]

        return rect_from_box

    def set_inner_frame_rect(self, box_coords, ref_box_coords, diff_thresh=2.0):
        """Set the bounding box coordinates of the inner frame by comparing the computed coordinates
        to a reference bounding box coordinates

        Parameters
        ----------
        box_coords : tuple
            A tuple representing a box coordinates (x, y, w, h)
        ref_box_coords : tuple
            A tuple representing a reference box coordinates (x, y, w, h)
        diff_thresh : float
            A difference threshold (in percent) between the sides ratio of the input box and
            the sides ratio of the reference box (default 2.0)

        Returns
        -------
        coords : tuple
            The coordinates of the chosen box
        diff_from_ratio : float
            The difference (in percent) between the sides ratio of the input (computed) box and
            the sides ratio of the reference box

        """
        # Get the sides ratio of both input box and reference box, compute their difference
        box_ratio = self.get_rectangle_sides_ratio(box_coords)
        ref_box_ratio = self.get_rectangle_sides_ratio(ref_box_coords)
        diff_from_ratio = self.compute_percent_change(box_ratio, ref_box_ratio)
        # Compare the computed difference to the difference threshold to set the inner frame box
        if diff_from_ratio < diff_thresh:
            coords = box_coords
        else:
            coords = ref_box_coords

        return coords, diff_from_ratio

    def compute_outer_frame_coords(self, inner_frame_rect):
        """Determine the outer frame rectangle coordinates given the frame thickness

        Parameters
        ----------
        inner_frame_rect : list
            A list of top -left and bottom-right coordinates tuples of the inner frame rectangle

        Returns
        -------
        outer_frame_rect : list
            A list of top -left and bottom-right coordinates tuples of the outer frame rectangle

        """
        pt1 = inner_frame_rect[0]
        pt2 = inner_frame_rect[1]
        # Given the frame thickness that is an attribute of Frame instance set in config.py,
        # compute the outer frame rectangle coordinates
        top_left = self.frame_params['frame_thickness'][0]
        bottom_right = self.frame_params['frame_thickness'][1]
        pt1_out = tuple(map(lambda x, y: x - y, pt1, top_left))
        pt2_out = tuple(map(lambda x, y: x + y, pt2, bottom_right))
        outer_frame_rect = [pt1_out, pt2_out]

        return outer_frame_rect

    def get_frame_roi(self, src, frame_coords, remove_holes=True):
        """Extract the frame region of interest from the computed coordinates

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the pre-processed image
        frame_coords : list
            A list of points representing the inner and outer frame rectangles coordinates
        remove_holes : bool
            Allow to remove the chassis holes from source as they could affect the casting analysis

        Returns
        -------
        frame_roi = ndarray
            3D array of `int` type representing the image of the frame ROI

        """
        # Frame_coords is made up of [pt1, pt2, pt1_out, pt2_out]
        in_top_left, in_bottom_right = frame_coords[0], frame_coords[1]
        out_top_left, out_bottom_right = frame_coords[2], frame_coords[3]
        blank = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        # Get a frame mask based on the inner and outer boxes coordinates
        cv2.rectangle(blank, out_top_left, out_bottom_right, (255, 255, 255), -1)
        cv2.rectangle(blank, in_top_left, in_bottom_right, (0, 0, 0), -1)

        # Remove frame holes, if requested
        if remove_holes:
            blank = self.remove_frame_holes_from_mask(blank, in_top_left, in_bottom_right,
                                                      out_top_left, out_bottom_right)
        # Use a bitwise AND operation to get the frame ROI image
        blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        frame_roi = cv2.bitwise_and(src, src, mask=blank)

        return frame_roi

    def draw_hough_lines(self, lines, dst, src,
                         show_val=False, show_lines=False, show_single=False):
        """Draw the detected frame lines

        Parameters
        ----------
        lines : list
            A nested list of lines
        dst : ndarray
            3D array of `int` type representing the destination image for drawing lines
        src : ndarray
            3D array of `int` type representing the source image for showing the original image
        show_val : bool
            Allow to show parametric coordinates values
        show_lines : bool
            Allow to show an image of the lines
        show_single : bool
            Allow to show the lines one at a time

        Returns
        -------
        dst : ndarray
            3D array of `int` type representing the destination image with the drawn lines

        """
        # Draw the lines detected through the Hough Line Transform --> OpenCV doc. for ref.
        if lines is not None:
            # Loop over lines list
            for i in range(0, len(lines)):
                # Extract rho and theta parametric coordinates from each line
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                # Show rho and theta values, if requested
                if show_val:
                    print("rho: ", rho, "\ttheta: ", theta)
                # Compute cartesian coordinates from parametric ones
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                # Draw the line on dst image
                cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                # Show the lines one at a time, if requested
                if show_single:
                    cv2.imshow("Detected Lines - Standard Hough Line Transform",
                               np.hstack([src, dst]))
                # Press `q` to stop the drawing
                if cv2.waitKey() & 0xFF == ord("q"):
                    break
            # Show the lines, if requested
            if show_lines:
                cv2.imshow("Frame", dst)
                cv2.waitKey()
                cv2.destroyAllWindows()

        return dst

    def remove_frame_holes_from_mask(self, src,
                                     in_top_left, in_bottom_right, out_top_left, out_bottom_right):
        """Remove the frame holes from the frame mask

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image
        in_top_left : tuple
            top-left coordinates of inner frame
        in_bottom_right : tuple
            bottom-right coordinates of inner frame
        out_top_left : tuple
            top-left coordinates of outer frame
        out_bottom_right : tuple
            bottom-right coordinates of outer frame

        Returns
        -------
        dst : ndarray
            3D array of `int` type representing the destination image with the drawn lines
        """
        dst = src.copy()
        # Get center of top and bottom frame holes
        pt1_x, pt1_y = (in_bottom_right[0]-in_top_left[0])//2, (in_top_left[1]-out_top_left[1])//2
        pt2_x, pt2_y = pt1_x, (out_bottom_right[1]-in_bottom_right[1])//2
        # Set holes radius based on the image size
        radius = src.shape[1] // 8
        # Draw two black filled circles on the destination image using above parameters
        cv2.circle(dst, (pt1_x, pt1_y), radius, (0, 0, 0), -1)
        cv2.circle(dst, (pt2_x, pt2_y), radius, (0, 0, 0), -1)

        return dst


class Sand:
    """
    A class for representing the central sandy area of a mold where the casting process occurs, and
    implementing the image processing techniques to analyse it

    Attributes
    ----------
    sand_area_coords : list
        A list of coordinate tuples that are obtained from frame roi detection step and are used to
        initialize a Sand object

    Methods
    -------
    get_sand_roi(src)
        Extract the sand region of interest from the coordinates computed in frame detection step

    """

    def __init__(self, sand_area_coords):
        # sand_area_coords is made up of [pt1, pt2]
        self.top_left = sand_area_coords[0]
        self.bottom_right = sand_area_coords[1]

    def get_sand_roi(self, src):
        """Extract the sand region of interest from the coordinates computed in frame detection step

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the pre-processed image

        Returns
        -------
        sand_roi : ndarray
            3D array of `int` type representing the image of the sand ROI

        """
        # Get the top-left and bottom-right ROI coordinate tuples from Sand instance attributes,
        # note that they correspond to the inner frame rectangle coordinates
        top_left = self.top_left
        bottom_right = self.bottom_right
        blank = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        # Get a sand mask based on the above coordinates
        cv2.rectangle(blank, top_left, bottom_right, (255, 255, 255), -1)
        mask = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        # Use a bitwise AND operation to get the sand ROI image
        sand_roi = cv2.bitwise_and(src, src, mask=mask)

        return sand_roi