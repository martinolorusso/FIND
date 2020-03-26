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

from misc.visualization import *
from processing.segmentation import Segmentation as Seg
from processing.image_processing import gray_and_blur, equalize_hist, define_kernel

class Casting:

    def __init__(self):
        pass

    def detect_contours(self, src):
        """Detect contours. Used to spot the molten metal castings."""
        # Get contours and hierarchy
        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy

    def detect_contours_with_canny(self, src, canny_thresh, ratio, aperture):
        """Detect contours with Canny algo. Used to spot the molten metal castings."""
        canny_output = cv2.Canny(src, canny_thresh, canny_thresh*ratio, aperture)
        # Get contours and hierarchy
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy

    def get_sorted_contours_and_areas(self, contours, descending=True):
        """Sort contours by area in ascending or descending order."""
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=descending)
        sorted_areas = [cv2.contourArea(cnt) for cnt in sorted_contours]
        sorted_contours_and_areas = zip(sorted_contours, sorted_areas)

        return sorted_contours_and_areas, sorted_contours, sorted_areas

    def select_largest_contours_and_areas(self, contours_and_areas, size_thresh_in_px=0):
        """Select all areas larger than the specified size threshold in pixels."""
        largest_contours = []
        largest_areas = []
        contours_and_areas = list(contours_and_areas)
        for cnt in contours_and_areas:
            contour = cnt[0]
            area = cnt[1]
            if area >= size_thresh_in_px:
                largest_contours.append(contour)
                largest_areas.append(area)

        return zip(largest_contours, largest_areas)

    def compute_total_area(self, contours_and_areas):
        """Compute contours total area."""
        if type(contours_and_areas) != list:
            contours_and_areas = list(contours_and_areas)

        total_area = sum([cnt[1] for cnt in contours_and_areas])

        return total_area

    def draw_contours(self, src, contours, hierarchy=None):
        """Draw contours."""
        dst = src.copy()
        for i in range(len(contours)):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.drawContours(dst, contours, i, color, -1, cv2.LINE_8, hierarchy, 0)
        cv2.imshow('Find contours', dst)
        cv2.waitKey()

    def draw_contours_with_areas(self, src, contours_and_areas,
                                 show_single_contour=True, label='Detected'):
        """Draw contours by."""
        dst = src.copy()
        contours_and_areas = list(contours_and_areas)
        contours = [cnt[0] for cnt in contours_and_areas]
        total_area = self.compute_total_area(contours_and_areas)
        for cnt in contours_and_areas:
            contour = cnt[0]
            area = cnt[1]
            # print(contour)
            # Draw contour area
            cv2.drawContours(dst, [contour], -1, (0, 255, 0), -1, 1)
            cv2.putText(dst, 'A=' + str(int(area)) + ' px', (180, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Show each contour area
            if show_single_contour:
                cv2.imshow("Casting contour by area", dst)
                # if the 'q' key is pressed, stop the loop
                if cv2.waitKey() & 0xFF == ord("q"):
                    break
            dst = src.copy()

        # Show all contour areas and total area in pixels
        cv2.drawContours(dst, contours, -1, (0, 255, 0), -1, 1)
        res = reshape_img_for_outcome_visualization(dst)
        cv2.putText(res, 'Total A=' + str(int(total_area)) + ' px', (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"{label} castings", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return total_area

    def compute_mold_casting_area(self, src, sand_roi, frame_roi, size_thresh_in_px=0):
        """Add description"""
        mold_casting_area = []
        labels = ['Sand', 'Frame']
        for idx, roi_src in enumerate([sand_roi, frame_roi]):
            contours, hierarchy = self.detect_contours(roi_src.copy())
            sorted_contours_areas, _, _ = self.get_sorted_contours_and_areas(contours)
            largest_contours_areas = self.select_largest_contours_and_areas(sorted_contours_areas,
                                                                            size_thresh_in_px)
            tot_roi_area = self.draw_contours_with_areas(src.copy(),
                                                        largest_contours_areas,
                                                        show_single_contour=False,
                                                        label=labels[idx])
            mold_casting_area.append(int(tot_roi_area))

        return mold_casting_area

    def detect_mold_issues(self, mold_casting_area, mold_casting_threshold):
        """Add description"""
        num_mold_issues = 0
        frame_clean = True
        frame_casting_area = mold_casting_area[1]
        mold_casting_area = int(sum(mold_casting_area))
        # if mold casting total area is larger than the set threshold
        if mold_casting_area >= mold_casting_threshold:
            # if casting area exceeds more than 30% of set threshold
            if mold_casting_area >= mold_casting_threshold * 1.30:
                num_mold_issues += 1
            num_mold_issues += 1
        # if frame area is not free from casting residues
        if frame_casting_area != 0:
            num_mold_issues += 1
            frame_clean = False

        return num_mold_issues, frame_clean

    def check_mold_state(self, num_issues_detected, frame_clean):
        """Add description"""
        if num_issues_detected == 0:
            mold_state = 'Mold is ok'
            warning_color = 'green'
        elif num_issues_detected == 1:
            mold_state = 'Warning: Mold may need to be inspected!'
            warning_color = 'yellow'
        else:
            mold_state = 'Warning: Mold needs to be inspected now!'
            warning_color = 'red'

        return mold_state, warning_color, frame_clean

    def update_mold_state(self, inspected,
                          mold_state, warning_color, mold_casting_area, frame_clean):
        """Add description"""
        inspected['mold_state'] = mold_state
        inspected['warning_color'] = warning_color
        inspected['mold_tot_area'] = sum(mold_casting_area)
        inspected['sand_roi_tot_area'] = mold_casting_area[0]
        inspected['frame_roi_tot_area'] = mold_casting_area[1]
        inspected['frame_clean'] = frame_clean

        return inspected



