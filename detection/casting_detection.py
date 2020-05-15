#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This python script contains the class and methods for detecting the casting surfaces in the ROIs
    and assessing the mold condition"""

__author__ = 'martino lorusso'


# Import the necessary packages
from misc.visualization import *


class Casting:
    """
    A class for representing the casting process output and assessing its quality

    Casting methods implement the image processing techniques to analyze it so to determine the
    inspected mold condition and update the relative information.
    See OpenCV documentation for reference

    Methods
    -------
    detect_contours(src)
        Detect contours in the src image
    detect_contours_with_canny(src, canny_thresh, ratio, aperture)
        Detect contours in the src image with Canny algo
    get_sorted_contours_and_areas( contours, descending=True)
        Sort contours by area in ascending or descending order
    select_largest_contours_and_areas(contours_and_areas, size_thresh_in_px=0)
        Select all areas larger than the specified size threshold in pixels
    compute_total_area(contours_and_areas)
        Given a list of contours, compute the total area
    draw_contours(src, contours, hierarchy=None)
        Draw filled contours
    draw_contours_with_areas(src, contours_and_areas, show_single_contour=True, label='Detected')
        Draw filled contours showing the value of their total area
    compute_mold_casting_area(src, sand_roi, frame_roi, size_thresh_in_px=0)
        Compute the total casting area in the mold by adding up the total casting areas of each ROI
    detect_mold_issues(mold_casting_area, mold_casting_threshold)
        Given the total casting area in the mold and a size threshold, detect possible issues
    check_mold_state(num_issues_detected, frame_clean)
        Infer the condition of the inspected mold from the provided information
    update_mold_state(inspected, mold_state, warning_color, mold_casting_area, frame_clean)
        Update the information about the inspected mold state based on its condition check

    """
    def __init__(self):
        pass

    def detect_contours(self, src):
        """Detect contours in the source image

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image

        Returns
        -------
        contours : list
            A list of detected contours where each contour is stored as a vector of points
        hierarchy : list
            A vector containing information about the image topology

        """
        # Retrieve contours and hierarchy using the algorithm [Suzuki85] --> OpenCV doc. for ref.
        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy

    def detect_contours_with_canny(self, src, canny_thresh, multiplier, aperture):
        """"Apply Canny Edge Detector to source image, then detect contours in the resulting mask

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image
        canny_thresh : float
            First threshold for the hysteresis procedure
        multiplier : int
            A multiplier of canny_thresh to compute second threshold for the hysteresis procedure
        aperture : int
            Aperture size for the Sobel() operator --> OpenCV doc. for ref.

        Returns
        -------
        contours : list
            A list of detected contours where each contour is stored as a vector of points
        hierarchy : list
            A vector containing information about the image topology

        """
        # Apply the Canny algo using the provided parameters --> OpenCV doc. for ref.
        canny_output = cv2.Canny(src, canny_thresh, canny_thresh*multiplier, aperture)
        # Retrieve contours and hierarchy using the algorithm [Suzuki85] --> OpenCV doc. for ref.
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy

    def get_sorted_contours_and_areas(self, contours, descending=True):
        """Sort contours by area in ascending or descending order

        Parameters
        ----------
        contours :
            A list of detected contours where each contour is stored as a vector of points
        descending : bool
            Allow to sort the contours by area, from the largest to the smallest (default True)

        Returns
        -------
        sorted_contours_and_areas : zip object
            A zip object containing the list of sorted contours and the list of corresponding areas
        sorted_contours : list
            A list of contours sorted by area
        sorted_areas : list
            A list of sorted areas

        """
        # Use the area of each contour as a key for sorting the contours and get the sorted lists
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=descending)
        sorted_areas = [cv2.contourArea(cnt) for cnt in sorted_contours]
        # Create a zip object from the sorted lists
        sorted_contours_and_areas = zip(sorted_contours, sorted_areas)

        return sorted_contours_and_areas, sorted_contours, sorted_areas

    def select_largest_contours_and_areas(self, contours_and_areas, size_thresh_in_px=0):
        """Select all areas larger than the specified size threshold in pixels

        Parameters
        ----------
        contours_and_areas : zip object
            A zip object containing the list of contours and the list of corresponding areas
        size_thresh_in_px : int
            A threshold of size in pixels to compare the area of each contour to (default 0)

        Returns
        -------
        largest_contours_and_areas : zip object
            A zip object containing the list of largest contours and the list of corresponding areas

        """
        largest_contours = []
        largest_areas = []
        contours_and_areas = list(contours_and_areas)
        # Loop over the nested list of contours and respective areas
        for cnt in contours_and_areas:
            contour = cnt[0]
            area = cnt[1]
            # Compare the area of a contour to the threshold of size in pixels and if it is
            # greater than that, add the contour and the area to the lists storing the largest ones
            if area >= size_thresh_in_px:
                largest_contours.append(contour)
                largest_areas.append(area)

        largest_contours_and_areas = zip(largest_contours, largest_areas)

        return largest_contours_and_areas

    def compute_total_area(self, contours_and_areas):
        """Given a list of contours, compute the total area

        Parameters
        ----------
        contours_and_areas : zip object or list
             A zip object or a list containing the list of contours and the list of respective areas

        Returns
        -------
        total_area = float
            The total area computed by adding up the areas of all contours

        """
        if type(contours_and_areas) != list:
            contours_and_areas = list(contours_and_areas)

        total_area = sum([cnt[1] for cnt in contours_and_areas])

        return total_area

    def draw_contours(self, src, contours, hierarchy=None):
        """Draw filled contours

        Parameters
        ----------
        src : ndarray
            3D array of `int` type representing the source image
        contours : list
            A list of contours where each contour is stored as a vector of points
        hierarchy : list, optional
            A vector containing information about the image topology

        Returns
        -------
            None

        """
        dst = src.copy()
        # Loop over the list of contours
        for i in range(len(contours)):
            # Draw the  contour with a random color
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.drawContours(dst, contours, i, color, -1, cv2.LINE_8, hierarchy, 0)
        # Show all contours in a copy of source image
        cv2.imshow('Find contours', dst)
        cv2.waitKey()

    def draw_contours_with_areas(self, src, contours_and_areas, show=True,
                                 show_single_contour=True, label='Detected'):
        """Draw filled contours showing the value of their total area

        src : ndarray
            3D array of `int` type representing the source image
        contours_and_areas : zip object
            A zip object containing the list of contours and the list of respective areas
        show_single_contour : bool
            Allow to show the value of each contour area
        label : str
            Add a label to the shown image (default "Detected")

        Returns
        -------
        total_area : float
            The total area computed by adding up the areas of all contours

        """
        dst = src.copy()
        contours_and_areas = list(contours_and_areas)
        contours = [cnt[0] for cnt in contours_and_areas]
        # Given the list of contours and respective areas, compute the total area
        total_area = self.compute_total_area(contours_and_areas)
        # Loop over the list of contours and areas
        for cnt in contours_and_areas:
            contour = cnt[0]
            area = cnt[1]
            # Draw the contour along with the value of its area
            cv2.drawContours(dst, [contour], -1, (0, 255, 0), -1, 1)
            cv2.putText(dst, 'A=' + str(int(area)) + ' px', (180, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Show each filled contour, if requested
            if show_single_contour:
                cv2.imshow("Casting contour by area", dst)
                # if the 'q' key is pressed, stop the loop
                if cv2.waitKey() & 0xFF == ord("q"):
                    break
            dst = src.copy()

        # Draw all filled contours and the total area in pixels
        cv2.drawContours(dst, contours, -1, (0, 255, 0), -1, 1)
        res = reshape_img_for_outcome_visualization(dst)
        cv2.putText(res, 'Total A=' + str(int(total_area)) + ' px', (40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show all filled contours and the total area
        if show:
            cv2.imshow(f"{label} casting", res)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return total_area

    def compute_mold_casting_area(self, src, sand_roi, frame_roi, size_thresh_in_px=0, show=True):
        """Compute total casting area in the mold by adding up the total casting areas of each ROI

        Parameters
        ----------
        src : ndarray
        sand_roi : ndarray
            3D array of `int` type representing the image of the sand ROI
        frame_roi : ndarray
            3D array of `int` type representing the image of the frame ROI
        size_thresh_in_px : int
            A threshold of size in pixels to compare the area of each contour to (default 0)
        show : bool
            Allow to show the total casting area (default True)

        Returns
        -------
        mold_casting_area : int
            The total casting area in the mold

        """
        mold_casting_area = []
        labels = ['Sand', 'Frame']
        # Loop over the list of ROI images
        for idx, roi_src in enumerate([sand_roi, frame_roi]):
            # Detect filled contours in the ROI, select the contours and areas larger than the set
            # threshold, compute the total area and append it to the total casting area in the mold
            contours, hierarchy = self.detect_contours(roi_src.copy())
            sorted_contours_areas, _, _ = self.get_sorted_contours_and_areas(contours)
            largest_contours_areas = self.select_largest_contours_and_areas(sorted_contours_areas,
                                                                            size_thresh_in_px)
            tot_roi_area = self.draw_contours_with_areas(src.copy(),
                                                         largest_contours_areas,
                                                         show=show,
                                                         show_single_contour=False,
                                                         label=labels[idx])
            mold_casting_area.append(int(tot_roi_area))

        return mold_casting_area

    def detect_mold_issues(self, mold_casting_area, mold_casting_threshold):
        """Given the total casting area in the mold and a size threshold, detect possible issues in
        the inspected mold

        Parameters
        ----------
        mold_casting_area : int
            The total casting area in the mold
        mold_casting_threshold : int
            A threshold to compare the total casting area to

        Returns
        -------
        num_mold_issues : int
            The number of detected issues in the inspected mold
        frame_clean : bool
            An indicator of the absence (if True) of casting residuals on the frame (that occurs
            when frame_casting_area = 0)

        """
        num_mold_issues = 0
        frame_clean = True
        # Get the total casting area in the frame ROI
        frame_casting_area = mold_casting_area[1]
        mold_casting_area = int(sum(mold_casting_area))
        # Check if the total casting area in the mold is larger than the set threshold
        if mold_casting_area >= mold_casting_threshold:
            # Check if the total casting area in the mold exceeds more than 30% of the set threshold
            if mold_casting_area >= mold_casting_threshold * 1.30:
                num_mold_issues += 1
            num_mold_issues += 1
        # Check if any casting residuals are on the frame
        if frame_casting_area != 0:
            num_mold_issues += 1
            frame_clean = False

        return num_mold_issues, frame_clean

    def check_mold_state(self, num_issues_detected, frame_clean):
        """Infer the condition of the inspected mold from the provided information

        Parameters
        ----------
        num_issues_detected : int
            The number of detected issues in the inspected mold
        frame_clean : bool
            An indicator of the absence (if True) of casting residuals on the frame

        Returns
        -------
        mold_state : str
            The state of the currently inspected mold
        warning_color : str
            A color representing the level of warning
        frame_clean : bool
            An indicator of the absence (if True) of casting residuals on the frame

        """
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
        """Update the information about the inspected mold state based on its condition check

        Parameters
        ----------
        mold_state : str
            The state of the currently inspected mold
        warning_color : str
            A color representing the level of warning
        mold_casting_area : int
            The total casting area in the mold
        frame_clean : bool
            An indicator of the absence (if True) of casting residuals on the frame

        Returns
        -------
        inspected : dict
            A dictionary containing the inspected mold information

        """
        # Update inspected dictionary values
        inspected['mold_state'] = mold_state
        inspected['warning_color'] = warning_color
        inspected['mold_tot_area'] = sum(mold_casting_area)
        inspected['sand_roi_tot_area'] = mold_casting_area[0]
        inspected['frame_roi_tot_area'] = mold_casting_area[1]
        inspected['frame_clean'] = frame_clean

        return inspected
