import cv2
import os
import sys
from PIL import Image
import numpy as np


from util.base_module import BaseModule

class DepthSegmenter(BaseModule):
    def __init__(self, input_size=(512, 512), visualize=False):
        """Initialize the Foreground Extractor."""
        self._visualize = visualize


    def run_step(self, frame):
        """Predict foreground from input."""
        depth_copy = frame.copy()

        depth_copy = (depth_copy * 255).astype(np.uint8)
        depth_copy = cv2.normalize(depth_copy, None, 0, 255, cv2.NORM_MINMAX)

        # Get the edges
        gradient = cv2.morphologyEx(depth_copy, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, edges = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)

        # Get what is very likely the bg
        _, sure_fg = cv2.threshold(depth_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find floor and ceilings based on the assumption that they have a slow big gradient
        gradient_y = cv2.Sobel(depth_copy, cv2.CV_64F, 0, 1, ksize=5)
        gradient_y_abs = cv2.convertScaleAbs(gradient_y)
        floor_and_ceiling = cv2.inRange(gradient_y_abs, 35, 100)
        kernel = np.ones((15, 15), np.uint8)
        floor_and_ceiling = cv2.morphologyEx(floor_and_ceiling, cv2.MORPH_OPEN, kernel)

        # Remove edges that are not in the sure_fg
        edges = cv2.bitwise_and(edges, edges, mask=sure_fg)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # add border to the image because feet are often missing
        border_height = 1
        edges = cv2.copyMakeBorder(edges, 0, border_height, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        kernel = np.ones((15, 15), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        edges = cv2.ximgproc.thinning(edges)

        # Fill the edge image by iterating each line (To much will be filled at this point)
        filled_image = np.zeros_like(edges)
        for row in range(edges.shape[0]):
            edge_pixels = np.where(edges[row] == 255)[0]  # Find the columns where edges exist
            if len(edge_pixels) > 0:
                start, end = edge_pixels[0], edge_pixels[-1]  # Get the first and last edge pixel in the row
                filled_image[row, start:end + 1] = 255  # Fill the gap with white (255) between edges
        filled_image = filled_image[:filled_image.shape[0] - border_height, :]

        # Create general BG mask and remove in filled_image
        bg_total = cv2.bitwise_not(sure_fg)
        bg_total = cv2.bitwise_or(bg_total, floor_and_ceiling)
        final_mask = cv2.bitwise_and(filled_image, filled_image, mask=cv2.bitwise_not(bg_total))
        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # Finally, find the biggest shape that got filtered out
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_shape = np.zeros_like(final_mask)
        biggest_contour_idx = np.argmax([cv2.contourArea(c) for c in contours])
        cv2.drawContours(biggest_shape, contours, biggest_contour_idx, (255), thickness=cv2.FILLED)

        # remove all outlier shapes
        final_shape = cv2.bitwise_and(final_mask, final_mask, mask=biggest_shape)

        if self._visualize:
            cv2.imshow('foreground mask', final_shape)

        return final_shape

