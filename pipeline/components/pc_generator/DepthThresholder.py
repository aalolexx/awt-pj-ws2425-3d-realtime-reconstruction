import cv2
import os
import sys
from PIL import Image
import numpy as np


from util.base_module import BaseModule

class DepthThresholder(BaseModule):
    def __init__(self, input_size=(512, 512), visualize=False):
        """Initialize the Foreground Extractor."""
        self._visualize = visualize


    def run_step(self, frame):
        """Predict foreground from input."""
        depth_copy = frame.copy()

        _, mask = cv2.threshold(depth_copy, 0.6, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        centeroids = [np.mean(contour, axis=0) for contour in contours]
        #target_point = [int(mask.shape[1] / 2), int(mask.shape[0] / (3/2))]
        image_center = np.asarray(mask.shape) / 2

        most_central_centeroid_idx = np.argmin([np.linalg.norm(c - image_center) for c in centeroids])
        #biggest_contour_idx = np.argmax([cv2.contourArea(c) for c in contours])


        final_mask = np.zeros(mask.shape)
        cv2.drawContours(image=final_mask, contours=contours, contourIdx=most_central_centeroid_idx, color=(255),
                         thickness=cv2.FILLED)

        if self._visualize:
            cv2.imshow('foreground mask', final_mask)

        return final_mask

