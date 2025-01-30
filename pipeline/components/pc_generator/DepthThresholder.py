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
        #depth_map = depth_image.copy()

        #scale_factor = 0.5  # Adjust this value as needed
        #new_size = (int(depth_map.shape[1] * scale_factor), int(depth_map.shape[0] * scale_factor))
        #depth_map = cv2.resize(depth_map, new_size)

        # Extract the foreground object
        mask = self.extract_foreground_with_grabcut(frame)

        cv2.imwrite('frame.jpg', frame)
        cv2.imwrite('mask.jpg', mask)

        if self._visualize:
            cv2.imshow('foreground mask', mask)

        return mask


    def extract_foreground_with_grabcut(self, frame):
        # Ensure the depth map is a NumPy array and normalize it for visualization
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert the single-channel depth map to a 3-channel image
        #frame_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Initialize the mask
        mask = np.ones(frame.shape[:2], np.uint8) * 2  # Initialize all pixels as probable background

        # Assume the center region is more likely to be foreground
        height, width = frame.shape[:2]
        rect = (width // 4, height // 4, width // 2, height // 2)
        mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 3  # Mark probable foreground

        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Apply the GrabCut algorithm
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Convert mask to binary: 1 (foreground), 0 (background)
        mask_foreground = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        return mask_foreground

