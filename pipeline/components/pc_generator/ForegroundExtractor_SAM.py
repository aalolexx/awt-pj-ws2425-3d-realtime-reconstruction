import cv2
import os
import sys
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from transformers import SamModel, SamProcessor

from util.base_module import BaseModule

class ForegroundExtractor(BaseModule):
    def __init__(self, visualize=False):
        """Initialize the Foreground Extractor."""
        self._visualize = visualize

        self._model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
        self._processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


    def run_step(self, frame):
        """Predict foreground from input."""
        image = Image.fromarray(frame)
        image = image.resize((512, 512))

        points = [255, 255]  # point location of the center

        inputs = self._processor(image, input_points=[[[points]]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                                   inputs["original_sizes"].cpu(),
                                                                   inputs["reshaped_input_sizes"].cpu())
        masks_numpy = np.array(masks[0][0])
        combined_mask = np.logical_or.reduce(masks_numpy)

        mask_image = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        print(mask_image.shape)
        print(combined_mask.shape)
        mask_image[:, :] = np.where(combined_mask == 1, 255, 0)

        if self._visualize:
            cv2.imshow("Foreground Mask", mask_image)

        return mask_image
