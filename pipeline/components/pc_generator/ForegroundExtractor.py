import cv2
import os
import sys
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from util.base_module import BaseModule

class ForegroundExtractor(BaseModule):
    def __init__(self, visualize=False):
        """Initialize the Foreground Extractor."""
        self._visualize = visualize

        self._rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'high'][0])
        self._rmbg_model.to('cuda')
        self._rmbg_model.eval()


    def run_step(self, image):
        """Predict foreground from input."""
        image_size = (512, 512)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        depth_pillow = Image.fromarray(image)
        input_images = transform_image(depth_pillow).unsqueeze(0).to('cuda')

        with torch.no_grad():
            preds = self._rmbg_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        mask_np = pred.numpy()
        mask = np.where(mask_np > 0.1, 255, 0).astype(np.uint8)

        if self._visualize:
            cv2.imshow("mask", cv2.merge((mask, mask, mask)))

        return mask
