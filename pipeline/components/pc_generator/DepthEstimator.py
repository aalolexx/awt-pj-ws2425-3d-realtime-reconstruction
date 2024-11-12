import cv2
import torch
import os
import sys

# Add vendor directory to path
vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vendor', 'Depth-Anything-V2'))
print(vendor_dir)
sys.path.append(vendor_dir)

from depth_anything_v2.dpt import DepthAnythingV2

#
# DOWNLOAD the model from here:
# https://huggingface.co/depth-anything/Depth-Anything-V2-Small
#

class DepthEstimator:
    def __init__(self):
        """Initialize the DepthEstimator."""
        self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load('../../../models/depth_anything_v2_vits.pth', map_location='cpu'))
        self.model.eval()
        
    def predict(self, image):
        """Predict depth from input."""
        depth = self.model.infer_image(image) # HxW raw depth map
        return depth
