import cv2
import torch
import os
import sys

from util.base_module import BaseModule

# Add vendor directory to path
vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vendor', 'Depth-Anything-V2'))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
sys.path.append(vendor_dir)

from depth_anything_v2.dpt import DepthAnythingV2

#
# DOWNLOAD the model from here:
# https://huggingface.co/depth-anything/Depth-Anything-V2-Small
#


class DepthEstimator(BaseModule):
    def __init__(self, input_size=480, visualize=False):
        """Initialize the DepthEstimator."""
        self._input_size = input_size
        self._visualize = visualize
        self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        device = 'cuda'
        if not torch.cuda.is_available():
            device = 'cpu'
            print('WARNING: CUDA is not available, using CPU instead.')
        
# Disable memory efficient attention to avoid xFormers issues
        torch.backends.cuda.enable_mem_efficient_sdp = False

        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'depth_anything_v2_vits.pth'), map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()

    def run_step(self, image):
        """Predict depth from input."""
        depth = self.model.infer_image(image, input_size=self._input_size)
        # normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        if self._visualize:
            cv2.imshow('depth', depth)

        return depth
