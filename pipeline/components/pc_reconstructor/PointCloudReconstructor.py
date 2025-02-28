import torch
import os
import sys
import importlib
import open3d as o3d
import numpy as np
import torch.nn.functional as F

from util.base_module import BaseModule

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
sys.path.append(model_dir)  # contains ModelClasses.py

"""
Uses our custom made Models to reconstruct a incomplete point cloud (given from our PC_Generator)
"""
class PointCloudReconstructor(BaseModule):
    def __init__(self, model_name, checkpoint_name, visualize=False):
        """Initialize the PointCloudReconstructor."""
        self._threshold = 0.20
        self._visualize = visualize
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        classes_module = importlib.import_module("ModelClasses")
        sys.modules['ModelClasses'] = classes_module
        ModelClass = getattr(classes_module, model_name)
        self._rc_model = ModelClass()  # Pass required arguments if needed
        state_dict = torch.load(f'../models/{checkpoint_name}', map_location=self._device)
        self._rc_model.load_state_dict(state_dict)
        self._rc_model.to(self._device)

        if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
            self.is_point_cloud_created = False


    #
    # Run Step
    #
    def run_step(self, pcd_incomplete):
        """Parse the pcd into a voxel grid and reconstruct it."""
        normalized_pcd, scaling_factor = self.normalize(pcd_incomplete)
        input_tensor = self.pointcloud_to_tensor(normalized_pcd)

        self._rc_model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self._device)
            model_output_tensor = self._rc_model(input_tensor)

            # post processing
            maxima_tensor = self.max_pooling(model_output_tensor, input_tensor)
            thresholded_point_cloud = self.construct_point_cloud_from_tensor(maxima_tensor)
            reconstructed_pcd = thresholded_point_cloud

        if self._visualize:
            self.visualize(reconstructed_pcd)

        return reconstructed_pcd, scaling_factor


    #
    # Normalize
    #
    def normalize(self, pcd):
        """Normalize point cloud into bounding box of (64, 128, 64)"""
        if pcd is None:
            return pcd
        
        if pcd.is_empty():
            return pcd
        
        bounding_box = pcd.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        pcd.translate(-center)

        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        extents = max_bound - min_bound
        scale_factors = []
        
        for d in extents:
            scale_factors.append(2.0 / d)

        scale_factors = np.array(scale_factors)
        scale_full = scale_factors * (31.5, 63.5, 31.5)

        points = np.asarray(pcd.points)
        points = points * scale_full
        points = points + (32,64,32)
        pcd.points = o3d.utility.Vector3dVector(np.around(points, decimals=4))
        return pcd, 1 / scale_full


    #
    # Point Cloud to Tensor
    #
    def pointcloud_to_tensor(self, pcd):
        """Extracts all points from a point cloud and transforms it to a tensor"""
        input_volume = np.zeros((64,128,64), dtype=np.uint8)
        if not pcd is None:
            if not pcd.is_empty():
                input_points = np.asarray(pcd.points, dtype=np.uint8)
                for (x, y, z) in input_points:
                    input_volume[x, y, z] = 1

        input_tensor = torch.tensor(input_volume, dtype=torch.float32)
        return input_tensor


    #
    # Max pooling / Non-maxima suppresion
    #
    def max_pooling(self, model_output, input_tensor):
        """Suppress points which are not the maxima"""
        # pad model output
        x_padded = F.pad(model_output, pad=(0, 1, 0, 1, 0, 1), mode='constant', value=0)

        # extract maxima
        pooled = F.max_pool3d(x_padded, kernel_size=2, stride=1, padding=0)
        mask = (model_output == pooled)
        suppressed = model_output * mask.float()
        suppressed = suppressed + input_tensor

        # returned point cloud consisting of maximas
        return suppressed


    #
    # Point Cloud to Tensor
    #
    def construct_point_cloud_from_tensor(self, tensor):
        """Threshold tensor and construct point cloud from tensor"""
        suppressed_numpy = tensor.squeeze().cpu().numpy()
        suppressed_thresholded = np.argwhere(suppressed_numpy > self._threshold)
        suppressed_points = np.array(suppressed_thresholded, dtype=np.float32)
        suppressed_point_cloud = o3d.geometry.PointCloud()
        suppressed_point_cloud.points = o3d.utility.Vector3dVector(suppressed_points)
        return suppressed_point_cloud


    #
    # Visualize
    #
    def visualize(self, point_cloud):
        """Visualizereconstructed point cloud"""
        if not self.is_point_cloud_created:
            self.vis.add_geometry(point_cloud)
            self.is_point_cloud_created = True
            self.pcd_placeholder = point_cloud
        else:
            # Update points and colors of the existing point cloud
            self.pcd_placeholder.points = point_cloud.points
            self.pcd_placeholder.colors = point_cloud.colors

        # Display the frame
        self.vis.update_geometry(self.pcd_placeholder)
        self.vis.poll_events()
        self.vis.update_renderer()

