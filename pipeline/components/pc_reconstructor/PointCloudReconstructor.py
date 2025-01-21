import torch
import os
import sys
import importlib
import open3d as o3d
import numpy as np

from util.base_module import BaseModule

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
sys.path.append(model_dir)  # contains ModelClasses.py


"""
Uses our custom made Models to reconstruct a incomplete point cloud (given from our PC_Generator)
"""
class PointCloudReconstructor(BaseModule):
    def __init__(self, model_name, checkpoint_name, visualize=False):
        """Initialize the PointCloudReconstructor."""
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
        # Parse the PCD into a voxel grid
        input_tensor = self.get_3d_tensor_from_pcd(pcd_incomplete).to(self._device)

        self._rc_model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch dimension + channel
            reconstructed_tensor = self._rc_model(input_tensor)
            reconstructed_tensor = reconstructed_tensor.squeeze(0).squeeze(0).cpu()

        if self._visualize:
            self.visualize(reconstructed_tensor)

        return reconstructed_tensor


    #
    # parse the pcd into a voxel tensor
    #
    def get_3d_tensor_from_pcd(self, pcd):
        points = np.asarray(pcd.points)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        grid_size = 32  # TODO IN PARAMS
        voxel_size = (max_bound - min_bound) / grid_size

        normalized_points = (points - min_bound) / voxel_size
        grid_points = np.floor(normalized_points).astype(int)
        grid_points = np.clip(grid_points, 0, grid_size - 1)
        grid_tensor = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.int32)
        for point in grid_points:
            grid_tensor[tuple(point)] = 1

        return grid_tensor.float()


    #
    # Visualize a 3D Tensor
    #
    def visualize(self, voxel_tensor, threshold=0.5):
        voxel_tensor = voxel_tensor.cpu()
        normalized_tensor = torch.where(voxel_tensor > threshold, 1, 0)
        occupied_indices = np.argwhere(normalized_tensor.numpy() > 0)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(occupied_indices)

        if not self.is_point_cloud_created:
            self.vis.add_geometry(point_cloud)
            self.is_point_cloud_created = True
            self.pcd_placeholder = point_cloud  # TODO aeh is the placeholder needed?
        else:
            # Update points and colors of the existing point cloud
            self.pcd_placeholder.points = point_cloud.points
            self.pcd_placeholder.colors = point_cloud.colors

        # Display the frame
        self.vis.update_geometry(self.pcd_placeholder)
        self.vis.poll_events()
        self.vis.update_renderer()