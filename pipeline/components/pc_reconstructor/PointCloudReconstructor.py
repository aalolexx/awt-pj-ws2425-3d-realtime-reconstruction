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
        self._threshold = 0.18
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

            # add bounding box to visualization
            min_bound = np.array([-1, -1, -1])
            max_bound = np.array([1, 1, 1])
            aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            aabb.color = (1, 0, 0)
            self.vis.add_geometry(aabb)


    #
    # Run Step
    #
    def run_step(self, pcd_incomplete):
        """Parse the pcd into a voxel grid and reconstruct it."""
        # Parse the PCD into a voxel grid
        normalized_pcd, scaling_factor = self.normalize_anti_isotropic(pcd_incomplete)
        input_tensor = self.pointcloud_to_tensor(normalized_pcd)

        self._rc_model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self._device)
            model_output_tensor = self._rc_model(input_tensor)

            # post processing
            maxima_tensor = self.max_pooling(model_output_tensor, input_tensor)
            #thresholded_point_cloud = self.construct_point_cloud_from_tensor(model_output_tensor)
            thresholded_point_cloud = self.construct_point_cloud_from_tensor(maxima_tensor)
            reconstructed_pcd = thresholded_point_cloud
            #reconstructed_pcd = self.reverse_scale_of_point_cloud(thresholded_point_cloud, scaling_factor)

        if self._visualize:
            self.visualize(reconstructed_pcd)

        return reconstructed_pcd


    #
    # normalizes point cloud to the borders of a (64x128x64) box
    #
    def normalize_anti_isotropic(self, pcd: o3d.geometry.PointCloud):
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
    # transforms pointcloud into tensor
    #
    def pointcloud_to_tensor(self, pcd):
        input_volume = np.zeros((64,128,64), dtype=np.uint8)
        if not pcd is None:
            if not pcd.is_empty():
                input_points = np.asarray(pcd.points, dtype=np.uint8)
                for (x, y, z) in input_points:
                    input_volume[x, y, z] = 1

        input_tensor = torch.tensor(input_volume, dtype=torch.float32)
        return input_tensor


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
    # Apply max pooling
    #
    def max_pooling(self, model_output, input_tensor):
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
    # Threshold tensor and construct point cloud from tensor
    #
    def construct_point_cloud_from_tensor(self, tensor):
        suppressed_numpy = tensor.squeeze().cpu().numpy()
        suppressed_thresholded = np.argwhere(suppressed_numpy > self._threshold)
        suppressed_points = np.array(suppressed_thresholded, dtype=np.float32)
        suppressed_point_cloud = o3d.geometry.PointCloud()
        suppressed_point_cloud.points = o3d.utility.Vector3dVector(suppressed_points)
        return suppressed_point_cloud


    #
    # Rescale point cloud to it's original position and scale
    # Optionally scale pointcloud into bounding box around the center
    #
    def reverse_scale_of_point_cloud(self, point_cloud, reverse_scale, should_scale_to_bounding_box=False):
        # reverse scale
        points = np.asarray(point_cloud.points)
        points = points - (32,64,32)
        points = points * reverse_scale
        rescaled_point_cloud = o3d.geometry.PointCloud()
        rescaled_point_cloud.points = o3d.utility.Vector3dVector(points)

        # calculate scale to transform pcd into bounding box (only for testing, remove in the final version)
        if should_scale_to_bounding_box:
            points = np.asarray(rescaled_point_cloud.points)
            min_bound = rescaled_point_cloud.get_min_bound()
            max_bound = rescaled_point_cloud.get_max_bound()
            extents = max_bound - min_bound
            max_scale = 2.0 / max(extents)
            bounding_box_scale = np.array([max_scale, max_scale, max_scale])
            points = points * bounding_box_scale
            rescaled_point_cloud.points = o3d.utility.Vector3dVector(points)

        return rescaled_point_cloud


    #
    # Visualize a 3D Tensor
    #
    def visualize(self, point_cloud):        
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