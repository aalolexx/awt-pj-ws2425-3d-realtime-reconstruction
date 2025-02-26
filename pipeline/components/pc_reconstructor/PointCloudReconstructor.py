import torch
import os
import sys
import importlib
import open3d as o3d
import numpy as np
import torch.nn.functional as F
import copy
import time

from util.base_module import BaseModule


model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
sys.path.append(model_dir)  # contains ModelClasses.py

"""
Uses our custom made Models to reconstruct a incomplete point cloud (given from our PC_Generator)
"""
class PointCloudReconstructor(BaseModule):
    def __init__(self, model_name, checkpoint_name, visualize=False):
        """Initialize the PointCloudReconstructor."""
        self._threshold = 0.25
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
            #self.vis.add_geometry(aabb)


    #
    # Run Step
    #
    def run_step(self, pcd_incomplete):
        """Parse the pcd into a voxel grid and reconstruct it."""
        # Parse the PCD into a voxel grid
        #return pcd_incomplete, np.array([1,1,1])
        normalized_pcd, scaling_factor = self.normalize_anti_isotropic(pcd_incomplete)
        input_tensor = self.pointcloud_to_tensor(normalized_pcd)

        self._rc_model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self._device)

            model_output_tensor = self._rc_model(input_tensor)

            # post processing
            #maxima_tensor = self.max_pooling(model_output_tensor, input_tensor)

            #maxima_tensor = self.nms_1d_axes_3d_torch(model_output_tensor)
            ## MAXIMUM

            t_in = model_output_tensor#.unsqueeze(0).unsqueeze(0)  # shape (1,1,Z,Y,X)

            # 1) X-direction pooling with kernel=(1,1,3)
            #    pad=1 ensures we include the "left" and "right" neighbors at edges.
            max_x = F.max_pool3d(t_in, kernel_size=(1,1,9), stride=1, padding=(0,0,4))
            mask_x = (t_in == max_x)

            # 2) Y-direction pooling with kernel=(1,3,1)
            max_y = F.max_pool3d(t_in, kernel_size=(1,9,1), stride=1, padding=(0,4,0))
            mask_y = (t_in == max_y)

            # 3) Z-direction pooling with kernel=(3,1,1)
            max_z = F.max_pool3d(t_in, kernel_size=(9,1,1), stride=1, padding=(4,0,0))
            mask_z = (t_in == max_z)

            # Combine
            final_mask = mask_x | mask_y | mask_z
            out = t_in * final_mask.float()

            # Squeeze on the GPU
            if self._visualize:
                rec_pcd = copy.deepcopy(out)
            squeezed = out.squeeze()  # still on GPU
            squeezed += 0.75
            coords_cpu = squeezed.detach().cpu().numpy().astype("uint8")

        if self._visualize:
            reconstructed_pcd = self.construct_point_cloud_from_tensor_fast(rec_pcd)
            self.visualize(reconstructed_pcd)

        return coords_cpu, scaling_factor


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


    def construct_point_cloud_from_tensor_fast(self, tensor):

        # Squeeze on the GPU
        squeezed = tensor.squeeze()  # still on GPU

        # Find coords on GPU
        coords_gpu = (squeezed > self._threshold).nonzero(as_tuple=False)

        # Convert to float if needed (still GPU)
        coords_gpu = coords_gpu.float()

        # Transfer only valid coords to CPU
        coords_cpu = coords_gpu.cpu().numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_cpu)
        return pcd


    def nms_1d_axes_3d_torch(self, tensor_3d):
        #left2  = torch.roll(tensor_3d,  shifts=2,  dims=2)
        left  = torch.roll(tensor_3d,  shifts=1,  dims=2)
        right = torch.roll(tensor_3d,  shifts=-1, dims=2)
        #right2 = torch.roll(tensor_3d,  shifts=-2, dims=2)
        mask_x = (tensor_3d >= left) & (tensor_3d >= right)# & (tensor_3d >= left2) & (tensor_3d >= right2)

        #up2    = torch.roll(tensor_3d,  shifts=2,  dims=1)
        up    = torch.roll(tensor_3d,  shifts=1,  dims=1)
        down  = torch.roll(tensor_3d,  shifts=-1, dims=1)
        #down2  = torch.roll(tensor_3d,  shifts=-2, dims=1)
        mask_y = (tensor_3d >= up) & (tensor_3d >= down)# & (tensor_3d >= up2) & (tensor_3d >= down2)

        #front2 = torch.roll(tensor_3d,  shifts=2,  dims=0)
        front = torch.roll(tensor_3d,  shifts=1,  dims=0)
        back  = torch.roll(tensor_3d,  shifts=-1, dims=0)
        #back2  = torch.roll(tensor_3d,  shifts=-2, dims=0)
        mask_z = (tensor_3d >= front) & (tensor_3d >= back)# & (tensor_3d >= front2) & (tensor_3d >= back2)
        final_mask = mask_x | mask_y | mask_z
        out_3d = tensor_3d * final_mask

        return out_3d


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
    def reverse_scale_of_point_cloud(self, point_cloud, reverse_scale, should_scale_to_bounding_box=True):
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

