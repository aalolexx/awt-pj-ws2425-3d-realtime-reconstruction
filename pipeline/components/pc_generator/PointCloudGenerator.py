import cv2
import time
import numpy as np
import open3d as o3d
import copy
from PIL import Image
import matplotlib.pyplot as plt

from util.base_module import BaseModule

class PointCloudGenerator(BaseModule):
    def __init__(self, visualize=False):
        """Initialize the PointCloudGenerator with webcam capture."""
        self._visualize = visualize
        if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
            self.is_point_cloud_created = False


    def create_point_cloud(self, foreground_mask, image_depth, equal_his=False, depth_threshold=0.1):
        # Convert depth to 3D points without perspective scaling
        if equal_his:
            equalized_depth = cv2.normalize(image_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            equalized_depth = cv2.equalizeHist(equalized_depth)
            depth = np.asarray(equalized_depth)
            depth = depth / 255
        else:
            depth = np.asarray(image_depth)


        width = np.shape(depth)[1]
        height = np.shape(depth)[0]

        depth = np.where(foreground_mask > 1, depth, 0)

        # Generate a 3D point grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        x = u
        y = v
        z = depth * 300  # Maintain straight-line scaling without perspective adjustment

        # Stack and filter valid points
        points = np.dstack((x, y, z)).reshape(-1, 3)

        # Filter points
        points = points[points[:, 2] != 0]

        # Create an Open3D point cloud from the resulting 3D points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Flip the point cloud (optional, depending on the coordinate system)
        pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        downpcd = pcd.voxel_down_sample(voxel_size=2)

        # POST PROCESSING
        downpcd, _ = downpcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.25)  # remove "noisy" outliers
        downpcd, _ = downpcd.remove_radius_outlier(nb_points=50, radius=15)  # remove big satellites

        # Apply DBSCAN clustering (only use biggest point cluster)
        labels = np.array(downpcd.cluster_dbscan(eps=10, min_points=50))
        if labels.max() > 0 and downpcd.has_points():
            largest_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            largest_cluster = downpcd.select_by_index(largest_cluster_indices)
            downpcd = largest_cluster

        return downpcd


    def run_step(self, foreground_mask, depth_image):
        """Process a single frame from the webcam.
        This method can be overridden to implement custom frame processing.
        """
        foreground_mask = cv2.resize(foreground_mask, (256, 256))
        depth_image = cv2.resize(depth_image, (256, 256))

        if np.max(foreground_mask) == 0 or np.max(depth_image) == 0:
            return None

        pcd = self.create_point_cloud(foreground_mask, depth_image, equal_his=False)

        if self._visualize:
            if not self.is_point_cloud_created:
                self.vis.add_geometry(pcd)
                self.is_point_cloud_created = True
                self.pcd_placeholder = pcd # TODO aeh is the placeholder needed?
            else:
                # Update points and colors of the existing point cloud
                self.pcd_placeholder.points = pcd.points
                self.pcd_placeholder.colors = pcd.colors

            # Display the frame
            self.vis.update_geometry(self.pcd_placeholder)
            self.vis.poll_events()
            self.vis.update_renderer()

        return pcd