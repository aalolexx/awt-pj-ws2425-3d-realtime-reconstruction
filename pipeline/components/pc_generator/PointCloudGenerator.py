import cv2
import time
import numpy as np
import open3d as o3d

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


    """
    POST FILTER the point cloud, remove outliers
    """
    def apply_erosion_filter(self, point_cloud, radius=10, min_neighbors=10):
        # Create KDTree
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        # Precompute neighbors
        points = np.asarray(point_cloud.points)
        num_points = points.shape[0]
        neighbors_count = np.zeros(num_points, dtype=int)

        for i in range(num_points):
            _, idx, _ = kdtree.search_radius_vector_3d(point_cloud.points[i], radius)
            neighbors_count[i] = len(idx)  # Count neighbors

        # Filter points
        mask = neighbors_count > min_neighbors
        filtered_points = points[mask]

        # Create a new point cloud with the filtered points
        filtered_point_cloud = o3d.geometry.PointCloud()
        filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        return filtered_point_cloud

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
        cv2.imshow("masked depth", depth)

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
        pcd.transform([[-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        downpcd = pcd.voxel_down_sample(voxel_size=2)

        # POST PROCESSING
        downpcd = self.apply_erosion_filter(downpcd, radius=5, min_neighbors=5)
        downpcd = self.apply_erosion_filter(downpcd, radius=5, min_neighbors=5)

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
                self.pcd_placeholder = pcd
            else:
                # Update points and colors of the existing point cloud
                self.pcd_placeholder.points = pcd.points
                self.pcd_placeholder.colors = pcd.colors

            # Display the frame
            self.vis.update_geometry(self.pcd_placeholder)
            self.vis.poll_events()
            self.vis.update_renderer()

        return pcd