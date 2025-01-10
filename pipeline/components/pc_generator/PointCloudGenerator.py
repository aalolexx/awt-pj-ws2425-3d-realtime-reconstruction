import cv2
import time
import numpy as np
import open3d as o3d

from util.base_module import BaseModule


class PointCloudGenerator(BaseModule):
    def __init__(self, visualize=False):
        """Initialize the PointCloudGenerator with webcam capture."""
        self._visualize = visualize
        if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
            self.is_point_cloud_created = False


    def get_human_mask(self, depth_image):
        _, mask = cv2.threshold(depth_image, 0.4, 1.0, cv2.THRESH_BINARY)
        mask = mask * 255
        mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        biggest_contour_idx = np.argmax([cv2.contourArea(c) for c in contours])

        #if most_central_centeroid_idx < len(contours):
        #    print(most_central_centeroid_idx)
        #    final_mask = np.zeros(depth_image.shape)
        #    cv2.drawContours(image=final_mask, contours=contours, contourIdx=most_central_centeroid_idx, color=(255), thickness=cv2.FILLED)
        #else:
        #    print(f"no contour found: c={len(contours)}, i={most_central_centeroid_idx}")
        #    final_mask = mask

        try:
            final_mask = np.zeros(depth_image.shape)
            cv2.drawContours(image=final_mask, contours=contours, contourIdx=biggest_contour_idx, color=(255), thickness=cv2.FILLED)
            #cv2.imshow('mask', final_mask)
            return final_mask, contours[biggest_contour_idx]
        except:
            #cv2.imshow('mask', mask)
            return mask, _
    

    def create_point_cloud(self, image_depth, equal_his=False, depth_threshold=0.1):
        # Convert depth to 3D points without perspective scaling
        if equal_his:
            equalized_depth = cv2.normalize(image_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            equalized_depth = cv2.equalizeHist(equalized_depth)
            depth = np.asarray(equalized_depth)
            depth = depth / 255
        else:
            depth = np.asarray(image_depth)

        # Cut image to actual human
        human_mask, contour = self.get_human_mask(depth)
        human_mask_bb = cv2.boundingRect(contour)
        hx, hy, hw, hh = human_mask_bb
        #cut_image_rgb = image_rgb[hy:hy+hh, hx:hx+hw]
        cut_image_depth = image_depth[hy:hy+hh, hx:hx+hw]
        depth = depth[hy:hy+hh, hx:hx+hw]
        human_mask = human_mask[hy:hy+hh, hx:hx+hw]

        width = np.shape(cut_image_depth)[1]
        height = np.shape(cut_image_depth)[0]

        # Generate a 3D point grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        x = u 
        y = v
        z = depth * 128 # Maintain straight-line scaling without perspective adjustment

        # Stack and filter valid points
        points = np.dstack((x, y, z)).reshape(-1, 3)
        
        #valid_points = points[depth.reshape(-1) > 0]  # Remove zero-depth points
        points_x = points[:, 0].astype(int)
        points_y = points[:, 1].astype(int)
        valid_points = points[(human_mask[points_y, points_x] == 255) & (depth.reshape(-1) > depth_threshold)] # Apply Human Mask # Remove zero-depth points

        # Create an Open3D point cloud from the resulting 3D points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)

        # Set the colors from the RGB image
        #rgb = np.asarray(image_rgb).reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        #pcd.colors = o3d.utility.Vector3dVector(rgb[depth.reshape(-1) > depth_threshold])

        # Flip the point cloud (optional, depending on the coordinate system)
        pcd.transform([[-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        downpcd = pcd.voxel_down_sample(voxel_size=2)
        return downpcd


    def run_step(self, depth_image):
        """Process a single frame from the webcam.
        This method can be overridden to implement custom frame processing.
        """
        depth_image = cv2.resize(depth_image, (256, 256))
        pcd = self.create_point_cloud(depth_image, equal_his=False)

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