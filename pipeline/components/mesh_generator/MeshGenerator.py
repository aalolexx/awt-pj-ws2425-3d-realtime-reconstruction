import numpy as np
import open3d as o3d

from util.base_module import BaseModule

"""
Constructs a mesh from the reconstructed point cloud
"""
class MeshGenerator(BaseModule):
    def __init__(self, visualize=False):
        self._visualize = visualize


    #
    # Run Step
    #
    def run_step(self, pcd_completed):
        return self.ball_pivoting(pcd_completed)
    

    #
    # Ball pivoting
    #
    def ball_pivoting(self, pcd):
        self.estimate_normals(pcd)

        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        size = max_bound - min_bound
        print(f"Point cloud size: {size}")

        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        print(f"Average distance between points: {avg_distance}")

        radii = [1, 1.5, 2, 3, 4, 5]  # Define radii for BPA, adjust based on scale of your point cloud
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )

        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
        self.colorize_normals(mesh)

        return mesh


    #
    # estimates the normals of a point cloud
    #
    def estimate_normals(self, pcd):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Compute the center of the point cloud (mean of all points)
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)

        # Get the current normals
        normals = np.asarray(pcd.normals)

        # Flip normals that point inward (to make them point outward)
        for i in range(len(points)):
            vector_to_center = points[i] - center  # Vector from point to center
            if np.dot(normals[i], vector_to_center) < 0:
                normals[i] = -normals[i]  # Flip the normal if it's pointing inward

        # Set the flipped normals back to the point cloud
        pcd.normals = o3d.utility.Vector3dVector(normals)


    #
    # colorizes the mesh for visualization purposes (can be removed to save some time)
    #
    def colorize_normals(mesh):
        mesh.compute_vertex_normals()
        # Get vertex normals
        vertex_normals = np.asarray(mesh.vertex_normals)
        # Initialize a color array for the vertices
        vertex_colors = np.zeros_like(vertex_normals)
        # Normalize the normals and map them to a color (e.g., red to blue color map)
        for i, normal in enumerate(vertex_normals):
            color = (normal + 1) / 2  # Normalize normal (from -1 to 1) to (0 to 1)
            vertex_colors[i] = color  # Assign color based on normal
        # Set the colors to the mesh's vertices
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)