import numpy as np
import open3d as o3d

from util.base_module import BaseModule

"""
Constructs a mesh from the reconstructed point cloud
"""
class MeshGenerator(BaseModule):
    def __init__(self, visualize=False):
        self._visualize = visualize

        '''if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
            self.is_mesh_created = False'''


    #
    # Run Step
    #
    def run_step(self, pcd_completed):
        mesh = self.ball_pivoting(pcd_completed)

        if self._visualize:
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        return mesh
    

    #
    # Ball pivoting
    #
    def ball_pivoting(self, pcd):
        #self.estimate_normals(pcd)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
        pcd.orient_normals_consistent_tangent_plane(10)

        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        radius = 1.5 * avg_distance
        radii = [1*radius, 1.5*radius, 2*radius, 2.5*radius]  # Define radii for BPA, adjust based on scale of your point cloud
        #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        #mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
        '''
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=7)
        vertices_to_remove = densities < np.quantile(densities, 0.08)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh_crop = mesh.crop(bbox)
        mesh_crop.compute_triangle_normals()'''

        #mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
        #self.colorize_normals(mesh)

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

    def visualize(self, mesh):
        if not self.is_mesh_created:
            self.vis.add_geometry(mesh)
            self.is_mesh_created = True
            self.mesh_placeholder = mesh  # TODO aeh is the placeholder needed?
        else:
            # Update points and colors of the existing point cloud
            self.mesh_placeholder.vertices = mesh.vertices
            self.mesh_placeholder.triangles = mesh.triangles

        # Display the frame
        self.vis.update_geometry(self.mesh_placeholder)
        self.vis.poll_events()
        self.vis.update_renderer()