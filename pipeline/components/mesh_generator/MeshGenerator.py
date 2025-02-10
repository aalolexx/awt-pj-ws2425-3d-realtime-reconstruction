import numpy as np
import open3d as o3d

from util.base_module import BaseModule
from skimage.measure import marching_cubes


"""
Constructs a mesh from the reconstructed point cloud
"""
class MeshGenerator(BaseModule):
    def __init__(self, visualize=False):
        self._visualize = visualize

        if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
            self.is_point_cloud_created = False


    #
    # Run Step
    #
    def run_step(self, pcd_completed, scaling_factor):
        #mesh = self.mesh_generation(pcd_completed)
        input_volume = np.zeros((64,128,64), dtype=np.float32)
        input_points = np.asarray(pcd_completed.points, dtype=np.uint8)

        for (x, y, z) in input_points:
            input_volume[x, y, z] = 1

        mesh = self.voxel_grid_to_mesh(input_volume, scaling_factor)

        if self._visualize:
            self.visualize(mesh)

        return mesh
    

    #
    # Ball pivoting
    #
    def mesh_generation(self, pcd):
        #self.estimate_normals(pcd)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        # NORMAL ESTIMATION
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
        pcd.orient_normals_consistent_tangent_plane(10)

        # BALL PIVOTING
        """
        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        radius = 1.5 * avg_distance
        radii = [1*radius, 1.5*radius, 2*radius, 2.5*radius]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        # FURTHER REMESHING
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=8000)
        mesh = mesh.filter_smooth_simple(number_of_iterations=2)
        """

        # POISSON
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)

        # MESH CLEANUP
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        #mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=8000)

        return mesh


    """
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
    """


    #
    # Update Vis Window
    #
    def visualize(self, mesh):
        if not self.is_point_cloud_created:
            self.vis.add_geometry(mesh)
            self.is_point_cloud_created = True
        else:
            self.vis.clear_geometries()
            self.vis.add_geometry(mesh)

        self.vis.poll_events()
        self.vis.update_renderer()


    #
    # Marching Cubes Approach
    #
    def voxel_grid_to_mesh(self, volume, scaling_factor, isolevel=0.1):

        verts, faces, norms, vals = marching_cubes(
            volume, 
            level=isolevel, 
            spacing=(1.0, 1.0, 1.0)
        )

        mesh = o3d.geometry.TriangleMesh()

        translation = np.array([32, 64, 32])
        mesh.vertices = o3d.utility.Vector3dVector((verts - translation) * scaling_factor)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        mesh_simplified = mesh.simplify_vertex_clustering(
            voxel_size=4,
            contraction=o3d.geometry.SimplificationContraction.Average
        )

        aabb = mesh_simplified.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        extents = max_bound - min_bound
        max_scale = 2.0 / max(extents)
        bounding_box_scale = np.array([max_scale, max_scale, max_scale])
        mesh_simplified.vertices = o3d.utility.Vector3dVector(mesh_simplified.vertices * bounding_box_scale)

        #mesh_simplified = mesh_simplified.filter_smooth_laplacian(number_of_iterations=1)
        mesh_simplified.compute_vertex_normals()

        return mesh_simplified