import numpy as np
import open3d as o3d

from util.base_module import BaseModule
from skimage.measure import marching_cubes

# flying edges appraoch
#import vtk
#from vtk.util import numpy_support
#from scipy.ndimage import zoom

"""
Constructs a mesh from the reconstructed point cloud
"""
class MeshGenerator(BaseModule):
    def __init__(self, visualize=False, approach='marching'):
        """Initialize the MeshGenerator."""
        self._visualize = visualize
        integrated_approaches = ['marching', 'ball', 'poisson', 'alpha']
        if approach in integrated_approaches:
            self._approach = approach
        else:
            print("No valid approach for mesh generation, using marching (marching cubes) instead")
            self._approach = 'marching'

        if self._visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=800, height=600)
            self.is_point_cloud_created = False


    #
    # Run Step
    #
    def run_step(self, pcd_completed, scaling_factor):
        """"Reconstruct mesh from completed point cloud"""
        input_volume = np.zeros((64,128,64), dtype=np.float32)
        input_points = np.asarray(pcd_completed.points, dtype=np.uint8)

        for (x, y, z) in input_points:
            input_volume[x, y, z] = 1

        if self._approach == 'marching':
            mesh = self.voxel_grid_to_mesh(input_volume, scaling_factor)
        elif self._approach == 'ball':
            mesh = self.ball_pivoting(pcd_completed)
        elif self._approach == 'poisson':
            mesh = self.poisson(pcd_completed)
        elif self._approach == 'alpha':
            mesh = self.alpha_shapes(pcd_completed)

        if self._visualize:
            self.visualize(mesh)

        return mesh
    

    #
    # Normal estimation
    #
    def normal_estimation(self, pcd):
        """Downsampling and normal estimation"""
        #self.estimate_normals(pcd)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        # Normal estimation
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
        pcd.orient_normals_consistent_tangent_plane(10)

        return pcd

    #
    # Colorize normals
    #
    def colorize_normals(self, mesh):
        """Colorizes normals (only used for testing purposes)"""
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


    #
    # Update Vis Window
    #
    def visualize(self, mesh):
        """Visualizes our reconstructed mesh"""
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
        """Applies marching cubes algorithm to reconstruct a mesh from point cloud"""
        # Run marching cubes algorithm
        verts, faces, norms, vals = marching_cubes(
            volume, 
            level=isolevel, 
            spacing=(1.0, 1.0, 1.0)
        )

        # Build the mesh
        mesh = o3d.geometry.TriangleMesh()

        # Rescale and reposition the mesh
        translation = np.array([32, 64, 32])
        mesh.vertices = o3d.utility.Vector3dVector((verts - translation) * scaling_factor)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Reduce the mesh complexity
        mesh_s = mesh.simplify_vertex_clustering(
            voxel_size=4,
            contraction=o3d.geometry.SimplificationContraction.Average
        )

        # Normalize the mesh to a bounding box
        aabb = mesh_s.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        extents = max_bound - min_bound
        max_scale = 2.0 / max(extents)
        bounding_box_scale = np.array([max_scale, max_scale, max_scale])
        mesh_s.vertices = o3d.utility.Vector3dVector(mesh_s.vertices * bounding_box_scale)

        mesh_s.compute_vertex_normals()

        return mesh_s


    #
    # Ball Pivoting
    #
    def ball_pivoting(self, pcd):
        """Uses the open3d ball pivoting algorithm to construct a mesh from point cloud"""
        pcd = self.normal_estimation(pcd)
        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        radius = 1.5 * avg_distance
        radii = [1 * radius, 1.5 * radius, 2 * radius, 2.5 * radius]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        mesh = self.mesh_cleanup(mesh)
        return mesh


    #
    # Poisson
    #
    def poisson(self, pcd):
        """Uses the open3d poisson algorithm to construct a mesh from point cloud"""
        pcd = self.normal_estimation(pcd)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        self.mesh_cleanup(mesh)

        return mesh


    #
    # Alpha Shapes
    #
    def alpha_shapes(self, pcd):
        """Uses the open3d alpha shapes algorithm to construct a mesh from point cloud"""
        alpha = 0.045
        print(f"alpha={alpha:.3f}")
        #        PointCloudReconstructor.reverse_scale_of_point_cloud(pcd, 1)
        points = np.asarray(pcd.points)
        points = points - (32,64,32)
        points = points * 1
        rescaled_point_cloud = o3d.geometry.PointCloud()
        rescaled_point_cloud.points = o3d.utility.Vector3dVector(points)

        points = np.asarray(rescaled_point_cloud.points)
        min_bound = rescaled_point_cloud.get_min_bound()
        max_bound = rescaled_point_cloud.get_max_bound()
        extents = max_bound - min_bound
        max_scale = 2.0 / max(extents)
        bounding_box_scale = np.array([max_scale, max_scale, max_scale])
        points = points * bounding_box_scale
        rescaled_point_cloud.points = o3d.utility.Vector3dVector(points)

        rescaled_point_cloud = rescaled_point_cloud.voxel_down_sample(voxel_size=0.02)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            rescaled_point_cloud,
            alpha
        )
        mesh = self.mesh_cleanup(mesh)

        return mesh

    #
    # Mesh cleanup
    #
    def mesh_cleanup(self, mesh):
        """Reduces the mesh complexity"""
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=8000)
        return mesh


    #
    # Optimized Flying Edge Approach
    #
    def optimized_flying_edges(self, volume, scaling):
        """"
        This is a very optimized and specific flying edge alogirthm to reconstruct the mesh
        For maximum speed we only return the points and faces and let unity construct the normals
        This function was added to test the limits of our approach
        """
        # Convert to float32 for faster processing
        volume = volume.astype(np.float32)

        # Downsample
        volume = zoom(volume, 0.5, order=1)

        # Convert NumPy array to VTK ImageData
        depth, height, width = volume.shape
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(width, height, depth)

        vtk_array = numpy_support.numpy_to_vtk(volume.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
        vtk_data.GetPointData().SetScalars(vtk_array)

        # Use Flying Edges Algorithm
        flying_edges = vtk.vtkFlyingEdges3D()
        flying_edges.SetInputData(vtk_data)
        flying_edges.SetValue(0, 0.005)  # Isosurface level
        flying_edges.Update()

        # Decimate to reduce trianglee
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(flying_edges.GetOutput())
        decimate.SetTargetReduction(0.2)  # Reduce triangle count by 30%
        decimate.Update()

        # Extract points and faces
        polydata = decimate.GetOutput()
        points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        faces = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        #mesh = o3d.geometry.TriangleMesh()
        #mesh.vertices = o3d.utility.Vector3dVector(points)
        #mesh.triangles = o3d.utility.Vector3iVector(faces)
        #mesh.compute_vertex_normals()

        # Scale points and faces to a bounding box
        points = points * scaling
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        extents = max_bound - min_bound
        max_scale = 2.0 / max(extents)
        bounding_box_scale = np.array([max_scale, max_scale, max_scale])
        points = points * bounding_box_scale

        return points, faces
        #return mesh
    

    #
    # Flying Edge Approach
    #
    def flying_edges_algorithm(self, volume):
        """
        Flying Edge Algorithm to construct the mesh from volume
        This algorithm is for testing purposes
        Rescaling to fit pipeline assumptions was not done here
        """
        depth, height, width = volume.shape
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(width, height, depth)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Flatten and copy data into VTK format
        vtk_array = numpy_support.numpy_to_vtk(num_array=volume.ravel(),
                                               deep=True,
                                               array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.GetPointData().SetScalars(vtk_array)

        # Apply Flying Edges 3D algorithm
        flying_edges = vtk.vtkFlyingEdges3D()
        flying_edges.SetInputData(vtk_data)
        flying_edges.SetValue(0, 0.5)
        flying_edges.Update()

        # Convert VTK PolyData to Open3D Mesh
        polydata = flying_edges.GetOutput()
        points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        faces = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        # Create Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        return mesh