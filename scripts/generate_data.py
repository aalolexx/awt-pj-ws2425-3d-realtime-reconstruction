import open3d as o3d
import numpy as np
import copy

def load_pcd(file_path):
    return o3d.io.read_point_cloud(file_path)

def draw_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def pcd_to_voxel_grid(pcd):
    return o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=1,
        min_bound=(-16,-16,-16),
        max_bound=(16,16,16))

def save_voxel_grid(file_path, voxel_grid):
    o3d.io.write_voxel_grid(file_path, voxel_grid)

def draw_pcd_bb(pcd, border):
  min_bound = np.array([-border, -border, -border])
  max_bound = np.array([border, border, border])
  bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
  bounding_box.color = (1, 0, 0)
  o3d.visualization.draw_geometries([pcd, bounding_box], window_name="Point Cloud Viewer", point_show_normal=False)

def rotate_point_cloud(pcd, axis, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    if axis.lower() == 'x':
        R = pcd.get_rotation_matrix_from_xyz((angle_radians, 0, 0))
    elif axis.lower() == 'y':
        R = pcd.get_rotation_matrix_from_xyz((0, angle_radians, 0))
    elif axis.lower() == 'z':
        R = pcd.get_rotation_matrix_from_xyz((0, 0, angle_radians))
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    
    center = pcd.get_center()
    #center = (0, 0, 0)
    pcd.rotate(R, center=center)

def normalize(pcd):
    bounding_box = pcd.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()
    pcd.translate(-center)
    max_extent = bounding_box.get_max_extent()
    scale_factor = 2.0 / max_extent
    pcd.scale(scale_factor, center=[0, 0, 0])
    return pcd

def normalize_anisotropic(pcd: o3d.geometry.PointCloud, pcd_cut: o3d.geometry.PointCloud):
    #bounding_box = pcd.get_axis_aligned_bounding_box()
    #center = bounding_box.get_center()
    #pcd.translate(-center)
    #pcd_cut.translate(-center)
    #bounding_box = pcd.get_axis_aligned_bounding_box()
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    #print("min bound full:", repr(pcd.get_min_bound()))
    #print("min bound cut: ", repr(pcd_cut.get_min_bound()))
    extents = max_bound - min_bound
    scale_factors = []
    for d in extents:
        #if abs(d) < 1e-12:
        #    scale_factors.append(1.0)
        #else:
        scale_factors.append(2.0 / d)

    scale_factors = np.array(scale_factors)
    #scale_factors = np.array(extents)

    scale_full = scale_factors * 16.0
    #print("scale full = ", scale_full)
    points = np.asarray(pcd.points)
    #print("points full type: ", points.dtype)
    #print("scale full type: ", scale_full.dtype)
    #print("lowest points: ", np.min(points, axis=0))
    #for p in range(points.shape[0]):
    #    points[p] = points[p] * scale_full
    points = points * scale_full
    pcd.points = o3d.utility.Vector3dVector(np.around(points, decimals=4))
    
    scale_cut = scale_factors * 16.0
    #print("scale cut = ", scale_cut)
    points_cut = np.asarray(pcd_cut.points)
    #print("points cut type: ", points.dtype)
    #print("scale cut type: ", scale_cut.dtype)
    #print("lowest points: ", np.min(points_cut, axis=0))
    #for p in range(points_cut.shape[0]):
    #    if (points_cut[p][0] == -0.4209879):
    #        print("was ist -0.4209879 * 38.00584319? ergebnis: ", -0.4209879 * 38.00584319)
    #    points_cut[p] = points_cut[p] * scale_cut
    points_cut = points_cut * scale_cut
    #pcd_cut.points = o3d.utility.Vector3dVector(points_cut)
    pcd_cut.points = o3d.utility.Vector3dVector(np.around(points_cut, decimals=4))

    #print("min bound full s:", pcd.get_min_bound())
    #print("min bound cut s: ", pcd_cut.get_min_bound())

    return pcd, pcd_cut

def generate_visible_view(pcd, rotation_angle_deg):
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    R = pcd.get_rotation_matrix_from_xyz((0, rotation_angle_rad, 0))
    rot_copy = copy.deepcopy(pcd).rotate(R, center=pcd.get_center())
    rot_copy = normalize(rot_copy)
    diameter = np.linalg.norm(np.asarray(rot_copy.get_max_bound()) - np.asarray(rot_copy.get_min_bound()))
    c = rot_copy.get_center()
    camera = o3d.core.Tensor([c[0], c[1], diameter], o3d.core.float64)
    radius = diameter * 100
    tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(rot_copy)
    _, pt_map = tensor_pcd.hidden_point_removal(camera, radius)
    tensor_pcd = tensor_pcd.select_by_index(pt_map)
    pcd_cut = tensor_pcd.to_legacy()
    return rot_copy, pcd_cut

if __name__ == "__main__":
    number_of_pcds = 2445
    total_pcds = 0
    for i in range(number_of_pcds):
        # load the pcd
        number = str(i).zfill(4)
        file_path = "data/normalized10000/" + number + "_full.ply"
        print(file_path)
        pcd = load_pcd(file_path)

        # fix rotation of tiled point clouds
        if i >= 526: # tilted PCDs start at 0526.ply
            rotate_point_cloud(pcd, 'x', 270)

        # normalize
        pcd = normalize(pcd)

        # generate cut versions of rotations
        rotation_angle = 30
        for angle in range(0, 359, rotation_angle):
            full_pcd, cut_pcd = generate_visible_view(pcd, angle)
            full_pcd, cut_pcd = normalize_anisotropic(full_pcd, cut_pcd)
            
            voxel_full = pcd_to_voxel_grid(full_pcd)
            voxel_cut = pcd_to_voxel_grid(cut_pcd)

            voxels_B   = voxel_full.get_voxels()
            voxels_A  = voxel_cut.get_voxels()
            indices_A = {(v.grid_index[0], v.grid_index[1], v.grid_index[2]) for v in voxels_A}
            indices_B = {(v.grid_index[0], v.grid_index[1], v.grid_index[2]) for v in voxels_B}

            common_indices = indices_A.intersection(indices_B)
            num_common_voxels = len(common_indices)
            if len(voxels_A) != num_common_voxels:
                #print("voxels in cut: ", indices_A)
                diff_indices_A_not_in_B = indices_A - indices_B
                print("Voxel(s) in 'voxel_cut' but not in 'voxel_full':", diff_indices_A_not_in_B)
                differing_voxel = next(iter(diff_indices_A_not_in_B))
                print("The single differing voxel is:", differing_voxel)

                #draw_pcd_bb(full_pcd,16)
                #draw_pcd_bb(cut_pcd,16)
                print("ERROR: PCD " + str(i) + " has an unexpected output. Expected common " + str(len(voxels_A)) + " points, but result showed " + str(num_common_voxels) + ", diff: " + str(abs(len(voxels_A) - num_common_voxels)))
            else:
                voxel_path_full = "data/voxel_new/" + str(total_pcds) + "_full.ply"
                voxel_path_cut = "data/voxel_new/" + str(total_pcds) + "_cut.ply"
                save_voxel_grid(voxel_path_full, voxel_full)
                save_voxel_grid(voxel_path_cut, voxel_cut)
                #draw_pcd_bb(full_pcd,16)
                #draw_pcd_bb(cut_pcd,16)
                total_pcds = total_pcds + 1
    #print("was ist -0.4209879 * 38.00584319? ergebnis: ", -0.4209879 * 38.00584319)