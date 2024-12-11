import argparse
import copy
import open3d as o3d
import numpy as np

"""
desctiption: loads pcd file (ie. ".ply" file)
input: 
    file_path: file path to point cloud file
output:
    loaded open3d point cloud object
"""
def load_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

"""
description: creates a point cloud from mesh file
input:
    pcd: open3d point cloud object
    new_max: new maximum value for points of point cloud
    new_min: new minimum value (offset) for points of point cloud
output:
    normalized point cloud
"""
def normalize_pcd(pcd, new_max=128.0, new_min=0.0):
    # move point cloud such that all values are over positive minimum (0,0,0)
    result = copy.deepcopy(pcd).translate(-pcd.get_min_bound(), relative=True)

    # isomorphic scale so that all points are between 0 and max new range
    scale = ((new_max - new_min) / (max(result.get_max_bound()) - min(result.get_min_bound())))
    result = result.scale(scale, center=(0,0,0))

    if new_min == 0.0:
       return result
    
    # move the point cloud to the new lower bound 
    result = result.translate((new_min, new_min, new_min), relative=True)
    return result

"""
description: creates a point cloud from mesh file
input:
    pcd: open3d point cloud object
    old_bounds_max: old maximal values in (x,y,z)
    old_bound_min: old minimal values in (x,y,z)
    new_max: new maximum value for points of point cloud
    new_min: new minimum value (offset) for points of point cloud
output:
    rotated view of point cloud
"""
def generate_visible_view(pcd, rotation_angle_deg):
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    R = pcd.get_rotation_matrix_from_xyz((0,rotation_angle_rad,0))
    rot_copy = copy.deepcopy(pcd).rotate(R, center=pcd.get_center())
    rot_copy = normalize_pcd(rot_copy, 128)
    diameter = np.linalg.norm(np.asarray(rot_copy.get_max_bound()) - np.asarray(rot_copy.get_min_bound()))
    c = rot_copy.get_center()
    camera = o3d.core.Tensor([c[0], c[1], diameter], o3d.core.float64)
    radius = diameter * 100
    tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(rot_copy)
    _, pt_map = tensor_pcd.hidden_point_removal(camera, radius)
    tensor_pcd = tensor_pcd.select_by_index(pt_map)
    pcd_cut = tensor_pcd.to_legacy()
    return rot_copy, pcd_cut

"""
description: saves point cloud in given file path
input:
    pcd: point cloud to be saved
    file_path: file path where the point cloud will be stored
"""
def save_pcd(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)

"""
This program loads point clouds and cuts the (from set perspective) non visible points in the back.
The program does from multiple angles by rotating the given point cloud
example call:
    python gen_rotated_cut_pcd.py -i "data/normalized10000" -o "data/dataset10000"
"""
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process input arguments.")
    
    # Add arguments
    parser.add_argument('-i', '--input', type=str, help="Input Path (eg. THuman2.1_Release/model)", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output Path (eg. data/pcd)", required=True)
    parser.add_argument('-r', '--rotation', type=int, help="Rotation angle per step", required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    pcd_file_path = args.input
    gen_file_path = args.output
    rotation_angle = 30
    if args.rotation:
       rotation_angle = args.rotation

    total_num_of_generated_pcd_pairs = 0
    # runs through every point cloud and normalizes them
    n = 2445
    for i in range(n):
        number = str(i).zfill(4)
        # track progress
        print("process pcd number:" + str(i))
        
        # build input file path:
        pcd_file = pcd_file_path + "/" + number + "_full.ply"
        
        # load mesh, create pcd and store pcd file
        pcd = load_pcd(pcd_file)
        for angle in range(0, 359, rotation_angle):
            print("generate pair number: " + str(total_num_of_generated_pcd_pairs))

            # generate point clouds
            pcd_full, pcd_cut = generate_visible_view(pcd, angle)
            
            # build output file path:
            full_file = gen_file_path + "/" + str(total_num_of_generated_pcd_pairs) + "_full" + ".ply"
            cut_file = gen_file_path + "/" + str(total_num_of_generated_pcd_pairs) + "_cut" + ".ply"
            total_num_of_generated_pcd_pairs = total_num_of_generated_pcd_pairs + 1

            # save point clouds
            save_pcd(pcd_full, full_file)
            save_pcd(pcd_cut, cut_file)

if __name__ == "__main__":
    main()