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
desctiption: creates voxel grid from point cloud
input: 
    pcd: open3d point cloud object
output:
    generated voxel grid from point cloud
"""
def pcd_to_voxel_grid(pcd):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=1, min_bound=(0,0,0), max_bound=(128,128,128))
    return voxel_grid

"""
description: saves voxel grid in given file path
input:
    file_path: file path where the point cloud will be stored
    voxel_grid: point cloud to be saved
"""
def save_voxel_grid(file_path, voxel_grid):
    o3d.io.write_voxel_grid(file_path, voxel_grid)

"""
This program loads point clouds and generates voxels of it.
The program expects full point clouds and cut point clouds in an input folder and stores the voxels into the output folder
example call:
    python pcd_to_voxel.py -i "data/dataset10000" -o "data/voxel10000"
"""
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process input arguments.")
    
    # Add arguments
    parser.add_argument('-i', '--input', type=str, help="Input Path (eg. THuman2.1_Release/model)", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output Path (eg. data/pcd)", required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    pcd_file_path = args.input
    voxel_file_path = args.output

    # runs through every point cloud and normalizes them
    n = 29340
    total_skiped = 0
    for i in range(n):
        # build input file path:
        pcd_full_file = pcd_file_path + "/" + str(i) + "_full" + ".ply"
        pcd_cut_file = pcd_file_path + "/" + str(i) + "_cut" + ".ply"

        # build output file path:
        voxel_full_file = voxel_file_path + "/" + str(i - total_skiped) + "_full" + ".ply"
        voxel_cut_file = voxel_file_path + "/" + str(i - total_skiped) + "_cut" + ".ply"

        # load pcd, create voxel grid, save voxel grid
        pcd_full = load_pcd(pcd_full_file)
        pcd_cut = load_pcd(pcd_cut_file)
        voxel_full = pcd_to_voxel_grid(pcd_full)
        voxel_cut = pcd_to_voxel_grid(pcd_cut)

        voxels_B   = voxel_full.get_voxels()
        voxels_A  = voxel_cut.get_voxels()
        indices_A = {(v.grid_index[0], v.grid_index[1], v.grid_index[2]) for v in voxels_A}
        indices_B = {(v.grid_index[0], v.grid_index[1], v.grid_index[2]) for v in voxels_B}

        common_indices = indices_A.intersection(indices_B)
        num_common_voxels = len(common_indices)
        if len(voxels_A) != num_common_voxels:
            print("ERROR: PCD " + str(i) + " has an unexpected output. Expected common " + str(len(voxels_A)) + " points, but result showed " + str(num_common_voxels) + ", diff: " + str(abs(len(voxels_A) - num_common_voxels)))
            total_skiped = total_skiped + 1
        else:
            save_voxel_grid(voxel_full_file, voxel_full)
            save_voxel_grid(voxel_cut_file, voxel_cut)
    
    print("Program finished: " + str(total_skiped) + " had to be skipped")

if __name__ == "__main__":
    main()