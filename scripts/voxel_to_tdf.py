import numpy as np
import argparse
from scipy.spatial import cKDTree
import open3d as o3d

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
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=4, min_bound=(0,0,0), max_bound=(32,32,32))
    return voxel_grid

"""
description: saves voxel grid in given file path
input:
    file_path: file path where the point cloud will be stored
    voxel_grid: point cloud to be saved
"""
def save_voxel_grid(file_path, voxel_grid):
    o3d.io.write_voxel_grid(file_path, voxel_grid)


def compute_tdf_simple(voxels, grid_resolution, truncation_distance):
    kdtree = cKDTree(voxels)

    # Generate a grid of voxel indices
    x = np.arange(grid_resolution)
    y = np.arange(grid_resolution)
    z = np.arange(grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    voxel_centers = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    distances, _ = kdtree.query(voxel_centers)
    distances = np.minimum(distances, truncation_distance)

    tdf = distances.reshape((grid_resolution, grid_resolution, grid_resolution))
    return tdf

"""
This program generates Truncated Distance Fields (TDF) from voxel grids
example call:
    python voxel_to_tdf.py -i "data/voxel10000" -o "data/tdf10000"
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
    voxel_file_path = args.input
    tdf_file_path = args.output

    # runs through every point cloud and normalizes them
    n = 26804
    for i in range(n):

        print("TDF: ", i)

        # build input file path:
        voxel_full_file = voxel_file_path + "/" + str(i) + "_full" + ".ply"
        voxel_cut_file = voxel_file_path + "/" + str(i) + "_cut" + ".ply"

        # build output file path:
        tdf_full_file = tdf_file_path + "/" + str(i) + "_full" + ".npy"
        tdf_cut_file = tdf_file_path + "/" + str(i) + "_cut" + ".npy"

        # load pcd, create voxel grid, save voxel grid
        pcd_full = load_pcd(voxel_full_file)
        pcd_cut = load_pcd(voxel_cut_file)
        voxel_grid_full = pcd_to_voxel_grid(pcd_full)
        voxel_grid_cut = pcd_to_voxel_grid(pcd_cut)

        voxels_full = np.array([np.asarray(voxel.grid_index) for voxel in voxel_grid_full.get_voxels()])
        voxels_cut = np.array([np.asarray(voxel.grid_index) for voxel in voxel_grid_cut.get_voxels()])

        tdf_full = compute_tdf_simple(voxels_full, grid_resolution=32, truncation_distance=10)
        tdf_cut = compute_tdf_simple(voxels_cut, grid_resolution=32, truncation_distance=10)

        np.save(tdf_full_file, tdf_full)
        np.save(tdf_cut_file, tdf_cut)

if __name__ == "__main__":
    main()