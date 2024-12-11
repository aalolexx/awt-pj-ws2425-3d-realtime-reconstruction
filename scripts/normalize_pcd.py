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
    old_bounds_max: old maximal values in (x,y,z)
    old_bound_min: old minimal values in (x,y,z)
    new_max: new maximum value for points of point cloud
    new_min: new minimum value (offset) for points of point cloud
output:
    normalized point cloud
"""
def normalize_pcd_old(pcd, old_bounds_max, old_bounds_min, new_max=128.0, new_min=0.0):
    # move point cloud such that all values are over positive minimum (0,0,0)
    translation_matrix = np.array([
        [1.0, 0.0, 0.0, -old_bounds_min[0]],
        [0.0, 1.0, 0.0, -old_bounds_min[1]],
        [0.0, 0.0, 1.0, -old_bounds_min[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    result = pcd.transform(translation_matrix)
    
    # isomorphic scale so that all points are between 0 and max new range
    scale = ((new_max - new_min) / (max(old_bounds_max) - min(old_bounds_min)))
    scale_matrix = np.array([
        [scale, 0.0,    0.0,    0.0],
        [0.0,   scale,  0.0,    0.0],
        [0.0,   0.0,    scale,  0.0],
        [0.0,   0.0,    0.0,    1.0]
    ])
    result = result.transform(scale_matrix)

    if new_min == 0:
       return result
    
    # move the point cloud to the new lower bound 
    translation_matrix2 = np.array([
        [1.0, 0.0, 0.0, new_min],
        [0.0, 1.0, 0.0, new_min],
        [0.0, 0.0, 1.0, new_min],
        [0.0, 0.0, 0.0, 1.0]
    ])
    result = result.transform(translation_matrix2)
    return result

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
description: saves point cloud in given file path
input:
    pcd: point cloud to be saved
    file_path: file path where the point cloud will be stored
"""
def save_pcd(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)

"""
This program loads the point clouds and normalizes them to a specific range (preperations for voxels)
example call:
    python normalize_pcd.py -i "data/pcd10000" -o "data/normalized10000"
"""
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process input arguments.")
    
    # Add arguments
    parser.add_argument('-i', '--input', type=str, help="Input Path (eg. THuman2.1_Release/model)", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output Path (eg. data/pcd)", required=True)
    parser.add_argument('-d', '--range', type=int, help="New max range of point cloud", required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    pcd_file_path = args.input
    normalize_file_path = args.output
    new_max_range = 128
    if args.range:
       new_max_range = args.range

    # runs through every point cloud and normalizes them
    n = 2445
    for i in range(n):
        number = str(i).zfill(4)
        # track progress
        print("normalizing:" + str(i))
        
        # build input file path:
        pcd_file = pcd_file_path + "/" + number + "_full.ply"

        # build output file path:
        normalized_file = normalize_file_path + "/" + number + "_full" + ".ply"
        
        # load mesh, create pcd and store pcd file
        pcd = load_pcd(pcd_file)
        pcd = normalize_pcd(pcd, new_max_range)
        save_pcd(pcd, normalized_file)

if __name__ == "__main__":
    main()