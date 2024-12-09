import argparse
import open3d as o3d

"""
desctiption: loads mesh file (ie. ".obj" file)
input: 
    file_path: File path to mesh file
output:
    loaded open3d mesh object
"""
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        print("Failed to load the mesh. Please check the file path.")
    return mesh

"""
description: creates a point cloud from mesh file
input:
    mesh: open3d mesh object
    num_points: number of points
output:
    open3d point cloud object
"""
def create_pcd_from_mesh(mesh, num_p=8192):
    point_cloud_poisson = mesh.sample_points_poisson_disk(number_of_points=num_p)
    return point_cloud_poisson

"""
description: saves point cloud in given file path
input:
    pcd: point cloud to be saved
    file_path: file path where the point cloud will be stored
"""
def save_pcd(pcd, file_path):
  o3d.io.write_point_cloud(file_path, pcd)

"""
This program loads the meshes from the THuman 2.1 dataset and extracts point clouds from the meshes.
The program can be taken as template to load the contents of another dataset.
Example call:
    python mesh_to_pcd.py -i "THuman2.1_Release/model" -o "data/pcd10000" -p 10000 -is
"""
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    
    # Add arguments
    parser.add_argument('-i', '--input', type=str, help="Input Path (eg. THuman2.1_Release/model)", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output Path (eg. data/pcd)", required=True)
    parser.add_argument('-p', '--points', type=int, help="Number of points in pcd (eg. 8192)", required=False)
    parser.add_argument('-is', '--input_seperated', action='store_true', help="Are the meshes in seperated files?", required=False)
    parser.add_argument('-os', '--output_seperated', action='store_true', help="Should the meshes be stored in seperated files?", required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    mesh_file_path = args.input
    pcd_file_path = args.output

    number_of_points = 8192
    if args.points:
        number_of_points = args.points
    
    seperated_inputs = args.input_seperated
    seperated_outputs = args.output_seperated

    # runs through every mesh and stores the the pointcloud data.
    n = 2445
    for i in range(n):
        number = str(i).zfill(4)
        # track progress
        print("processing mesh:" + str(i))
        
        # build input file path:
        input_sep_path = ""
        if seperated_inputs:
            input_sep_path = number + "/"
        mesh_file = mesh_file_path + "/" + input_sep_path + number + ".obj"

        # build output file path:
        output_sep_path = ""
        if seperated_outputs:
            output_sep_path = str(i).zfill(4) + "/"
        pcd_file = pcd_file_path + "/" + output_sep_path + number + "_full" + ".ply"
        
        # load mesh, create pcd and store pcd file
        print(mesh_file)
        print(pcd_file)
        mesh = load_mesh(mesh_file)
        pcd = create_pcd_from_mesh(mesh, number_of_points)
        save_pcd(pcd, pcd_file)

if __name__ == "__main__":
    main()