import open3d as o3d
import numpy as np

filepath = "data/tdf_new/0_full.npy"

def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])

def tdf_to_pcd(tdf):
    zero_indices = np.argwhere(tdf == 0)
    coordinates = zero_indices.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    return pcd

def load_tdf(filename):
    return np.load(filename)

if __name__ == "__main__":
    tdf = load_tdf(filepath)
    pcd = tdf_to_pcd(tdf)
    visualize(pcd)