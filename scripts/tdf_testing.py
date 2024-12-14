import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def compute_tdf_simple(point_cloud, grid_resolution, truncation_distance):

    kdtree = cKDTree(point_cloud)

    x = np.arange(grid_resolution)
    y = np.arange(grid_resolution)
    z = np.arange(grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    voxel_centers = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    #print("Voxels:", voxel_centers)

    distances, _ = kdtree.query(voxel_centers)
    distances = np.minimum(distances, truncation_distance)
    #print("Distances:", distances)

    tdf = distances.T.reshape((grid_resolution, grid_resolution, grid_resolution), order='C')

    return np.transpose(np.flip(tdf, axis=(1,2)), (0, 2, 1))[:, ::-1, ::-1]

if __name__ == "__main__":
    np.random.seed(42)
    point_cloud = np.random.rand(100, 3) * 128
    voxel_grid = point_cloud.astype(int)

    x_values = [voxel[0] for voxel in voxel_grid]
    #print("All x-values:", sorted(x_values))
    #print(voxel_grid[:, 1])
    #print(voxel_grid[:, 2])

    grid_resolution = 128
    truncation_distance = 10

    tdf = compute_tdf_simple(voxel_grid, grid_resolution, truncation_distance)

    tdf_slice = tdf[0, :, :]  # YZ slice

    voxels_on_level_under_10 = voxel_grid[voxel_grid[:, 0] < 10]

    #print(voxels_on_level_under_10)
    plt.figure(figsize=(8, 6))
    plt.imshow(tdf_slice, extent=[0, grid_resolution, 0, grid_resolution], origin='lower', cmap='viridis')
    plt.colorbar(label='Truncated Distance')
    plt.scatter(voxels_on_level_under_10[:, 1], voxels_on_level_under_10[:, 2], c='red', s=1, label='Point Cloud (slice)')
    plt.title('Central Slice of the Truncated Distance Field')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig("first_layer_example.png", dpi=300)
    plt.show()
