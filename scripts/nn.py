import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
from torch.utils.data import random_split

# ------ EncoderDecoder3D ------
class EncoderDecoder3D(nn.Module):
    def __init__(self):
        super(EncoderDecoder3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # (16, 64, 64, 64)
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 32, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 16, 16, 16)
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 8, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),  # (64, 8, 8, 8)
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 32, 32, 32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=8, stride=4, padding=2),  # (1, 128, 128, 128)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# ------ VoxelDataset ------
class VoxelDataset(Dataset):
    def __init__(self, data_dir, voxel_resolution=128):
        self.data_dir = data_dir
        self.voxel_resolution = voxel_resolution

        self.cut_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_cut.ply")])
        self.full_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_full.ply")])

        assert len(self.cut_files) == len(self.full_files), "Mismatch between cut and full file counts"

    def load_voxel(self, file_path):
      point_cloud = o3d.io.read_point_cloud(file_path)
      voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
          point_cloud, voxel_size=1, min_bound=(0,0,0), max_bound=(self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)
      )

      voxels = np.zeros((self.voxel_resolution,) * 3, dtype=np.float32)
      for voxel in voxel_grid.get_voxels():
          x, y, z = voxel.grid_index
          if 0 <= x < self.voxel_resolution and 0 <= y < self.voxel_resolution and 0 <= z < self.voxel_resolution:
              voxels[x, y, z] = 1.0

      return voxels

    def __len__(self):
        return len(self.cut_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_dir, self.cut_files[idx])
        target_path = os.path.join(self.data_dir, self.full_files[idx])

        input_voxel = self.load_voxel(input_path)
        target_voxel = self.load_voxel(target_path)

        input_tensor = torch.tensor(input_voxel, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_voxel, dtype=torch.float32).unsqueeze(0)

        return input_tensor, target_tensor

# ------ evaluate_model ------
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# ------ train_model ------
def train_model():
    data_dir = "data/voxel10000"
    batch_size = 8
    num_epochs = 1
    learning_rate = 0.001
    train_split = 0.8

    dataset = VoxelDataset(data_dir)
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Dataset Size: {total_size}")
    print(f"Training Size: {train_size}")
    print(f"Testing Size: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EncoderDecoder3D()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print("epoch: " + str(epoch))
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print("batch: " + str(batch_idx))
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print("loss: " + str(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training complete!")

    evaluate_model(model, test_loader, criterion, device)

    torch.save(model.state_dict(), "encoder_decoder3d.pth")
    print("Model saved to encoder_decoder3d.pth")

# ------ predict ------
def predict():
    model = EncoderDecoder3D()
    model.load_state_dict(torch.load("encoder_decoder3d.pth"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data_dir = "data/voxel10000"
    dataset = VoxelDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    dataloader_iter = iter(train_loader)
    inputs, targets = next(dataloader_iter)

    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        outputs = model(inputs)
    
    outputs.to(device)
    non_zero_count = torch.count_nonzero(inputs)
    print(f"Number of non-zero elements: {non_zero_count}")

   # Ensure the tensor is on the CPU
    if inputs.is_cuda:
        inputs = inputs.cpu()

    if outputs.is_cuda:
        outputs = outputs.cpu()
        
    if targets.is_cuda:    
        targets = targets.cpu()
    
    voxel_inputs = inputs.squeeze().numpy()  # Shape: [128, 128, 128]
    voxel_outputs = outputs.squeeze().numpy()  # Shape: [128, 128, 128]
    voxel_targets = targets.squeeze().numpy()  # Shape: [128, 128, 128]
    
    occupied_inputs = np.argwhere(voxel_inputs > 0.5)
    occupied_outputs = np.argwhere(voxel_outputs > 0.05)
    occupied_targets = np.argwhere(voxel_targets > 0.5)

    points_inputs = np.array(occupied_inputs, dtype=np.float32)  # Shape: [N, 3]
    point_cloud_inputs = o3d.geometry.PointCloud()
    point_cloud_inputs.points = o3d.utility.Vector3dVector(points_inputs)
    
    points_outputs = np.array(occupied_outputs, dtype=np.float32)  # Shape: [N, 3]
    point_cloud_outputs = o3d.geometry.PointCloud()
    point_cloud_outputs.points = o3d.utility.Vector3dVector(points_outputs)
    
    points_targets = np.array(occupied_targets, dtype=np.float32)  # Shape: [N, 3]
    point_cloud_targets = o3d.geometry.PointCloud()
    point_cloud_targets.points = o3d.utility.Vector3dVector(points_targets)
    
    print(f"Number of input points: {len(point_cloud_inputs.points)}")
    print(f"Number of output points: {len(point_cloud_outputs.points)}")
    print(f"Number of true points: {len(point_cloud_targets.points)}")

    o3d.visualization.draw_geometries([point_cloud_inputs], window_name="Voxel input")
    o3d.visualization.draw_geometries([point_cloud_outputs], window_name="Voxel output")
    o3d.visualization.draw_geometries([point_cloud_targets], window_name="Voxel truth")

if __name__ == "__main__":
    train_model()
    #predict()
