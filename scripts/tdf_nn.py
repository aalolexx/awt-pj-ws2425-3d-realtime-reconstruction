import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split
import open3d as o3d

def load_volume(path):
    return np.load(path)

class EncoderDecoder3D(nn.Module):
    def __init__(self):
        super(EncoderDecoder3D, self).__init__()
        
        # Encoder: Downsampling from 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),   # (16, 16, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 8, 8, 8)
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck: Down to 4, then process and return to same spatial size
        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 4, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),  # (32, 4, 4, 4)
            nn.ReLU(inplace=True)
        )
        
        # Decoder: Upsampling from 4 -> 8 -> 16 -> 32
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 8, 8, 8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),   # (8, 16, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1),    # (1, 32, 32, 32)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class VolumeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cut_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_cut.npy")])
        self.full_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_full.npy")])
        assert len(self.cut_files) == len(self.full_files), "Mismatch between cut and full file counts"

    def __len__(self):
        return len(self.cut_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_dir, self.cut_files[idx])
        target_path = os.path.join(self.data_dir, self.full_files[idx])

        input_volume = load_volume(input_path)
        target_volume = load_volume(target_path)
        
        input_tensor = torch.tensor(input_volume, dtype=torch.float32).unsqueeze(0)
        output_tensor = torch.tensor(target_volume, dtype=torch.float32).unsqueeze(0)
        
        return input_tensor, output_tensor


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

def train_model():
    train_split = 0.8
      # Create dataset and dataloader
    dataset = VolumeDataset("data/tdf10000")

    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = EncoderDecoder3D()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        print("epoch: " + str(epoch))
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print("batch: " + str(batch_idx))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print("loss: " + str(loss))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    print("Training complete!")
    evaluate_model(model, test_loader, criterion, device)

    torch.save(model.state_dict(), "tdf_weights.pth")
    print("Model saved to tdf_weights.pth")

# Example usage
if __name__ == "__main__":

    train_model()
    #volume = load_volume("data/tdf10000/0_full.npy")
    #coords = np.argwhere(volume == 0)
    ## coords will be an N x 3 array, where N is the number of voxels with value 0
    ## The coordinates are returned as (z, y, x) by default.
#
    ## If you want to visualize these points as a point cloud in Open3D:
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
#
    ## Visualize the point cloud
    #o3d.visualization.draw_geometries([pcd])

