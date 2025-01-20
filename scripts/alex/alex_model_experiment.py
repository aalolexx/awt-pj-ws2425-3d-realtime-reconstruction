import math
import torch
import torch.nn as nn
import open3d as o3d
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset, SubsetRandomSampler


"""
THE AUTO ENCODER MODEL
"""

class MaskedPointCloudEncoderDecoder(nn.Module):
    def __init__(self, input_dim=32):
        super(MaskedPointCloudEncoderDecoder, self).__init__()

        # Encoder: Maps input point cloud to a latent representation
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        stride_factor = 2*2*2

        # Fully connected layers to create a bottleneck
        grid_count = (input_dim // stride_factor) ** 3 # 4 = amount of padding (multiply them)
        self.flatBottleneck = nn.Sequential(
            nn.Flatten(),  # Flatten to (N, 64 * grid_count)
            nn.Dropout(p=0.2),
            nn.Linear(32 * grid_count, 16 * grid_count),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(16 * grid_count, 16 * grid_count),
            nn.BatchNorm1d(16 * grid_count),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(16 * grid_count, 32 * grid_count),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(32, input_dim // stride_factor, input_dim // stride_factor, input_dim // stride_factor))  # Reshape back
        )

        # Decoder: Maps latent representation back to point cloud
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output values in [0, 1]
        )

    def forward(self, x):
        original_input = x  # Save the original input to use for masking

        x = self.encoder(x)  # Encode spatial features
        x = self.flatBottleneck(x)
        x = self.decoder(x)  # Decode back to voxel grid

        # Mask for original voxels that are 1
        mask = original_input > 0.8
        x = torch.where(mask, torch.tensor(1.0).to(x.device), x)
        return x


#
# LOSS METHODS
#

"""
COMBINED LOSS
"""

#def soft_dice_loss(pred, target):
#    smooth = 1e-6
#    intersection = (pred * target).sum()
#    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
#    return 1 - dice

#
# IoU Loss
#
class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps  # Small constant to avoid division by zero

    def forward(self, preds, targets):
        # Ensure preds are within [0, 1] range
        preds = torch.clamp(preds, 0, 1)
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds + targets) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = 1 - iou
        return iou_loss


#
# Voxel Loss Count
#
class VoxelCountLoss(nn.Module):
    def __init__(self, target_ratio=2.0):
        """
        Custom loss combining IoU loss with a voxel count penalty.

        Args:
            target_ratio (float): Target ratio of predicted voxels to input voxels.
        """
        super(VoxelCountLoss, self).__init__()
        self.target_ratio = target_ratio

    def forward(self, pred, input_voxels):
        # Compute the count penalty
        pred_count = torch.sum(pred, dim=[1, 2, 3, 4])  # Predicted voxel count
        input_count = torch.sum(input_voxels, dim=[1, 2, 3, 4])  # Input voxel count
        # Desired voxel count
        target_count = self.target_ratio * input_count
        # Normalize count penalty by total grid size
        grid_size = pred[0].numel()  # Total number of voxels per grid (C * D * H * W)
        count_penalty = torch.abs(pred_count - target_count) / grid_size
        count_penalty = count_penalty.mean()
        return count_penalty


"""
def masked_loss_with_gaussian(original_input, pred, target):
    # Mask for non-zero voxels in the target (which are the original valid ones)
    mask = original_input == 1  # Voxel is 1 in the target

    loss_mse_fn = nn.MSELoss(reduction='none')
    loss_mse = loss_mse_fn(pred, target)

    loss_bce_fn = nn.BCELoss(reduction='none')
    loss_bce = loss_bce_fn(
        torch.where(original_input == 1, torch.tensor(1.0).to(original_input.device), 0),
        torch.where(target == 1, torch.tensor(1.0).to(target.device), 0),
    )

    # only penalize non mask values
    loss_mse = loss_mse * (~mask)
    loss_bce = loss_bce * (~mask)
    return loss_mse.mean() + loss_bce.mean()
"""


def masked_loss_binary(original_input, pred, target):
    # IoU Loss
    iou_loss_fn = IoULoss()
    loss_iou = iou_loss_fn(pred, target)

    # Voxel Count Loss
    voxel_count_loss_fn = VoxelCountLoss()
    count_loss = voxel_count_loss_fn(pred, original_input)

    # Masked BCE Loss
    # Mask for non-zero voxels in the target (which are the original valid ones)
    mask = original_input > 0.8  # Voxel is 1 in the target

    loss_fn_bce = nn.BCELoss(reduction='none')
    loss_bce = loss_fn_bce(pred, target)

    # Combine all losses
    loss = loss_bce * (~mask)
    combined_loss = loss.mean() + (count_loss * 0.5) + (loss_iou * 0.5)
    return combined_loss


"""
DATASET PREPARATIONS
"""
class VoxelGridDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Root directory containing subfolders with .ply files.
            split (str): 'train' or 'val'. Determines the file names.
            target_points (int): Number of points to standardize the point clouds to.
            transform (callable, optional): Optional transform to apply to the point clouds.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        self.voxel_size = None
        self.grid_min_bound = None
        self.grid_min_bound = None

        # Gather all subfolder paths and corresponding files
        """
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                incomplete_path = os.path.join(subfolder_path, f"{subfolder}h.ply")
                ground_truth_path = os.path.join(subfolder_path, f"{subfolder}.ply")
                if os.path.exists(incomplete_path) and os.path.exists(ground_truth_path):
                    self.samples.append((incomplete_path, ground_truth_path))
        """

        lst = os.listdir(root_dir)  # your directory path
        number_files = len(lst)

        for i in range(math.floor(number_files / 2)):
            cut_file_path = os.path.join(root_dir, f"{i}_cut.ply")
            full_file_path = os.path.join(root_dir, f"{i}_full.ply")
            if os.path.exists(cut_file_path) and os.path.exists(full_file_path):
                self.samples.append((cut_file_path, full_file_path))

        print(f"Created dataset with {len(self.samples)} entries")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the incomplete and ground truth point clouds
        incomplete_path, ground_truth_path = self.samples[idx]

        incomplete_pcd = o3d.io.read_point_cloud(incomplete_path)
        ground_truth_pcd = o3d.io.read_point_cloud(ground_truth_path)

        incomplete_points = self.get_3d_tensor_from_pcd(incomplete_pcd)
        ground_truth_points = self.get_3d_tensor_from_pcd(ground_truth_pcd)

        # add one color channel for tensor format
        incomplete_points = incomplete_points.unsqueeze(0)
        ground_truth_points = ground_truth_points.unsqueeze(0)

        # apply gaussian filter as a alternative to signed distance fields
        #sigma = 1
        #incomplete_points_sdf = VoxelGridDataset.apply_gaussian_filter_with_preservation(incomplete_points, sigma)
        #ground_truth_points_sdf = VoxelGridDataset.apply_gaussian_filter_with_preservation(ground_truth_points, sigma)

        #return incomplete_points_sdf, ground_truth_points_sdf

        return incomplete_points, ground_truth_points

    @staticmethod
    def apply_gaussian_filter_with_preservation(voxel_grid, sigma):
        """
        Applies a Gaussian filter to a 3D voxel grid while preserving original values of 1.
        """
        # Create the 1D Gaussian kernel
        size = int(2 * (3 * sigma) + 1)  # Kernel size (3 sigma rule)
        x = torch.linspace(-size // 2, size // 2, steps=size)
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize to ensure sum of 1

        # Create the 3D Gaussian kernel
        kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)
        kernel_3d = kernel_3d / kernel_3d.sum()  # Normalize again for safety
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Apply Gaussian filter using conv3d
        padding = size // 2  # To retain the input size
        blurred = F.conv3d(voxel_grid, kernel_3d, padding=padding)
        mask = (voxel_grid == 1).float()  # Binary mask where the original values are 1

        # Combine the smoothed field with the original values
        result = mask * voxel_grid + (1 - mask) * blurred

        return result

    def get_3d_tensor_from_pcd(self, pcd):
        # Extract the points
        points = np.asarray(pcd.points)
        # Get the bounding box of the point cloud to determine the extent of the grid
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        # Define the resolution of the 3D grid (e.g., 50x50x50 grid)
        grid_size = 32  # TODO IN PARAMS
        voxel_size = (max_bound - min_bound) / grid_size

        self.voxel_size = voxel_size
        self.grid_min_bound = min_bound
        self.grid_min_bound = max_bound

        # Normalize the points to the grid space
        normalized_points = (points - min_bound) / voxel_size
        # Round the points to the nearest grid cell
        grid_points = np.floor(normalized_points).astype(int)
        # Clamp the values to ensure they stay within grid bounds
        grid_points = np.clip(grid_points, 0, grid_size - 1)
        # Create the 3D tensor (grid), initially filled with zeros
        grid_tensor = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.int32)
        # Mark the grid cells corresponding to the points as occupied (1)
        for point in grid_points:
            grid_tensor[tuple(point)] = 1
        return grid_tensor.float()


"""
TRAINING METHODS
"""

def validate_autoencoder(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for incomplete_pc, ground_truth_pc in dataloader:
            incomplete_pc = incomplete_pc.to(device)
            ground_truth_pc = ground_truth_pc.to(device)

            # Get the reconstructed point cloud
            reconstructed_pc = model(incomplete_pc)

            # Calculate the loss against the ground truth
            loss = masked_loss_binary(incomplete_pc, reconstructed_pc, ground_truth_pc)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_with_ground_truth(model, optimizer, train_loader, val_loader, epochs=50, device='cuda'):
    model.to(device)

    best_loss = float('inf')
    patience = 7

    train_losses = []
    val_losses = []

    print("starting epochs")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        batch_index = 0
        total_length = len(train_loader)

        for incomplete_pc, ground_truth_pc in train_loader:

            # process loging
            if batch_index % 50 == 0:
                print(f"epoch {epoch} progress {batch_index}/{total_length}")
            batch_index += 1

            incomplete_pc = incomplete_pc.to(device)
            ground_truth_pc = ground_truth_pc.to(device)

            # Reconstruct point cloud
            reconstructed_pc = model(incomplete_pc)

            # Compute loss against ground truth
            loss = masked_loss_binary(incomplete_pc, reconstructed_pc, ground_truth_pc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate_autoencoder(model, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Early stopping
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience = 7  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                print("EARLY STOPPING DUE TO MISSING PROGRESS")
                break

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print("saving epoch model")
        torch.save(model, f'./model_epoch_{epoch}.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_training.png')


"""
LOAD TRAINING SET
"""
 # Dataset root directory
root_dir = "../../../datasets/voxel_new"

# Create dataset and dataloaders
full_dataset = VoxelGridDataset(root_dir=root_dir, split="train")

#limit to 10k
#indices = np.random.choice(len(full_dataset), size=10000, replace=False)
#trainset_to_use = Subset(full_dataset, indices)
trainset_to_use = full_dataset

# Define the split ratio (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(trainset_to_use))
val_size = len(trainset_to_use) - train_size
print(f"train size: {train_size}, val size: {val_size}")
train_dataset, val_dataset = random_split(trainset_to_use, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

demo_entry = val_dataset[1]
print(np.shape(demo_entry))


"""
START ACTUAL TRAINING
"""

# Initialize model, optimizer
epochs = 25
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = MaskedPointCloudEncoderDecoder()
model.to(device)

# Train with grund truth
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_with_ground_truth(model, optimizer, train_loader, val_loader, epochs=epochs, device=device)
