import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

# ==================================================================================================
#  Voxel Auto Encoder
# ==================================================================================================
class VoxelAutoEncoder(nn.Module):
    """
    A simple 3D CNN-based encoder-decoder model.
    Input/Output shape (B, 1, 64, 128, 64).
    """
    def __init__(self):
        super(VoxelAutoEncoder, self).__init__()

        # Encoder path
        self.enc1 = self.double_conv(1, 32)  # Input channels = 1, output channels = 32
        self.enc2 = self.double_conv(32, 64)
        self.enc3 = self.double_conv(64, 128)
        self.enc4 = self.double_conv(128, 256)
        
        # Downsampling layers
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self.double_conv(256, 512)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(512, 256)  # Concatenate + Conv
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(64, 32)
        
        # Final output layer
        self.final = nn.Conv3d(32, 1, kernel_size=1)  # Output channels = 1 (filled voxel volume)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # Output: (32, 64, 128, 64)
        enc2 = self.enc2(self.pool(enc1))  # Output: (64, 32, 64, 32)
        enc3 = self.enc3(self.pool(enc2))  # Output: (128, 16, 32, 16)
        enc4 = self.enc4(self.pool(enc3))  # Output: (256, 8, 16, 8)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # Output: (512, 4, 8, 4)
        
        # Decoder
        dec4 = self.upconv4(bottleneck)  # Output: (256, 8, 16, 8)
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))  # Skip connection
        
        dec3 = self.upconv3(dec4)  # Output: (128, 16, 32, 16)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))  # Skip connection
        
        dec2 = self.upconv2(dec3)  # Output: (64, 32, 64, 32)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))  # Skip connection
        
        dec1 = self.upconv1(dec2)  # Output: (32, 64, 128, 64)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))  # Skip connection
        
        # Final output layer
        out = self.final(dec1)  # Output: (1, 64, 128, 64)
        
        return self.sigmoid(out)