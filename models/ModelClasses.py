import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

# ==================================================================================================
#  Small U-Net Auto Encoder
# ==================================================================================================
class SmallUnetAutoEncoder(nn.Module):
    """
    A simple 3D CNN-based encoder-decoder model.
    Input/Output shape (B, 1, 64, 128, 64).
    """
    def __init__(self):
        super(SmallUnetAutoEncoder, self).__init__()

        # ======= Encoder Path =====================================================================
        self.enc1 = self.double_conv(1, 8)  # Input channels = 1, output channels = 8
        self.enc2 = self.double_conv(8, 16)
        self.enc3 = self.double_conv(16, 32)
        self.enc4 = self.double_conv(32, 64)
        
        # ======= Downsampling Layers ==============================================================
        self.pool = nn.MaxPool3d(2)
        
        # ======= Bottleneck =======================================================================
        self.bottleneck = self.double_conv(64, 128)
        
        # ======= Decoder Path =====================================================================
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(128, 64)
        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(64, 32)
        self.upconv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(32, 16)
        self.upconv1 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(16, 8)
        
        # ======= Final Output Layer ===============================================================
        self.final = nn.Conv3d(8, 1, kernel_size=1)  # Output channels = 1
        
        # ======= Sigmoid ==========================================================================
        self.sigmoid = nn.Sigmoid()

    # ======= Double Convolution ===================================================================
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    # ======= Forward Pass =========================================================================
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))
        
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))
        
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        
        # Final output layer and sigmoid
        out = self.final(dec1)
        probabilities = self.sigmoid(out)
        return probabilities
    

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
        self.enc1 = self.double_conv(1, 16)  # Input channels = 1, output channels = 16
        self.enc2 = self.double_conv(16, 32)
        self.enc3 = self.double_conv(32, 64)
        self.enc4 = self.double_conv(64, 128)
        
        # Downsampling layers
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self.double_conv(128, 256)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(256, 128)
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(128, 64)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(64, 32)
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(32, 16)
        
        # Final output layer
        self.final = nn.Conv3d(16, 1, kernel_size=1)  # Output channels = 1

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
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))
        
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))
        
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        
        # Final output layer
        out = self.final(dec1)  # Output: (1, 64, 128, 64)
        
        return self.sigmoid(out)