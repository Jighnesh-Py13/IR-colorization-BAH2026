import torch
import torch.nn as nn

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN) for 2x Super-Resolution.
    Inputs: Low-Resolution single-band TIR image patch of shape (B, 1, 256, 256)
    Outputs: High-Resolution single-band TIR image patch of shape (B, 1, 512, 512)
    This model is lightweight and highly optimized for inference on RTX 3050.
    """
    def __init__(self, upscale_factor=2, in_channels=1, out_channels=1):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        
        # Layer 1: Feature Extraction
        # Extracts local features with a receptive field of 5x5.
        # Input shape:  (B, 1, H, W)
        # Output shape: (B, 64, H, W)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Non-Linear Mapping
        # Map low-resolution features to intermediate representations.
        # Input shape:  (B, 64, H, W)
        # Output shape: (B, 32, H, W)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Sub-pixel Feature Map Projection
        # Projects feature maps into upscaled sub-pixel space.
        # Output channels must be: out_channels * (upscale_factor ** 2)
        # For upscale_factor=2 and out_channels=1, output channel count is 4.
        # Input shape:  (B, 32, H, W)
        # Output shape: (B, 4, H, W)
        self.conv3 = nn.Conv2d(32, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        
        # PixelShuffle upscales spatial dimensions using the sub-pixel channel elements.
        # Input shape:  (B, 4, H, W)
        # Output shape: (B, 1, H * 2, W * 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        # Input tensor shape: (B, 1, 256, 256)
        
        # Step 1: Feature extraction (1 -> 64 channels)
        # Tensor shape flow: (B, 1, 256, 256) -> (B, 64, 256, 256)
        x = self.relu1(self.conv1(x))
        
        # Step 2: Non-linear mapping (64 -> 32 channels)
        # Tensor shape flow: (B, 64, 256, 256) -> (B, 32, 256, 256)
        x = self.relu2(self.conv2(x))
        
        # Step 3: Projection to sub-pixel channel maps (32 -> 4 channels)
        # Tensor shape flow: (B, 32, 256, 256) -> (B, 4, 256, 256)
        x = self.conv3(x)
        
        # Step 4: Reorganize sub-pixel channels to spatial grid (PixelShuffle 2x)
        # Tensor shape flow: (B, 4, 256, 256) -> (B, 1, 512, 512)
        x = self.pixel_shuffle(x)
        
        return x

# Alias for backward compatibility with existing training and inference scripts
PlaceholderSRModel = ESPCN
