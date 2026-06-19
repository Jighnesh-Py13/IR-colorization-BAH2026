import torch
import torch.nn as nn

class PlaceholderSRModel(nn.Module):
    """
    Placeholder model for the 200m -> 100m Super-Resolution task.
    Will be replaced with actual ESPCN/SRResNet architecture.
    """
    def __init__(self, upscale_factor=2):
        super(PlaceholderSRModel, self).__init__()
        self.upscale_factor = upscale_factor
        # Minimal layer to verify pipeline
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Dummy upscale by repeating pixels to match the target shape
        x = self.conv(x)
        return nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='nearest')
