import torch
import torch.nn as nn

class PlaceholderColorModel(nn.Module):
    """
    Placeholder model for the 100m TIR -> 100m RGB Colorization task.
    Will be replaced with U-Net or Pix2Pix.
    """
    def __init__(self):
        super(PlaceholderColorModel, self).__init__()
        # Map 1 channel input to 3 channels output
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)
