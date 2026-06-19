import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class SuperResolutionDataset(Dataset):
    """
    Dataset class for the 200m -> 100m Super-Resolution task.
    Loads co-registered 'tir_200m.npy' (input) and 'tir_100m_512.npy' (target) patches.
    """
    def __init__(self, patches_dir, transform=None):
        """
        Args:
            patches_dir (str): Path to output/patches/ containing product sample directories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patches_dir = patches_dir
        self.transform = transform
        self.sample_paths = self._find_samples()

    def _find_samples(self):
        # Locate all sample folders under product directories
        return glob.glob(os.path.join(self.patches_dir, '*', 'sample_*'))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        
        # Load pre-saved numpy arrays
        lr_tir = np.load(os.path.join(sample_path, 'tir_200m.npy'))
        hr_tir = np.load(os.path.join(sample_path, 'tir_100m_512.npy'))
        
        # Ensure channel dimension is present: (H, W) -> (1, H, W)
        if lr_tir.ndim == 2:
            lr_tir = np.expand_dims(lr_tir, axis=0)
        if hr_tir.ndim == 2:
            hr_tir = np.expand_dims(hr_tir, axis=0)

        # Convert to float32 Tensor
        lr_tir = torch.from_numpy(lr_tir).float()
        hr_tir = torch.from_numpy(hr_tir).float()

        sample = {'lr_tir': lr_tir, 'hr_tir': hr_tir}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ColorizationDataset(Dataset):
    """
    Dataset class for the 100m TIR -> 100m RGB Colorization task.
    Loads co-registered 'tir_100m_512.npy' (input) and 'rgb_100m_512.npy' (target) patches.
    """
    def __init__(self, patches_dir, transform=None):
        """
        Args:
            patches_dir (str): Path to output/patches/ containing product sample directories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patches_dir = patches_dir
        self.transform = transform
        self.sample_paths = self._find_samples()

    def _find_samples(self):
        return glob.glob(os.path.join(self.patches_dir, '*', 'sample_*'))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        
        # Load pre-saved numpy arrays
        tir_100m = np.load(os.path.join(sample_path, 'tir_100m_512.npy'))
        rgb_100m = np.load(os.path.join(sample_path, 'rgb_100m_512.npy'))
        
        # Ensure channel dimension is present for TIR: (H, W) -> (1, H, W)
        if tir_100m.ndim == 2:
            tir_100m = np.expand_dims(tir_100m, axis=0)
            
        # RGB is already saved as (3, H, W) by merge/downscale scripts, but ensure format
        if rgb_100m.ndim == 2:
            # Fallback if 2D
            rgb_100m = np.expand_dims(rgb_100m, axis=0)

        # Convert to float32 Tensor
        tir_100m = torch.from_numpy(tir_100m).float()
        rgb_100m = torch.from_numpy(rgb_100m).float()

        sample = {'tir_100m': tir_100m, 'rgb_100m': rgb_100m}

        if self.transform:
            sample = self.transform(sample)

        return sample
