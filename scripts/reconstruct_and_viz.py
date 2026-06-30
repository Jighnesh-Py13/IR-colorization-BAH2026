import os
import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
import subprocess
from utils.visualization import percentile_stretch

def main():
    print("Starting reconstruction and visualization script...")
    
    # 1. Create input/mock_product folder if not exists
    os.makedirs(os.path.join('input', 'mock_product'), exist_ok=True)
    
    # 2. Load the 200m TIR patch from sample_006
    sample_dir = os.path.join('output', 'patches', 'demo', 'sample_006')
    tir_200m_npy_path = os.path.join(sample_dir, 'tir_200m.npy')
    tir_200m = np.load(tir_200m_npy_path)
    
    # Save it as input/mock_product/mock_product_B10.tif
    tir_200m_tif_path = os.path.join('input', 'mock_product', 'mock_product_B10.tif')
    # Save as uint16 TIFF to match Landsat 9 B10 format
    tifffile.imwrite(tir_200m_tif_path, tir_200m.astype(np.uint16))
    print(f"Saved {tir_200m_tif_path} with shape {tir_200m.shape}")
    
    # 3. Run inference using python -m src.inference
    command = [
        'python', '-m', 'src.inference',
        '--product_id', 'mock_product',
        '--input_dir', 'input/mock_product',
        '--sr_weights', 'weights/espcn.pth',
        '--color_weights', 'weights/color_unet.pth',
        '--color_base_channels', '32'
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    
    # 4. Load outputs and targets for visualization
    # Inputs/Targets:
    tir_200m = np.load(os.path.join(sample_dir, 'tir_200m.npy')).squeeze()
    tir_100m_gt = np.load(os.path.join(sample_dir, 'tir_100m_512.npy')).squeeze()
    rgb_100m_gt = np.load(os.path.join(sample_dir, 'rgb_100m_512.npy')) # Shape (3, 512, 512)
    
    # Outputs:
    sr_out = tifffile.imread(os.path.join('output', 'model_outputs', 'tir_superresolved_100m', 'mock_product.tif')).squeeze()
    color_out = tifffile.imread(os.path.join('output', 'model_outputs', 'colorized_tir_100m', 'mock_product.tif')) # Shape (3, 512, 512)
    
    print("Loaded data:")
    print(f"tir_200m shape: {tir_200m.shape}, range: [{tir_200m.min()}, {tir_200m.max()}]")
    print(f"tir_100m_gt shape: {tir_100m_gt.shape}, range: [{tir_100m_gt.min()}, {tir_100m_gt.max()}]")
    print(f"sr_out shape: {sr_out.shape}, range: [{sr_out.min()}, {sr_out.max()}]")
    print(f"rgb_100m_gt shape: {rgb_100m_gt.shape}, range: [{rgb_100m_gt.min()}, {rgb_100m_gt.max()}]")
    print(f"color_out shape: {color_out.shape}, range: [{color_out.min()}, {color_out.max()}]")
    
    # 5. Prepare images for plotting
    # Stretch inputs using percentile stretch
    tir_200m_stretched = percentile_stretch(tir_200m)
    tir_100m_gt_stretched = percentile_stretch(tir_100m_gt)
    sr_out_stretched = percentile_stretch(sr_out)
    
    # GT RGB is loaded as BGR (B2, B3, B4). For matplotlib, transpose to (H, W, 3) and convert to RGB
    # BGR (idx 0=B, 1=G, 2=R) -> RGB (idx 2=R, 1=G, 0=B)
    rgb_gt_plt = np.stack([rgb_100m_gt[2], rgb_100m_gt[1], rgb_100m_gt[0]], axis=-1)
    rgb_gt_stretched = percentile_stretch(rgb_gt_plt)
    
    # Grayscale TIR (for context) - just the ground truth TIR stretched
    tir_gray_stretched = percentile_stretch(tir_100m_gt)
    
    # Output colorized image channel ordering:
    # Let's inspect the output. The inference script reorders:
    # blue = color_out[2], green = color_out[1], red = color_out[0], color_out_final = B, G, R
    # Since tifffile saves it as BGR, when we read it, color_out is BGR.
    # So to show it in matplotlib, we map BGR -> RGB:
    color_out_plt = np.stack([color_out[2], color_out[1], color_out[0]], axis=-1)
    color_out_stretched = percentile_stretch(color_out_plt)
    
    # 6. Generate the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Super-Resolution
    # Col 1: Low-Res Input
    im1 = axes[0, 0].imshow(tir_200m_stretched, cmap='magma')
    axes[0, 0].set_title("Input: Raw Low-Res TIR (200m)\n[Shape: 256x256]", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Col 2: High-Res Target
    im2 = axes[0, 1].imshow(tir_100m_gt_stretched, cmap='magma')
    axes[0, 1].set_title("Target: Ground Truth TIR (100m)\n[Shape: 512x512]", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Col 3: ESPCN Output
    im3 = axes[0, 2].imshow(sr_out_stretched, cmap='magma')
    axes[0, 2].set_title("Output: Super-Resolved TIR (ESPCN)\n[Shape: 512x512]", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Colorization
    # Col 1: Grayscale Input for context
    axes[1, 0].imshow(tir_gray_stretched, cmap='gray')
    axes[1, 0].set_title("Input: Raw TIR Grayscale\n(For Context)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Col 2: RGB Target
    axes[1, 1].imshow(rgb_gt_stretched)
    axes[1, 1].set_title("Target: Ground Truth RGB (100m)\n[Shape: 512x512]", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Col 3: U-Net Output
    axes[1, 2].imshow(color_out_stretched)
    axes[1, 2].set_title("Output: Colorized RGB (U-Net)\n[Shape: 512x512]", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('output', exist_ok=True)
    comparison_path = os.path.join('output', 'results_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated comparison plot at: {comparison_path}")

if __name__ == '__main__':
    main()
