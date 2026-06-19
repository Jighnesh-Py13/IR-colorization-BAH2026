import os
import argparse
import numpy as np
import torch
import tifffile
from utils.file_utils import find_file
from utils.logging_utils import setup_logging
from src.models.super_resolution import PlaceholderSRModel
from src.models.colorization import PlaceholderColorModel

def run_inference(product_id, input_dir, output_dir, sr_model, color_model, device):
    """
    Executes the joint inference pipeline on a single product.
    1. Loads the 200m TIR image.
    2. Passes it through the SR model to produce 100m TIR.
    3. Passes the 100m TIR through the Colorization model to produce 100m RGB.
    4. Saves results in the mandated submission structure under output/model_outputs/.
    """
    logger = setup_logging('inference', output_dir)
    logger.info(f"Running inference for product: {product_id}")
    
    # 1. Locate the 200m TIR band B10
    tir_200m_path = find_file(input_dir, '_B10')
    if not tir_200m_path:
        logger.error(f"Could not find band B10 image under {input_dir}")
        return
        
    tir_200m_img = tifffile.imread(tir_200m_path)
    h, w = tir_200m_img.shape[-2:]
    
    # Convert image data to tensor
    img_tensor = torch.from_numpy(tir_200m_img).float().to(device)
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add Batch and Channel dimensions (1, 1, H, W)
    elif img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)              # (1, C, H, W)
        
    # 2. Run Super-Resolution
    sr_model.eval()
    with torch.no_grad():
        sr_out_tensor = sr_model(img_tensor)
        
    # 3. Run Colorization
    color_model.eval()
    with torch.no_grad():
        color_out_tensor = color_model(sr_out_tensor)
        
    # Convert tensors back to numpy arrays
    sr_out = sr_out_tensor.squeeze().cpu().numpy()
    color_out = color_out_tensor.squeeze().cpu().numpy()
    
    # Define and create output folders
    sr_out_dir = os.path.join(output_dir, 'model_outputs', 'tir_superresolved_100m')
    color_out_dir = os.path.join(output_dir, 'model_outputs', 'colorized_tir_100m')
    os.makedirs(sr_out_dir, exist_ok=True)
    os.makedirs(color_out_dir, exist_ok=True)
    
    # 4. Save results to disk
    sr_out_path = os.path.join(sr_out_dir, f'{product_id}.tif')
    color_out_path = os.path.join(color_out_dir, f'{product_id}.tif')
    
    # Ensure standard band ordering for colorized RGB: Blue (Layer 1), Green (Layer 2), Red (Layer 3)
    # If the network output was generated as (Red, Green, Blue), we reorder it here:
    if color_out.ndim == 3 and color_out.shape[0] == 3:
        # Reorder channels: Red (idx 0), Green (idx 1), Blue (idx 2) -> Blue, Green, Red
        # Note: adjust this logic depending on how you train your colorization model output channel mapping!
        blue = color_out[2]
        green = color_out[1]
        red = color_out[0]
        color_out_final = np.stack([blue, green, red], axis=0)
    else:
        color_out_final = color_out
        
    tifffile.imwrite(sr_out_path, sr_out.astype(np.float32))
    tifffile.imwrite(color_out_path, color_out_final.astype(np.float32))
    
    logger.info(f"Super-Resolved TIFF saved to: {sr_out_path}")
    logger.info(f"Colorized TIFF saved to: {color_out_path}")

def main():
    parser = argparse.ArgumentParser(description='Joint Inference Pipeline (SR + Colorization)')
    parser.add_argument('--product_id', type=str, required=True, help='Product ID to process.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the product band images.')
    parser.add_argument('--output_dir', type=str, default='output', help='Output base directory.')
    parser.add_argument('--sr_weights', type=str, default=None, help='Path to SR model weights.')
    parser.add_argument('--color_weights', type=str, default=None, help='Path to colorization model weights.')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    sr_model = PlaceholderSRModel().to(device)
    color_model = PlaceholderColorModel().to(device)
    
    if args.sr_weights and os.path.exists(args.sr_weights):
        sr_model.load_state_dict(torch.load(args.sr_weights, map_location=device))
    if args.color_weights and os.path.exists(args.color_weights):
        color_model.load_state_dict(torch.load(args.color_weights, map_location=device))
        
    run_inference(args.product_id, args.input_dir, args.output_dir, sr_model, color_model, device)

if __name__ == '__main__':
    main()
