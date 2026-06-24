import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.datasets import ColorizationDataset
from src.models.colorization import UNet

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single epoch of training.
    """
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        tir_100m = batch['tir_100m'].to(device)
        rgb_100m = batch['rgb_100m'].to(device)
        
        optimizer.zero_grad()
        outputs = model(tir_100m)
        loss = criterion(outputs, rgb_100m)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * tir_100m.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    """
    Main training script entry point for the Colorization model.
    """
    # Configuration
    parser = argparse.ArgumentParser(description='Train U-Net Colorization Model')
    parser.add_argument('--patches_dir', type=str, default=os.path.join('output', 'patches'),
                        help='Path to output/patches/ containing product sample directories.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels for U-Net architecture.')
    parser.add_argument('--no_bilinear', action='store_true', help='Disable bilinear upsampling (uses ConvTranspose2d instead).')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save trained weights.')
    parser.add_argument('--checkpoint_name', type=str, default='color_unet.pth', help='Filename for the saved model weights.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Dataset and DataLoader
    dataset = ColorizationDataset(patches_dir=args.patches_dir)
    if len(dataset) == 0:
        print(f"No patches found in {args.patches_dir}. Please run driver.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded dataset with {len(dataset)} samples. Batch size: {args.batch_size}")
    
    # Model, Loss, Optimizer
    model = UNet(
        in_channels=1, 
        out_channels=3, 
        base_channels=args.base_channels, 
        bilinear=not args.no_bilinear
    ).to(device)
    
    # Print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net initialized with base_channels={args.base_channels}, bilinear={not args.no_bilinear}")
    print(f"Total trainable parameters: {total_params:,}")
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Simple training loop
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.6f}")
        
    # Save checkpoint
    os.makedirs(args.weights_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.weights_dir, args.checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved weights to {checkpoint_path}")

if __name__ == '__main__':
    main()
