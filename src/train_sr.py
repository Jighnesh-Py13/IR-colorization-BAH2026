import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.datasets import SuperResolutionDataset
from src.models.super_resolution import ESPCN

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single epoch of training.
    """
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        lr_tir = batch['lr_tir'].to(device)
        hr_tir = batch['hr_tir'].to(device)
        
        # Normalize inputs and targets to [0, 1]
        lr_tir_norm = (lr_tir - 22474.0) / (26186.0 - 22474.0)
        hr_tir_norm = (hr_tir - 22459.0) / (26593.0 - 22459.0)
        
        optimizer.zero_grad()
        outputs = model(lr_tir_norm)
        loss = criterion(outputs, hr_tir_norm)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * lr_tir.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    """
    Main training script entry point for the Super-Resolution model.
    """
    # Configuration
    parser = argparse.ArgumentParser(description='Train ESPCN Super-Resolution Model')
    parser.add_argument('--patches_dir', type=str, default=os.path.join('output', 'patches'),
                        help='Path to output/patches/ containing product sample directories.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save trained weights.')
    parser.add_argument('--checkpoint_name', type=str, default='espcn.pth', help='Filename for the saved model weights.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Dataset and DataLoader
    dataset = SuperResolutionDataset(patches_dir=args.patches_dir)
    if len(dataset) == 0:
        print(f"No patches found in {args.patches_dir}. Please run driver.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded dataset with {len(dataset)} samples. Batch size: {args.batch_size}")
    
    # Model, Loss, Optimizer
    model = ESPCN(upscale_factor=2, in_channels=1, out_channels=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Simple training loop
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.6f}")
        
    # Save checkpoint
    os.makedirs(args.weights_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.weights_dir, args.checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved weights to {checkpoint_path}")

if __name__ == '__main__':
    main()
