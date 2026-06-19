import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.datasets import ColorizationDataset
from src.models.colorization import PlaceholderColorModel

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
    patches_dir = os.path.join('output', 'patches')
    epochs = 1
    batch_size = 4
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Dataset and DataLoader
    dataset = ColorizationDataset(patches_dir=patches_dir)
    if len(dataset) == 0:
        print(f"No patches found in {patches_dir}. Please run driver.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model, Loss, Optimizer
    model = PlaceholderColorModel().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Simple training loop
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        
    # Save checkpoint
    os.makedirs('weights', exist_ok=True)
    checkpoint_path = os.path.join('weights', 'color_placeholder.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved weights to {checkpoint_path}")

if __name__ == '__main__':
    main()
