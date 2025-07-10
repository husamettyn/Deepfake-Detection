#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.vgg import vgg11_bn
from env import SINGULAR_DIR
from tqdm import tqdm # Import tqdm
from dataloader import SingularFlowDataset

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    # Wrap the loader with tqdm for a progress bar
    # leave=False removes the bar after completion
    progress_bar = tqdm(loader, desc="Training", leave=False, unit="batch")
    for x,y in progress_bar:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * x.size(0)
        progress_bar.set_postfix(loss=f"{batch_loss:.4f}") # Update bar with current loss
        
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    # Wrap the loader with tqdm for a progress bar
    progress_bar = tqdm(loader, desc="Evaluating", leave=False, unit="batch")
    for x,y in progress_bar:
        x,y = x.to(device), y.to(device)
        pred = model(x) # pred is now a raw logit
        total_loss += criterion(pred, y).item() * x.size(0)
        
        # --- CHANGE 1: Accuracy calculation based on logits ---
        # A positive logit corresponds to a probability > 0.5 (predicted class 1)
        # A negative logit corresponds to a probability < 0.5 (predicted class 0)
        correct += ((pred > 0).float() == y).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--singular_dir', default=SINGULAR_DIR,
                        help="Path to singular flow files produced by generate_singular_flows.py")
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--save_dir', type=str, default='saved_vgg11')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Normalize(mean=[0.5,0.5], std=[0.5,0.5])

    train_ds = SingularFlowDataset(args.singular_dir, split='train',
                                   train_ratio=args.train_ratio,
                                   transform=transform)
    val_ds   = SingularFlowDataset(args.singular_dir, split='val',
                                   train_ratio=args.train_ratio,
                                   transform=transform)

    # Note: If you get the shared memory error again, set num_workers=0 or restart
    # docker with --shm-size=8g
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = vgg11_bn(pretrained=False, num_classes=1, in_channels=2).to(device)
    
    # --- CHANGE 2: Use the numerically stable loss function ---
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    # The main loop now clearly separates the progress of each part
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        # This print statement now acts as a summary after the progress bars are done
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch, 'val_acc': val_acc},
                       os.path.join(args.save_dir, 'best_vgg11.pth'))
            print(" → Saved new best model")

    # final checkpoint
    torch.save({'state_dict': model.state_dict()},
               os.path.join(args.save_dir, 'final_vgg11.pth'))

if __name__ == '__main__':
    main()