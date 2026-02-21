import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path
import json

from dataset import GPSDataset
from model import SiameseTracker
from config import ARTIFACTS_DIR, TRAIN_CONFIG

# Enable CuDNN benchmark for speed
torch.backends.cudnn.benchmark = True

import argparse
import sys

# ... imports ...

def train(resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    audit_report = ARTIFACTS_DIR / "gps_audit_report.json"
    train_ds = GPSDataset(audit_report, split="train")
    val_ds = GPSDataset(audit_report, split="val")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True
    )
    
    # Model Init
    model = SiameseTracker().to(device)
    
    start_epoch = 0
    checkpoint = None
    ckpt_path = ARTIFACTS_DIR / "last_model.pt"
    backup_path = ARTIFACTS_DIR / "last_model.bak"

    # Resume Logic
    if resume:
        print(f"Attempting to resume...")
        
        # Try primary checkpoint
        loaded_path = None
        if ckpt_path.exists():
            try:
                print(f"Checking integrity of {ckpt_path}...")
                # Map to CPU to avoid GPU OOM if file is weird, and verify structure
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                loaded_path = ckpt_path
            except Exception as e:
                print(f"Warning: {ckpt_path} seems corrupted: {e}")
        
        # Try backup if primary failed or didn't exist
        if loaded_path is None and backup_path.exists():
            try:
                print(f"Trying backup checkpoint {backup_path}...")
                checkpoint = torch.load(backup_path, map_location="cpu")
                loaded_path = backup_path
            except Exception as e:
                print(f"Warning: Backup {backup_path} also corrupted: {e}")

        # Load if successful
        if checkpoint is not None:
            print(f"Successfully loaded {loaded_path}")
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming at epoch {start_epoch}")
        else:
             print("No valid checkpoint found. Starting fresh.")


    
    # Optional: compile (PyTorch 2.0)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile!")
    except:
        pass
        
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=TRAIN_CONFIG["epochs"])
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=TRAIN_CONFIG["mixed_precision"])

    # Load optimizer/scheduler state if resuming.
    # Older checkpoints may not contain every key.
    if resume and checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: could not load optimizer state: {e}")
        else:
            print("Warning: checkpoint missing optimizer_state_dict; continuing without it.")

        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: could not load scheduler state: {e}")
        else:
            print("Warning: checkpoint missing scheduler_state_dict; continuing without it.")

    # Training Loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        for img1, img2, target in pbar:
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP (Mixed Precision)
            with torch.cuda.amp.autocast(enabled=TRAIN_CONFIG["mixed_precision"]):
                output = model(img1, img2)
                loss = criterion(output, target)
            
            # Scaler for backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, target in val_loader:
                img1, img2, target = img1.to(device), img2.to(device), target.to(device)
                output = model(img1, img2)
                loss = criterion(output, target)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save Last Model (with backup rotation)
        last_ckpt_path = ARTIFACTS_DIR / "last_model.pt"
        backup_ckpt_path = ARTIFACTS_DIR / "last_model.bak"
        
        # Rotate: last -> bak
        if last_ckpt_path.exists():
            try:
                 import shutil
                 shutil.copy2(last_ckpt_path, backup_ckpt_path)
            except Exception as e:
                print(f"Warning: could not create backup: {e}")

        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, last_ckpt_path)
        except Exception as e:
            print(f"CRITICAL: Failed to save checkpoint at epoch {epoch}: {e}")


        # Checkpoint Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save raw weights for best to keep it simple for inference
            torch.save(model.state_dict(), ARTIFACTS_DIR / "best_model.pt")
            print("Saved Best Model")
            
    # Export ONNX
    print("Exporting ONNX...")
    model.eval()
    dummy_input1 = torch.randn(1, 3, *TRAIN_CONFIG["img_size"]).to(device)
    dummy_input2 = torch.randn(1, 3, *TRAIN_CONFIG["img_size"]).to(device)
    onnx_path = ARTIFACTS_DIR / "gps_model.onnx"
    try:
        torch.onnx.export(
            model, 
            (dummy_input1, dummy_input2), 
            onnx_path,
            input_names=["img1", "img2"], 
            output_names=["delta_translation"],
            opset_version=11
        )
        print(f"Exported to {onnx_path}")
    except Exception as e:
        print(f"ONNX Export failed: {e}")

    # Plotting (User requested graphs)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(ARTIFACTS_DIR / "loss_plot.png")
        print(f"Loss plot saved to {ARTIFACTS_DIR}/loss_plot.png")
    except ImportError:
        print("matplotlib not installed, skipping plot.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
