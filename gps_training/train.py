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

# PyTorch 2.x A100 Accelerators
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import argparse
import sys
import os
import atexit
import fcntl

# ... imports ...


def _acquire_file_lock(lock_path: Path) -> int:
    os.makedirs(lock_path.parent, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd: int, lock_path: Path) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    try:
        os.remove(lock_path)
    except OSError:
        pass

def kill_gpu_hogs():
    """Clear GPU memory before training."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _is_checkpoint_valid(ckpt_path: Path) -> bool:
    """Check if a PyTorch checkpoint is readable and not corrupt."""
    import torch
    if not ckpt_path.exists() or ckpt_path.stat().st_size < 1024 * 50:  # < 50KB is suspicious for ResNet
        return False
    try:
        torch.load(ckpt_path, map_location='cpu')
        return True
    except Exception as e:
        print(f"⚠️ Checkpoint {ckpt_path.name} is corrupt: {e}", flush=True)
        return False

def resolve_scheduler_max_lr(train_config: dict) -> float:
    """Resolve max_lr from config and enforce a safe LR relationship."""
    base_lr = float(train_config["learning_rate"])
    max_lr = float(train_config.get("max_lr", base_lr))
    if max_lr < base_lr:
        raise ValueError(
            f"Invalid LR config: max_lr ({max_lr}) cannot be lower than learning_rate ({base_lr})."
        )
    return max_lr

def print_training_config(train_args: dict):
    """Print the full training configuration so it's visible in logs."""
    print(f"\n{'─'*60}", flush=True)
    print(f"  📋 TRAINING CONFIGURATION")
    print(f"{'─'*60}")
    for k, v in train_args.items():
        print(f"  {k:<20}: {v}")
    print(f"{'─'*60}\n", flush=True)

def _sync_results_to_drive(artifacts_dir: Path):
    """Best-effort sync from local SSD runs to Drive."""
    drive_runs = os.environ.get("UAV_PROJECT_DIR")
    if drive_runs and str(artifacts_dir).startswith("/content"):
        drive_results = os.path.join(drive_runs, "gps_model")
        os.makedirs(drive_results, exist_ok=True)
        print(f"\n☁️  Syncing results to Drive: {drive_results}", flush=True)
        _sync_cmd = f'rsync -a --info=progress2 "{artifacts_dir}/" "{drive_results}/"'
        import subprocess
        subprocess.run(_sync_cmd, shell=True, check=False)
        print(f"  ✓ Results synced to Drive", flush=True)

def train(epochs=None, batch=None, device=None, resume=False):
    kill_gpu_hogs()
    
    lock_path = ARTIFACTS_DIR / ".gps_train.lock"
    lock_fd = _acquire_file_lock(lock_path)
    atexit.register(_release_file_lock, lock_fd, lock_path)

    # CLI Overrides
    if epochs is not None: TRAIN_CONFIG["epochs"] = epochs
    if batch is not None: TRAIN_CONFIG["batch_size"] = batch
    if device is not None: TRAIN_CONFIG["device"] = device

    print_training_config(TRAIN_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    audit_report = ARTIFACTS_DIR / "gps_audit_report.json"
    train_ds = GPSDataset(audit_report, split="train")
    val_ds = GPSDataset(audit_report, split="val")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    def collate_drop_none(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            raise RuntimeError("All samples in batch failed to load (empty batch).")
        img1, img2, delta = zip(*batch)
        import torch
        return torch.stack(img1, 0), torch.stack(img2, 0), torch.stack(delta, 0)

    w_count = TRAIN_CONFIG["num_workers"]
    train_loader = DataLoader(
        train_ds, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=w_count,
        pin_memory=True,
        persistent_workers=True if w_count > 0 else False,
        prefetch_factor=4 if w_count > 0 else None,
        drop_last=True,
        worker_init_fn=seed_worker,
        collate_fn=collate_drop_none
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=collate_drop_none
    )
    
    # Model Init
    model = SiameseTracker().to(device)
    
    start_epoch = 0
    checkpoint = None
    ckpt_path = ARTIFACTS_DIR / "last_model.pt"
    backup_path = ARTIFACTS_DIR / "last_model.bak"
    
    if resume:
        print(f"Attempting to resume...")
        if ckpt_path.exists() and _is_checkpoint_valid(ckpt_path):
            print(f"Resuming from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
        else:
            print("❌ last_model.pt is corrupt or missing! Checking backup...", flush=True)
            if backup_path.exists() and _is_checkpoint_valid(backup_path):
                 print(f"🔄 Resuming from backup {backup_path}...", flush=True)
                 checkpoint = torch.load(backup_path, map_location=device)
            else:
                 print("⚠️ Backup is also missing or corrupt. Starting fresh training.", flush=True)
                 resume = False
    
    if resume and checkpoint is not None:
        print("Successfully loaded checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming at epoch {start_epoch}")    
    # Optional: compile (PyTorch 2.0)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile!")
    except:
        pass
        
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    max_lr = resolve_scheduler_max_lr(TRAIN_CONFIG)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=TRAIN_CONFIG["epochs"],
    )
    criterion = nn.MSELoss()

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
            img1 = img1.to(device, non_blocking=True).float() / 255.0
            img2 = img2.to(device, non_blocking=True).float() / 255.0
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # A100 Native BFloat16 without GradScaler
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=TRAIN_CONFIG["mixed_precision"]):
                output = model(img1, img2)
                loss = criterion(output, target)
            
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, target in val_loader:
                img1 = img1.to(device, non_blocking=True).float() / 255.0
                img2 = img2.to(device, non_blocking=True).float() / 255.0
                target = target.to(device, non_blocking=True)
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

        tmp_ckpt_path = last_ckpt_path.with_suffix('.pt.tmp')
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, tmp_ckpt_path)
            import os
            os.replace(tmp_ckpt_path, last_ckpt_path)
        except Exception as e:
            print(f"CRITICAL: Failed to save checkpoint atomically at epoch {epoch}: {e}")
            if tmp_ckpt_path.exists():
                tmp_ckpt_path.unlink()


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
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    # Final Drive Sync
    _sync_results_to_drive(ARTIFACTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Tracker on GPS dataset")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=str, help="Batch size (int)")
    parser.add_argument("--device", type=str, help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    batch_val = args.batch
    if batch_val is not None:
        try:
            batch_val = int(batch_val)
        except ValueError:
            pass

    train(
        epochs=args.epochs,
        batch=batch_val,
        device=args.device,
        resume=args.resume
    )
