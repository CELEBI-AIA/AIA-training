import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it using:")
    print("  pip install ultralytics")
    sys.exit(1)

from config import ARTIFACTS_DIR, DATASET_DIR, TRAIN_CONFIG
from build_dataset import build_dataset
import subprocess
import os
import time

# New function to kill GPU hogs
def kill_gpu_hogs():
    # This function is intended to kill processes that might be holding GPU memory.
    # The actual implementation details are not provided in this change request,
    # so it's left as a placeholder.
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train(epochs=None, batch=None, device=None, model_path=None, resume=False):
    kill_gpu_hogs()
    
    # User requested to include dataset optimization in the training script
    # We run it here to ensure data is always fresh and optimized
    # Auto-optimize dataset if not resuming
    if not resume:
        print("🔄 Optimizing Dataset (Smart Downsampling)...")
        build_dataset()

    # Use config values if not provided
    epochs = epochs if epochs is not None else TRAIN_CONFIG['epochs']
    batch = batch if batch is not None else TRAIN_CONFIG['batch']
    device = device if device is not None else TRAIN_CONFIG['device']
    model_path = model_path if model_path is not None else TRAIN_CONFIG['model']

    # Resume logic
    if resume:
        # Auto-detect latest checkpoint
        project_results = TRAIN_CONFIG['project']
        if not project_results.exists():
             print(f"Error: Project directory {project_results} not found.")
             sys.exit(1)
             
        # Find all last.pt files in the project directory
        checkpoints = list(project_results.rglob("last.pt"))
        if not checkpoints:
            print(f"Error: No 'last.pt' found in {project_results}")
            sys.exit(1)
            
        # Sort by modification time (newest first)
        latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Found latest checkpoint: {latest_ckpt}")
        
        model_path = latest_ckpt
    else:
        print(f"Starting training for {model_path} on UAV dataset...")

    print(f"Epochs: {epochs}, Batch: {batch}, Device: {device}")

    # Load a model
    model = YOLO(model_path)

    # Path to dataset config
    yaml_path = DATASET_DIR / "dataset.yaml"
    
    if not yaml_path.exists():
        print(f"Error: Dataset config not found at {yaml_path}")
        print("Please run build_dataset.py first.")
        sys.exit(1)

    # Train the model
    try:
        train_args = {
            "data": str(yaml_path),
            "epochs": epochs,
            "batch": batch,
            "imgsz": TRAIN_CONFIG['imgsz'],
            "device": device,
            "project": str(TRAIN_CONFIG['project']),
            "name": TRAIN_CONFIG['name'],
            "exist_ok": True,
            "verbose": True,
            "workers": TRAIN_CONFIG['workers'],
            "amp": TRAIN_CONFIG['amp'],
            "cache": TRAIN_CONFIG['cache'],
            "resume": resume # Pass resume flag
        }
        
        # Add advanced params from config if they exist
        optional_params = ['patience', 'cos_lr', 'overlap_mask', 'mosaic', 'rect', 'multi_scale']
        for p in optional_params:
            if p in TRAIN_CONFIG:
                train_args[p] = TRAIN_CONFIG[p]

        results = model.train(**train_args)
        
        print("\nTraining completed.")
        
        # Validation
        print("\nRunning validation...")
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on UAV dataset")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=str, help="Batch size (int or -1 for autobatch)")
    parser.add_argument("--device", type=str, help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--model", type=str, help="Model path or size (e.g. yolov8s.pt)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    # Convert batch to int if it's a number string, handle '-1'
    batch_val = args.batch
    if batch_val is not None:
        try:
            batch_val = int(batch_val)
        except ValueError:
            pass # Keep as string if it's something like 'auto' or '-1'

    train(epochs=args.epochs, batch=batch_val, device=args.device, model_path=args.model, resume=args.resume)
