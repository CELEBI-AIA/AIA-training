from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "gps_model"

# Ensure artifacts dir exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

import os

# Training Config
TRAIN_CONFIG = {
    "epochs": 5, # Test run
    "batch_size": 16, # Adjusted for 6GB VRAM
    "img_size": (256, 256), 
    "device": "0", 
    "num_workers": 8, # Fixed 8 workers
    "learning_rate": 1e-4,
    "lambda_traj": 0.1, # Weight for trajectory loss
    "mixed_precision": True, # Enable AMP
    "pin_memory": True,
    "frame_cache_size": 256,
}
