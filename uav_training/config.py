from pathlib import Path

# Base paths
# Getting the project root relative to this config file
# structure: project_root/uav_training/config.py -> project_root is parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "uav_model"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = ARTIFACTS_DIR / "dataset_uap_uai"
AUDIT_REPORT = ARTIFACTS_DIR / "audit_report.json"

# Filter criteria for datasets
TARGET_CLASSES = {
    "uap": 0,
    "uai": 1,
    "human": 2,
    "person": 2,
    "insan": 2,
    "vehicle": 3,
    "car": 3,
    "tasit": 3,
    "arac": 3,
    "araba": 3
}

import os

# Training project dir can be overridden via env var (e.g. Colab -> Drive)
_project_dir = os.environ.get("UAV_PROJECT_DIR", str(ARTIFACTS_DIR / "training_results"))

# Training Params
TRAIN_CONFIG = {
    "epochs": 10, # Increased for better convergence as per user request
    "batch": 4, # Optimized for 6GB VRAM
    "imgsz": 640, # Optimized for speed
    "device": 0,
    "model": "yolov8s.pt", 
    "project": Path(_project_dir),
    "name": "uav_v3_optimized",
    "workers": 8, # Optimal for 12-thread CPU stability
    "amp": True, 
    "cache": True, # RAM Cache for faster training
    "exist_ok": True,
    "patience": 50, 
    "cos_lr": True, 
    "close_mosaic": 2, # Disable mosaic only for last 2 epochs (since total is 10)
    "overlap_mask": True,
    "mosaic": 0.5, # Re-enabling mild mosaic for better generalization with more epochs
    "rect": False, 
    "multi_scale": False, 
}
