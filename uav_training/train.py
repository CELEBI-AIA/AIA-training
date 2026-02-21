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
import csv
import shutil


def kill_gpu_hogs():
    """Clear GPU memory before training."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_best_metrics(results_dir: Path) -> dict:
    """
    Parse results.csv from a YOLO training run and extract the best mAP scores.
    Returns dict with 'mAP50' and 'mAP50-95' as floats.
    """
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        print(f"⚠️ results.csv not found at {results_csv}")
        return {"mAP50": 0.0, "mAP50-95": 0.0}

    best_map50 = 0.0
    best_map50_95 = 0.0

    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Column names have leading spaces in Ultralytics output
                # Try to find mAP columns dynamically
                for key, val in row.items():
                    k = key.strip().lower()
                    try:
                        v = float(val.strip())
                    except (ValueError, AttributeError):
                        continue

                    if "map50-95" in k or "map50_95" in k or k.endswith("map"):
                        best_map50_95 = max(best_map50_95, v)
                    elif "map50" in k:
                        best_map50 = max(best_map50, v)
    except Exception as e:
        print(f"⚠️ Error parsing results.csv: {e}")

    return {"mAP50": best_map50, "mAP50-95": best_map50_95}


def rename_and_export_best(results_dir: Path, drive_dest: str | None = None) -> Path | None:
    """
    Rename best.pt with mAP scores appended to filename.
    Optionally copy to Google Drive.
    Returns the renamed path, or None on failure.
    """
    best_pt = results_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"⚠️ best.pt not found at {best_pt}")
        return None

    metrics = get_best_metrics(results_dir)
    map50 = metrics["mAP50"]
    map50_95 = metrics["mAP50-95"]

    # Format scores: 0.8534 -> "0.853"
    new_name = f"best_mAP50-{map50:.3f}_mAP50-95-{map50_95:.3f}.pt"
    renamed_path = best_pt.parent / new_name

    # Rename locally
    shutil.copy2(best_pt, renamed_path)
    print(f"\n✅ Renamed: {best_pt.name} → {new_name}")

    # Upload to Drive if destination specified
    if drive_dest:
        os.makedirs(drive_dest, exist_ok=True)
        drive_path = Path(drive_dest) / new_name
        shutil.copy2(renamed_path, drive_path)
        print(f"☁️  Uploaded to Drive: {drive_path}")

        # Also copy results.csv and results.png
        for extra in ["results.csv", "results.png", "confusion_matrix.png",
                       "confusion_matrix_normalized.png", "PR_curve.png",
                       "F1_curve.png"]:
            src = results_dir / extra
            if src.exists():
                shutil.copy2(src, Path(drive_dest) / extra)
                print(f"☁️  Uploaded: {extra}")

    return renamed_path


def train(epochs=None, batch=None, device=None, model_path=None, resume=False):
    kill_gpu_hogs()

    # Auto-optimize dataset if not resuming
    if not resume:
        print("🔄 Optimizing Dataset (Smart Downsampling)...", flush=True)
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
        print(f"Starting training for {model_path} on UAV dataset...", flush=True)

    print(f"Epochs: {epochs}, Batch: {batch}, Device: {device}", flush=True)

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
            "resume": resume
        }

        # Add advanced params from config if they exist
        optional_params = ['patience', 'cos_lr', 'overlap_mask', 'mosaic',
                           'rect', 'multi_scale', 'close_mosaic']
        for p in optional_params:
            if p in TRAIN_CONFIG:
                train_args[p] = TRAIN_CONFIG[p]

        results = model.train(**train_args)

        print("\nTraining completed.", flush=True)

        # Validation
        print("\nRunning validation...", flush=True)
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50}", flush=True)
        print(f"mAP50-95: {metrics.box.map}", flush=True)

        # Determine results directory
        results_dir = Path(str(TRAIN_CONFIG['project'])) / TRAIN_CONFIG['name']

        # Post-training: rename best.pt with scores and upload to Drive
        drive_dest = os.environ.get("DRIVE_UPLOAD_DIR")
        renamed = rename_and_export_best(results_dir, drive_dest)

        return {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "results_dir": str(results_dir),
            "best_pt": str(renamed) if renamed else None, 
        }

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
