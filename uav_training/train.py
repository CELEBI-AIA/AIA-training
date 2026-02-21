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
import os
import time
import csv
import shutil

# Version — keep in sync with uav_training/__init__.py
__version__ = "0.6.0"

print(f"\n🛰️  UAV Training Pipeline v{__version__}", flush=True)


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
        print(f"⚠️ results.csv not found at {results_csv}", flush=True)
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
        print(f"⚠️ Error parsing results.csv: {e}", flush=True)

    return {"mAP50": best_map50, "mAP50-95": best_map50_95}


def rename_and_export_best(results_dir: Path, drive_dest: str | None = None) -> Path | None:
    """
    Rename best.pt with mAP scores appended to filename.
    Export to Drive/models/<unique_folder>/ with all analysis files.
    Returns the renamed path, or None on failure.
    """
    from datetime import datetime

    best_pt = results_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"⚠️ best.pt not found at {best_pt}", flush=True)
        return None

    metrics = get_best_metrics(results_dir)
    map50 = metrics["mAP50"]
    map50_95 = metrics["mAP50-95"]

    # Format scores: 0.8534 -> "0.853"
    new_name = f"best_mAP50-{map50:.3f}_mAP50-95-{map50_95:.3f}.pt"
    renamed_path = best_pt.parent / new_name

    # Rename locally
    shutil.copy2(best_pt, renamed_path)
    print(f"\n✅ Renamed: {best_pt.name} → {new_name}", flush=True)

    # All analysis files to export
    EXPORT_FILES = [
        # Results
        "results.csv", "results.png",
        # Confusion matrices
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        # Curves
        "PR_curve.png", "F1_curve.png", "R_curve.png", "P_curve.png",
        # Labels
        "labels.jpg", "labels_correlogram.jpg",
        # Training args
        "args.yaml",
        # Batch visualizations
        "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg",
        "val_batch0_labels.jpg", "val_batch0_pred.jpg",
        "val_batch1_labels.jpg", "val_batch1_pred.jpg",
        "val_batch2_labels.jpg", "val_batch2_pred.jpg",
    ]

    if drive_dest:
        # ── Create unique model folder under Drive/models/ ──
        # Format: 2026-02-21_yolov8l_mAP50-0.853_mAP50-95-0.620
        model_name = TRAIN_CONFIG.get("model", "yolo").replace(".pt", "")
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = f"{date_str}_{model_name}_mAP50-{map50:.3f}_mAP50-95-{map50_95:.3f}"

        models_dir = os.path.join(drive_dest, "models", folder_name)
        os.makedirs(models_dir, exist_ok=True)

        print(f"\n☁️  Exporting to: {models_dir}", flush=True)

        # Copy renamed best.pt
        shutil.copy2(renamed_path, os.path.join(models_dir, new_name))
        print(f"  ✓ {new_name}", flush=True)

        # Copy last.pt too
        last_pt = results_dir / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, os.path.join(models_dir, "last.pt"))
            print(f"  ✓ last.pt", flush=True)

        # Copy all analysis files
        copied = 0
        for fname in EXPORT_FILES:
            src = results_dir / fname
            if src.exists():
                shutil.copy2(src, os.path.join(models_dir, fname))
                copied += 1
                print(f"  ✓ {fname}", flush=True)

        print(f"\n  📊 Exported {copied + 2} files to {models_dir}", flush=True)

        # Also copy best.pt to flat DRIVE_UPLOAD for quick access
        os.makedirs(drive_dest, exist_ok=True)
        shutil.copy2(renamed_path, os.path.join(drive_dest, new_name))
        print(f"  ☁️  Quick access copy: {os.path.join(drive_dest, new_name)}", flush=True)

    return renamed_path


def print_training_config(train_args: dict):
    """Print the full training configuration so it's visible in logs."""
    print(f"\n{'─'*60}", flush=True)
    print(f"  📋 TRAINING CONFIGURATION")
    print(f"{'─'*60}")
    for k, v in train_args.items():
        print(f"  {k:<20}: {v}")
    print(f"{'─'*60}\n", flush=True)


def train(epochs=None, batch=None, device=None, model_path=None, resume=False):
    kill_gpu_hogs()

    # Auto-optimize dataset if not resuming AND not already built
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not resume and not yaml_path.exists():
        print("🔄 Optimizing Dataset (Smart Downsampling)...", flush=True)
        build_dataset()
    elif yaml_path.exists():
        from config import is_colab
        if is_colab():
            print(f"⚡ Dataset already built ({yaml_path}) — skipping rebuild", flush=True)

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
             print(f"Error: Project directory {project_results} not found.", flush=True)
             sys.exit(1)

        # Find all last.pt files in the project directory
        checkpoints = list(project_results.rglob("last.pt"))
        if not checkpoints:
            print(f"Error: No 'last.pt' found in {project_results}", flush=True)
            sys.exit(1)

        # Sort by modification time (newest first)
        latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Found latest checkpoint: {latest_ckpt}", flush=True)

        model_path = latest_ckpt
    else:
        print(f"🚀 Starting training: {model_path} on UAV dataset", flush=True)

    print(f"  Epochs: {epochs}  |  Batch: {batch}  |  Device: {device}", flush=True)

    # GPU info before training
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            print(f"  GPU: {gpu}  |  VRAM: {vram_free:.1f}/{vram_total:.1f} GB free", flush=True)
    except Exception:
        pass

    # Load a model
    model = YOLO(model_path)

    # Path to dataset config
    yaml_path = DATASET_DIR / "dataset.yaml"

    if not yaml_path.exists():
        print(f"Error: Dataset config not found at {yaml_path}", flush=True)
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
                           'rect', 'multi_scale', 'close_mosaic',
                           'deterministic', 'save_period']
        for p in optional_params:
            if p in TRAIN_CONFIG:
                train_args[p] = TRAIN_CONFIG[p]

        # Print full config so it appears in logs
        print_training_config(train_args)

        results = model.train(**train_args)

        print("\n✅ Training completed.", flush=True)

        # Print GPU usage after training
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.max_memory_allocated(0) / (1024**3)
                print(f"  Peak GPU VRAM used: {vram_used:.1f} GB", flush=True)
        except Exception:
            pass

        # Validation
        print("\n🔍 Running validation...", flush=True)
        metrics = model.val()
        print(f"\n{'='*60}", flush=True)
        print(f"  📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"  mAP50     : {metrics.box.map50:.4f}")
        print(f"  mAP50-95  : {metrics.box.map:.4f}")
        print(f"{'='*60}\n", flush=True)

        # Determine results directory
        results_dir = Path(str(TRAIN_CONFIG['project'])) / TRAIN_CONFIG['name']

        # Post-training: rename best.pt with scores and upload to Drive
        drive_dest = os.environ.get("DRIVE_UPLOAD_DIR")
        renamed = rename_and_export_best(results_dir, drive_dest)

        # Copy full results from local SSD to Drive for persistence
        drive_runs = os.environ.get("UAV_PROJECT_DIR")
        if drive_runs and str(results_dir).startswith("/content/runs"):
            drive_results = os.path.join(drive_runs, TRAIN_CONFIG['name'])
            os.makedirs(drive_results, exist_ok=True)
            print(f"\n☁️  Syncing results to Drive: {drive_results}", flush=True)
            _sync_cmd = f'rsync -a --info=progress2 "{results_dir}/" "{drive_results}/"'
            import subprocess
            subprocess.run(_sync_cmd, shell=True, check=False)
            print(f"  ✓ Results synced to Drive", flush=True)

        return {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "results_dir": str(results_dir),
            "best_pt": str(renamed) if renamed else None,
        }

    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
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
