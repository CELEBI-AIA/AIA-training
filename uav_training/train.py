import argparse
import sys
from pathlib import Path
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
import threading

# Enable V8 cuDNN API for better A100 performance
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
# Reduce VRAM fragmentation — must be set before any CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] >= 8:
        print(f"ℹ️ Ampere+ GPU (sm_{cc[0]}{cc[1]}) — native AMP will auto-select BF16", flush=True)
    else:
        print(f"ℹ️ GPU sm_{cc[0]}{cc[1]} — AMP will use FP16", flush=True)


# Version — keep in sync with uav_training/__init__.py
__version__ = "0.8.13"

print(f"\n🛰️  UAV Training Pipeline v{__version__}", flush=True)


def kill_gpu_hogs():
    """Clear GPU memory aggressively — sync, collect, then release cached blocks."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _is_checkpoint_valid(ckpt_path: Path) -> bool:
    """Check if a PyTorch checkpoint is readable and not corrupt.

    Tries weights_only=True first (avoids unpickling custom classes),
    but YOLO checkpoints contain DetectionModel so that will be rejected.
    Falls back to full load with explicit gc.collect() to free RAM.
    """
    import torch
    import gc
    if not ckpt_path.exists() or ckpt_path.stat().st_size < 1024 * 1024:  # < 1MB is suspicious for YOLO
        return False
    try:
        data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        del data
        gc.collect()
        return True
    except Exception as e:
        print(f"⚠️ Checkpoint {ckpt_path.name} is corrupt: {e}", flush=True)
        return False


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
        # Format: 2026-02-21_yolo11m_mAP50-0.853_mAP50-95-0.620
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


DRIVE_CP = Path(os.environ.get("UAV_PROJECT_DIR", "/content/drive/MyDrive/AIA/checkpoints"))
_SYNC_LOCK = threading.Lock()
_SYNC_IN_FLIGHT = False

def _sync_to_drive(save_dir, run_name):
    """Non-blocking Drive sync — checkpoint kaybını önler.

    Copies to a temp file first, then renames to avoid partially-written
    checkpoints on the Drive side.
    """
    try:
        target_dir = DRIVE_CP / run_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in ["weights/best.pt", "weights/last.pt"]:
            src = Path(save_dir) / f
            if src.exists():
                dst = target_dir / Path(f).name
                tmp = dst.with_suffix('.pt.tmp')
                shutil.copy2(src, tmp)
                tmp.replace(dst)
    except Exception as e:
        print(f"[DRIVE WARN] {e}", flush=True)

_LAST_SYNC_EPOCH = -1

def checkpoint_guard(trainer):
    """Drive sync every save_period epochs — GPU pipeline'ı bloklamaz."""
    global _SYNC_IN_FLIGHT, _LAST_SYNC_EPOCH
    save_period = int(TRAIN_CONFIG.get("save_period", 1))
    epoch = int(trainer.epoch) + 1  # Ultralytics epoch is 0-based.
    if epoch % save_period != 0:
        return
    with _SYNC_LOCK:
        if _SYNC_IN_FLIGHT or epoch == _LAST_SYNC_EPOCH:
            return
        _SYNC_IN_FLIGHT = True
        _LAST_SYNC_EPOCH = epoch

    def _sync_job():
        global _SYNC_IN_FLIGHT
        try:
            _sync_to_drive(trainer.save_dir, trainer.save_dir.name)
        finally:
            with _SYNC_LOCK:
                _SYNC_IN_FLIGHT = False

    threading.Thread(
        target=_sync_job,
        daemon=True
    ).start()

def _sync_results_to_drive(results_dir: Path, run_name: str):
    """Final sync from local SSD runs to Drive."""
    _sync_to_drive(results_dir, run_name)
    print(f"\n☁️  Final sync to Drive completed", flush=True)


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "cudnn" in msg)


def _log_precision_policy() -> None:
    """Emit a single precision policy line for run-to-run verification."""
    compile_mode = TRAIN_CONFIG.get("compile", False)
    gpu_capability = "cpu"
    bf16_hw = False
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        gpu_capability = f"sm_{cc[0]}{cc[1]}"
        bf16_hw = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    print(
        "[PRECISION] "
        f"gpu_capability={gpu_capability} "
        f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"tf32_cudnn={torch.backends.cudnn.allow_tf32} "
        f"bf16_hw={bf16_hw} amp=native "
        f"compile={compile_mode}",
        flush=True,
    )


def _train_single_phase(model_path, *, run_name, epochs, batch, device, imgsz=None, resume=False, phase_overrides=None):
    """Run one YOLO training phase and return summary metrics."""
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        print(f"Error: Dataset config not found at {yaml_path}", flush=True)
        print("Please run build_dataset.py first.")
        sys.exit(1)

    train_args = {
        "data": str(yaml_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz if imgsz is not None else TRAIN_CONFIG["imgsz"],
        "device": device,
        "project": str(TRAIN_CONFIG["project"]),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
        "workers": TRAIN_CONFIG["workers"],
        "amp": TRAIN_CONFIG["amp"],
        "cache": TRAIN_CONFIG["cache"],
        "resume": resume,
    }

    optional_params = [
        "optimizer", "momentum", "nbs",
        "patience", "cos_lr", "overlap_mask", "mosaic", "rect", "multi_scale",
        "close_mosaic", "deterministic", "save_period", "compile",
        "lr0", "lrf", "warmup_epochs", "weight_decay",
        "scale", "copy_paste", "copy_paste_mode", "flipud", "bgr",
        "box", "cls", "dfl",
    ]
    for p in optional_params:
        if p in TRAIN_CONFIG:
            train_args[p] = TRAIN_CONFIG[p]

    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"[AMP] BF16 hardware support: {bf16_ok}", flush=True)

    if phase_overrides:
        train_args.update(phase_overrides)

    attempt_args = train_args.copy()
    max_attempts = 4
    results = None
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        print(f"\n🔁 Training attempt {attempt}/{max_attempts}", flush=True)
        print_training_config(attempt_args)

        # #region agent log
        import json as _json, pathlib as _pl
        _logf = _pl.Path("debug-4e729f.log")
        _vram_data = {}
        if torch.cuda.is_available():
            _vram_data = {"allocated_mb": round(torch.cuda.memory_allocated(0)/1e6,1), "reserved_mb": round(torch.cuda.memory_reserved(0)/1e6,1), "max_allocated_mb": round(torch.cuda.max_memory_allocated(0)/1e6,1)}
        with open(_logf, "a") as _lf:
            _lf.write(_json.dumps({"sessionId":"4e729f","hypothesisId":"H-D","location":"train.py:pre_train_attempt","message":"VRAM before YOLO.train","data":{"attempt":attempt,"batch":attempt_args.get("batch"),"imgsz":attempt_args.get("imgsz"),"compile":attempt_args.get("compile"),"vram":_vram_data},"timestamp":int(time.time()*1000)}) + "\n")
        # #endregion

        model = YOLO(model_path)
        model.add_callback("on_fit_epoch_end", checkpoint_guard)
        try:
            results = model.train(**attempt_args)
            break
        except RuntimeError as exc:
            last_exc = exc

            # #region agent log
            _oom_vram = {}
            if torch.cuda.is_available():
                _oom_vram = {"allocated_mb": round(torch.cuda.memory_allocated(0)/1e6,1), "reserved_mb": round(torch.cuda.memory_reserved(0)/1e6,1), "max_allocated_mb": round(torch.cuda.max_memory_allocated(0)/1e6,1)}
            with open(_logf, "a") as _lf:
                _lf.write(_json.dumps({"sessionId":"4e729f","hypothesisId":"H-D" if _is_cuda_oom_error(exc) else "H-B","location":"train.py:train_exception","message":"training exception","data":{"attempt":attempt,"is_oom":_is_cuda_oom_error(exc),"error":str(exc)[:200],"vram":_oom_vram},"timestamp":int(time.time()*1000)}) + "\n")
            # #endregion

            if not _is_cuda_oom_error(exc) or attempt == max_attempts:
                raise

            print(f"⚠️ CUDA OOM detected on attempt {attempt}: {exc}", flush=True)
            next_args = attempt_args.copy()
            recovery_action = None

            if bool(next_args.get("compile", False)):
                next_args["compile"] = False
                recovery_action = "compile=False"
            elif isinstance(next_args.get("batch"), int) and next_args["batch"] > 8:
                next_args["batch"] = max(8, next_args["batch"] // 2)
                next_args["nbs"] = next_args["batch"]
                recovery_action = f"batch={next_args['batch']},nbs={next_args['batch']}"
            elif int(next_args.get("imgsz", 0)) > 896:
                next_args["imgsz"] = 896
                recovery_action = "imgsz=896"
            elif isinstance(next_args.get("batch"), int) and next_args["batch"] > 4:
                next_args["batch"] = max(4, next_args["batch"] // 2)
                next_args["nbs"] = next_args["batch"]
                recovery_action = f"batch={next_args['batch']},nbs={next_args['batch']}"

            if recovery_action is None:
                raise

            print(f"🛟 Applying OOM fallback: {recovery_action}", flush=True)
            del model
            kill_gpu_hogs()
            attempt_args = next_args

    if results is None and last_exc is not None:
        raise last_exc

    print("\n✅ Training completed.", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  📊 FINAL RESULTS ({run_name})")
    print(f"{'='*60}")
    print(f"  mAP50     : {results.box.map50:.4f}")
    print(f"  mAP50-95  : {results.box.map:.4f}")
    print(f"{'='*60}\n", flush=True)

    results_dir = Path(str(TRAIN_CONFIG["project"])) / run_name
    drive_dest = os.environ.get("DRIVE_UPLOAD_DIR")
    renamed = rename_and_export_best(results_dir, drive_dest)
    _sync_results_to_drive(results_dir, run_name)

    return {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "results_dir": str(results_dir),
        "best_pt": str(renamed) if renamed else None,
    }


def setup_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed all RNGs and configure CUDA determinism / benchmark mode.

    deterministic=False (default) → A100 kernel autotuning ON, max throughput.
    deterministic=True            → full reproducibility, significant speed cost.
    """
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(
        f"[SEED] seed={seed} deterministic={deterministic} "
        f"benchmark={torch.backends.cudnn.benchmark} "
        f"TF32={torch.backends.cuda.matmul.allow_tf32} "
        f"matmul_precision=high",
        flush=True,
    )


def train(epochs=None, batch=None, device=None, model_path=None, resume=False, two_phase=False):
    det = bool(TRAIN_CONFIG.get("deterministic", False))
    setup_seed(42, deterministic=det)
    kill_gpu_hogs()

    # Ensure dataset.yaml exists for both fresh and resume runs.
    # Resume without dataset metadata can happen after Colab runtime resets.
    yaml_path = DATASET_DIR / "dataset.yaml"
    needs_build = False

    if not yaml_path.exists():
        if resume:
            print("⚠️  Resume requested but dataset.yaml is missing — rebuilding dataset...", flush=True)
        needs_build = True
    elif not resume and yaml_path.exists():
        # Check if existing dataset uses symlinks (old format → Drive FUSE → slow)
        train_imgs = DATASET_DIR / "train" / "images"
        if train_imgs.exists():
            sample = next(train_imgs.iterdir(), None)
            if sample and sample.is_symlink():
                print("⚠️  Dataset uses symlinks (old format) → rebuilding with copies...", flush=True)
                import shutil as _shutil
                _shutil.rmtree(DATASET_DIR, ignore_errors=True)
                needs_build = True
            else:
                from config import is_colab
                if is_colab():
                    print(f"⚡ Dataset ready ({yaml_path}) — skipping rebuild", flush=True)
        else:
            needs_build = True

    if needs_build:
        print("🔄 Building dataset (hard-copy to local SSD)...", flush=True)
        build_dataset()

    # Use config values if not provided
    epochs = epochs if epochs is not None else TRAIN_CONFIG["epochs"]
    batch = batch if batch is not None else TRAIN_CONFIG["batch"]
    device = device if device is not None else TRAIN_CONFIG["device"]
    model_path = model_path if model_path is not None else TRAIN_CONFIG["model"]

    if two_phase and resume:
        print("Error: --two-phase and --resume cannot be used together.", flush=True)
        sys.exit(1)

    # Resume logic
    if resume:
        latest_ckpt = None

        # #region agent log
        import json as _json, pathlib as _pl
        _logf = _pl.Path("debug-4e729f.log")
        _mem_before = {}
        if torch.cuda.is_available():
            _mem_before = {"allocated_mb": round(torch.cuda.memory_allocated(0)/1e6,1), "reserved_mb": round(torch.cuda.memory_reserved(0)/1e6,1)}
        with open(_logf, "a") as _lf:
            _lf.write(_json.dumps({"sessionId":"4e729f","hypothesisId":"H-A","location":"train.py:resume_start","message":"memory before checkpoint validation","data":{"vram":_mem_before},"timestamp":int(time.time()*1000)}) + "\n")
        # #endregion

        # 1) CLI --model takes priority (bootstrap passes Drive checkpoint here)
        if model_path and Path(str(model_path)).exists() and _is_checkpoint_valid(Path(str(model_path))):
            latest_ckpt = Path(str(model_path))
            print(f"[RESUME] source=cli checkpoint={latest_ckpt}", flush=True)

        # 2) Search local project directory
        if latest_ckpt is None:
            project_results = TRAIN_CONFIG["project"]
            if project_results.exists():
                checkpoints = list(project_results.rglob("last.pt"))
                if checkpoints:
                    candidate = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    if _is_checkpoint_valid(candidate):
                        latest_ckpt = candidate
                        print(f"[RESUME] source=local checkpoint={latest_ckpt}", flush=True)

        # 3) Fallback: search Drive runs directory
        if latest_ckpt is None:
            drive_runs = os.environ.get("UAV_PROJECT_DIR")
            if drive_runs:
                drive_path = Path(drive_runs)
                if drive_path.exists():
                    checkpoints = list(drive_path.rglob("last.pt"))
                    if checkpoints:
                        candidate = max(checkpoints, key=lambda p: p.stat().st_mtime)
                        if _is_checkpoint_valid(candidate):
                            latest_ckpt = candidate
                            print(f"[RESUME] source=drive checkpoint={latest_ckpt}", flush=True)

        # #region agent log
        _mem_after = {}
        if torch.cuda.is_available():
            _mem_after = {"allocated_mb": round(torch.cuda.memory_allocated(0)/1e6,1), "reserved_mb": round(torch.cuda.memory_reserved(0)/1e6,1)}
        with open(_logf, "a") as _lf:
            _lf.write(_json.dumps({"sessionId":"4e729f","hypothesisId":"H-A","location":"train.py:resume_end","message":"memory after checkpoint validation","data":{"vram":_mem_after,"found": latest_ckpt is not None},"timestamp":int(time.time()*1000)}) + "\n")
        # #endregion

        if latest_ckpt is None:
            print("❌ No valid 'last.pt' found in CLI path, local runs, or Drive. Aborting resume.", flush=True)
            sys.exit(1)

        model_path = latest_ckpt
    else:
        print("[RESUME] mode=fresh", flush=True)
        print(f"🚀 Starting training: {model_path} on UAV dataset", flush=True)

    print(f"  Epochs: {epochs}  |  Batch: {batch}  |  Device: {device}", flush=True)

    # GPU info before training
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            bf16_native = "Yes" if cc[0] >= 8 else "No"
            print(
                f"  GPU: {gpu} (sm_{cc[0]}{cc[1]})  |  VRAM: {vram_free:.1f}/{vram_total:.1f} GB free  |  BF16 native: {bf16_native}",
                flush=True,
            )
    except Exception:
        pass

    _log_precision_policy()

    try:
        if not two_phase:
            return _train_single_phase(
                model_path,
                run_name=TRAIN_CONFIG["name"],
                epochs=epochs,
                batch=batch,
                device=device,
                resume=resume,
            )

        phase1_epochs = int(TRAIN_CONFIG.get("phase1_epochs", 85))
        phase2_epochs = int(TRAIN_CONFIG.get("phase2_epochs", 15))
        if epochs is not None and int(epochs) > 1:
            # Keep phase-2 length stable, assign the remainder to phase-1.
            phase2_epochs = min(phase2_epochs, int(epochs) - 1)
            phase1_epochs = int(epochs) - phase2_epochs

        phase1_name = f"{TRAIN_CONFIG['name']}_phase1"
        phase2_name = f"{TRAIN_CONFIG['name']}_phase2"
        print(f"\n🚀 Two-phase training active (phase1={phase1_epochs}, phase2={phase2_epochs})", flush=True)

        phase1_result = _train_single_phase(
            model_path,
            run_name=phase1_name,
            epochs=phase1_epochs,
            batch=batch,
            device=device,
            resume=False,
        )

        phase2_model_path = phase1_result.get("best_pt")
        if not phase2_model_path or not os.path.exists(phase2_model_path):
            phase2_model_path = str(Path(phase1_result["results_dir"]) / "weights" / "best.pt")
            
            # Safeguard: if best.pt wasn't saved/renamed properly, fallback to last.pt
            if not os.path.exists(phase2_model_path) or not _is_checkpoint_valid(Path(phase2_model_path)):
                print("⚠️  phase1 best.pt not found or invalid! Falling back to last.pt", flush=True)
                phase2_model_path = str(Path(phase1_result["results_dir"]) / "weights" / "last.pt")
                if not os.path.exists(phase2_model_path) or not _is_checkpoint_valid(Path(phase2_model_path)):
                     raise FileNotFoundError(f"Failed to find any valid phase1 weights in {phase1_result['results_dir']}")

        phase2_batch = batch
        if isinstance(batch, int):
            phase2_batch = max(1, batch // 2)

        phase2_overrides = {
            "mosaic": TRAIN_CONFIG.get("phase2_mosaic", 0.2),
            "close_mosaic": TRAIN_CONFIG.get("phase2_close_mosaic", 10),
            "lr0": TRAIN_CONFIG.get("phase2_lr0", TRAIN_CONFIG.get("lr0", 0.0015)),
        }

        phase2_result = _train_single_phase(
            phase2_model_path,
            run_name=phase2_name,
            epochs=phase2_epochs,
            batch=phase2_batch,
            device=device,
            imgsz=int(TRAIN_CONFIG.get("phase2_imgsz", 896)),
            resume=False,
            phase_overrides=phase2_overrides,
        )

        return {"phase1": phase1_result, "phase2": phase2_result}

    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 on UAV dataset")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=str, help="Batch size (int or -1 for autobatch)")
    parser.add_argument("--device", type=str, help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--model", type=str, help="Model path or size (e.g. yolo11m.pt)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--two-phase", action="store_true", help="Run 50+15 two-phase training profile")

    args = parser.parse_args()

    # Convert batch to int if it's a number string, handle '-1'
    batch_val = args.batch
    if batch_val is not None:
        try:
            batch_val = int(batch_val)
        except ValueError:
            pass # Keep as string if it's something like 'auto' or '-1'

    train(
        epochs=args.epochs,
        batch=batch_val,
        device=args.device,
        model_path=args.model,
        resume=args.resume,
        two_phase=args.two_phase,
    )
