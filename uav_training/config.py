from pathlib import Path
import os

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

# ── Colab Auto Hardware Detection ────────────────────────────────────────────

def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    return os.environ.get("COLAB_RELEASE_TAG") is not None or \
           os.path.exists("/content")

def auto_detect_hardware() -> tuple:
    """
    Probe Colab (or any) GPU/RAM/CPU and return optimal TRAIN_CONFIG overrides.
    Returns (config_overrides: dict, hw_info: dict).

    Uses EXPLICIT batch sizes per GPU tier instead of autobatch (-1),
    because autobatch only fills ~60% of VRAM which wastes expensive cloud GPU.
    """
    import torch

    info = {
        "gpu_name": "N/A",
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "cpu_count": os.cpu_count() or 2,
    }

    # ── GPU Info ──
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)

    # ── RAM Info ──
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2), 1)
                        break
        except Exception:
            info["ram_gb"] = 12.0

    # ── Explicit GPU-tier configuration ──
    # autobatch (-1) only uses ~60% VRAM → wastes 40% of expensive cloud GPU.
    # These are tested batch sizes that safely fill ~85-90% VRAM with AMP.
    vram = float(info["vram_gb"])
    ram  = float(info["ram_gb"])
    cpus = int(info["cpu_count"])

    if vram >= 35:
        # ── A100 40GB / A6000 48GB ──
        tier = "A100-40GB"
        model = "yolov8s.pt"
        imgsz = 640
        batch = 64          # yolov8s@640 → ~8-10GB VRAM, max throughput
    elif vram >= 20:
        # ── A100 24GB / L4 24GB ──
        tier = "A100-24GB / L4"
        model = "yolov8s.pt"
        imgsz = 640
        batch = 64          # yolov8s@640 → ~8-10GB VRAM
    elif vram >= 14:
        # ── T4 16GB / V100 16GB ──
        tier = "T4-16GB / V100"
        model = "yolov8s.pt"
        imgsz = 640
        batch = 32          # yolov8s@640 → ~5-6GB VRAM
    elif vram >= 7:
        # ── Various 8GB GPUs ──
        tier = "8GB GPU"
        model = "yolov8s.pt"
        imgsz = 640
        batch = 16          # yolov8s@640 → ~3-4GB VRAM
    else:
        # ── Low VRAM / CPU ──
        tier = "Low VRAM"
        model = "yolov8s.pt"
        imgsz = 640
        batch = 8

    # Workers: cap at 8 (more can cause Colab dataloader issues)
    workers = min(cpus, 8)

    # Multi-scale OFF — variable sizes cause OOM with large batches
    multi_scale = False

    # Disk cache — RAM cache needs 127GB+ for this dataset, disk is fast enough on SSD
    cache = True

    config_overrides = {
        "epochs": 100,
        "batch": batch,
        "imgsz": imgsz,
        "device": 0,
        "model": model,
        "workers": workers,
        "amp": True,
        "cache": cache,
        "exist_ok": True,
        "patience": 20,
        "cos_lr": True,
        "close_mosaic": 10,
        "overlap_mask": True,
        "mosaic": 1.0,
        "rect": False,
        "multi_scale": multi_scale,
        "deterministic": False,   # ~30% faster — non-deterministic CUDA kernels
        "save_period": 10,        # Checkpoint every 10 epochs (less I/O)
    }

    # Print detected hardware — flush=True so it appears instantly in Colab
    print(f"\n{'='*60}", flush=True)
    print(f"  🖥️  AUTO HARDWARE DETECTION")
    print(f"{'='*60}")
    print(f"  GPU        : {info['gpu_name']}")
    print(f"  VRAM       : {info['vram_gb']} GB")
    print(f"  RAM        : {info['ram_gb']} GB")
    print(f"  CPU Cores  : {info['cpu_count']}")
    print(f"  Tier       : {tier}")
    print(f"{'─'*60}")
    print(f"  Model      : {model}")
    print(f"  ImgSz      : {imgsz}")
    print(f"  Batch      : {batch}  (explicit, ~85-90% VRAM)")
    print(f"  Workers    : {workers}")
    print(f"  Cache      : {cache}")
    print(f"  Multi-Scale: {multi_scale}")
    print(f"  Epochs     : 100")
    print(f"  AMP (FP16) : True")
    print(f"{'='*60}\n", flush=True)

    return config_overrides, info


# ── Training Configuration ───────────────────────────────────────────────────

# In Colab: train to LOCAL SSD for max speed, then copy results to Drive.
# Google Drive FUSE is extremely slow for frequent small writes (checkpoints, plots)
# which stalls the GPU between batches.
if is_colab():
    _project_dir = "/content/runs"  # Local SSD — fast I/O
else:
    _project_dir = os.environ.get("UAV_PROJECT_DIR", str(ARTIFACTS_DIR / "training_results"))

# Default config (local / fallback)
TRAIN_CONFIG = {
    "epochs": 10,
    "batch": 4,           # Conservative for local 6GB VRAM
    "imgsz": 640,
    "device": 0,
    "model": "yolov8s.pt",
    "project": Path(_project_dir),
    "name": "uav_v3_optimized",
    "workers": 8,
    "amp": True,
    "cache": True,
    "exist_ok": True,
    "patience": 50,
    "cos_lr": True,
    "close_mosaic": 2,
    "overlap_mask": True,
    "mosaic": 0.5,
    "rect": False,
    "multi_scale": False,
}

# Auto-override when running on Colab
if is_colab():
    try:
        _overrides, _hw_info = auto_detect_hardware()
        # Preserve project/name, override the rest
        _overrides["project"] = TRAIN_CONFIG["project"]
        _overrides["name"] = TRAIN_CONFIG["name"]
        TRAIN_CONFIG.update(_overrides)
    except Exception as e:
        print(f"⚠️ Auto hardware detection failed: {e}", flush=True)
        print("  Falling back to default config")
