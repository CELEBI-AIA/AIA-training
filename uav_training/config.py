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

def auto_detect_hardware() -> dict:
    """
    Probe Colab (or any) GPU/RAM/CPU and return optimal TRAIN_CONFIG overrides.
    Designed to squeeze every drop of performance from cloud hardware.
    """
    import torch

    info = {
        "gpu_name": "N/A",
        "vram_gb": 0,
        "ram_gb": 0,
        "cpu_count": os.cpu_count() or 2,
    }

    # ── GPU Info ──
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_mem / (1024 ** 3), 1)

    # ── RAM Info ──
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        # fallback: read /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2), 1)
                        break
        except Exception:
            info["ram_gb"] = 12  # safe default

    # ── Determine optimal config based on hardware ──
    vram = info["vram_gb"]
    ram = info["ram_gb"]
    cpus = info["cpu_count"]

    # Model selection based on VRAM
    if vram >= 24:
        model = "yolov8l.pt"    # Large model for A100/L4 etc
    elif vram >= 16:
        model = "yolov8m.pt"    # Medium model for T4 16GB / V100
    else:
        model = "yolov8s.pt"    # Small model for lower VRAM

    # Image size
    imgsz = 1280 if vram >= 24 else 640

    # Batch: let YOLO autobatch find the max
    batch = -1  # autobatch

    # Workers: use all available CPUs
    workers = cpus

    # Multi-scale: only if enough VRAM headroom
    multi_scale = vram >= 16

    # Cache strategy: RAM cache since cloud has plenty
    cache = "ram" if ram >= 12 else True

    # More epochs for better convergence in cloud
    epochs = 100

    # Patience scales with epochs
    patience = 20

    config_overrides = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": 0,
        "model": model,
        "workers": workers,
        "amp": True,
        "cache": cache,
        "exist_ok": True,
        "patience": patience,
        "cos_lr": True,
        "close_mosaic": 10,
        "overlap_mask": True,
        "mosaic": 1.0,         # Full mosaic for cloud training
        "rect": False,
        "multi_scale": multi_scale,
    }

    # Print detected hardware
    print(f"\n{'='*60}")
    print(f"  🖥️  AUTO HARDWARE DETECTION")
    print(f"{'='*60}")
    print(f"  GPU        : {info['gpu_name']}")
    print(f"  VRAM       : {info['vram_gb']} GB")
    print(f"  RAM        : {info['ram_gb']} GB")
    print(f"  CPU Cores  : {info['cpu_count']}")
    print(f"{'─'*60}")
    print(f"  Model      : {model}")
    print(f"  ImgSz      : {imgsz}")
    print(f"  Batch      : autobatch (max GPU utilization)")
    print(f"  Workers    : {workers}")
    print(f"  Cache      : {cache}")
    print(f"  Multi-Scale: {multi_scale}")
    print(f"  Epochs     : {epochs}")
    print(f"{'='*60}\n")

    return config_overrides, info


# ── Training Configuration ───────────────────────────────────────────────────

# Training project dir can be overridden via env var (e.g. Colab -> Drive)
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
        print(f"⚠️ Auto hardware detection failed: {e}")
        print("  Falling back to default config")
