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


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    return os.environ.get("COLAB_RELEASE_TAG") is not None or \
           os.path.exists("/content")


# On Colab: dataset lives entirely on LOCAL SSD for max I/O speed.
# build_dataset.py copies images here (no symlinks!), YOLO reads from here.
# On local dev: dataset stays in artifacts/ as usual.
if is_colab():
    DATASET_DIR = Path("/content/dataset_built")
else:
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

# is_colab() is defined at top of file (needed early for DATASET_DIR)

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

    # ── TPU Detection ──
    # YOLO/Ultralytics requires CUDA. TPUs (v6e-1, v5e-1) use XLA which
    # is NOT compatible with YOLO's training loop. Detect and warn.
    is_tpu = os.environ.get("TPU_NAME") is not None or \
             os.environ.get("COLAB_TPU_ADDR") is not None or \
             os.path.exists("/dev/accel0")
    if is_tpu and vram == 0:
        print("\n" + "!" * 60, flush=True)
        print("  ⚠️  TPU RUNTIME DETECTED — YOLO EĞİTİMİ İÇİN UYGUN DEĞİL!")
        print("  YOLO/Ultralytics CUDA (GPU) gerektirir, TPU/XLA desteklemez.")
        print("  Lütfen runtime'ı GPU'ya değiştirin:")
        print("     Runtime → Change runtime type → GPU (T4/L4/A100/H100)")
        print("!" * 60 + "\n", flush=True)
        # Fall back to CPU-like settings
        tier = "TPU (unsupported)"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 8
        info["gpu_name"] = f"TPU ({os.environ.get('TPU_NAME', 'unknown')})"

    elif vram >= 70:
        # ── H100 80GB ──
        tier = "H100-80GB"
        model = "yolo11m.pt"
        imgsz = 1024        # High-res for small objects
        batch = 64          # Reduced from 96 due to 1024px
    elif vram >= 35:
        # ── A100 40GB / A6000 48GB ──
        tier = "A100-40GB"
        model = "yolo11m.pt"
        imgsz = 1024        # High-res for small objects
        batch = 32          # Reduced from 48 due to 1024px
    elif vram >= 20:
        # ── L4 24GB ──
        tier = "L4-24GB"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 32           # yolo11m@640 → ~12-15GB VRAM
    elif vram >= 14:
        # ── T4 16GB / V100 16GB ──
        tier = "T4-16GB"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 16           # yolo11m@640 → ~10-12GB VRAM
    elif vram >= 7:
        # ── Various 8GB GPUs ──
        tier = "8GB GPU"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 8
    else:
        # ── Low VRAM / CPU ──
        tier = "Low VRAM / CPU"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 4

    # ── CPU Worker & Thread Configuration ──
    # DO NOT restrict DataLoader workers with OMP. Just limit torch inter-ops.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "1"

    workers = min(cpus, 8)

    # ── Safe VRAM Batch Sizing (Target ~80% Capacity) ──
    usable_vram = vram * 0.80
    model_overhead_gb = 3.5
    available = max(0.0, usable_vram - model_overhead_gb)
    gb_per_sample = 0.72 if imgsz == 1024 else ((imgsz / 640) ** 2 * 0.28)
    safe_batch = int(available / gb_per_sample)
    batch = max(8, (safe_batch // 8) * 8)

    # Multi-scale OFF — variable sizes cause OOM with large batches
    multi_scale = False

    # Cache OFF — read directly from NVMe SSD
    cache = False

    # ── Linear LR Scaling ──
    # Scaled properly to match new dynamic batch sizing
    base_batch = 16
    base_lr = 0.01
    scaled_lr0 = base_lr * (batch / base_batch)
    warmup_epochs = 5.0 # Increased for larger batch sizes

    config_overrides = {
        "epochs": 65,            # 50 + 15 two-phase profile
        "phase1_epochs": 50,
        "phase2_epochs": 15,
        "phase2_imgsz": 896,
        "phase2_mosaic": 0.2,
        "phase2_close_mosaic": 5,
        "phase2_lr0": scaled_lr0 * 0.1,  # Reduced LR in phase 2
        "batch": batch,
        "imgsz": imgsz,
        "device": 0,
        "model": model,
        "workers": workers,
        "amp": True,
        "amp_dtype": "bf16",
        "cache": cache,
        "exist_ok": True,
        "patience": 30,           # Extended early stopping tolerance
        "cos_lr": True,
        "close_mosaic": 5,       # Keep clean samples for the final epochs
        "overlap_mask": True,
        
        # ── Augmentation (UAV & Small Object Optimized) ──
        "mosaic": 1.0,            
        "scale": 0.4,             # Increased for better scale variation for small objects
        "copy_paste": 0.3,        # Increased context simulation
        "copy_paste_mode": "flip",
        "flipud": 0.5,            
        "fliplr": 0.5,            # Added symmetric flips
        "hsv_h": 0.015,           # Adding slight color variation
        "hsv_s": 0.7,             # Sunset/fog simulation
        "hsv_v": 0.4,             # Day/night transitions
        "bgr": 0.05,              
        
        # ── Optimizer & Hyperparameters ──
        "lr0": scaled_lr0,        # Linearly scaled 
        "lrf": 0.01,              # Final LR fraction
        "warmup_epochs": warmup_epochs, 
        "weight_decay": 0.0005,
        "label_smoothing": 0.05,  
        "box": 7.5,
        "cls": 0.7,
        "dfl": 1.5,
        
        # ── Mechanics ──
        "rect": False,
        "multi_scale": multi_scale,
        "deterministic": False,   # ~30% faster — non-deterministic CUDA kernels
        "compile": True,          # torch.compile via YOLO's own pipeline (20-40% faster)
        "save_period": 5,         # Reduced from 10 to 5 to prevent data loss on Colab interrupts
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
    print(f"  Epochs     : 65 (phase1=50, phase2=15)")
    print(f"  AMP (BF16 tgt) : True")
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
    "epochs": 65,
    "phase1_epochs": 50,
    "phase2_epochs": 15,
    "phase2_imgsz": 896,
    "phase2_mosaic": 0.2,
    "phase2_close_mosaic": 10,
    "phase2_lr0": 0.0015,
    "batch": 4,           # Conservative for local 6GB VRAM
    "imgsz": 640,
    "device": 0,
    "model": "yolo11m.pt",
    "project": Path(_project_dir),
    "name": "uav_v3_optimized",
    "workers": 8,
    "amp": True,
    "cache": True,
    "exist_ok": True,
    "patience": 30,
    "cos_lr": True,
    "close_mosaic": 5,
    "overlap_mask": True,
    "mosaic": 1.0,
    "scale": 0.4,
    "copy_paste": 0.3,
    "copy_paste_mode": "flip",
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "bgr": 0.05,
    "lr0": 0.02,
    "lrf": 0.01,
    "warmup_epochs": 5.0,
    "weight_decay": 0.0005,
    "label_smoothing": 0.05,
    "box": 7.5,
    "cls": 0.7,
    "dfl": 1.5,
    "min_bbox_norm": 0.004,
    "include_test_in_val": False,
    "rect": False,
    "multi_scale": False,
    "deterministic": False,
    "compile": True,
    "save_period": 5,
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
