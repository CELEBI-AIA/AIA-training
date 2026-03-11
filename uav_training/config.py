from pathlib import Path
import os
import sys
# Base paths
# Getting the project root relative to this config file
# structure: project_root/uav_training/config.py -> project_root is parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

_KNOWN_TRAIN_DATA_DIRS = {"UAI_UAP", "drone-vision-project", "megaset"}


def _looks_like_train_data_dir(path: Path) -> bool:
    """Return True when path looks like TRAIN_DATA root (contains known source dirs)."""
    if not path.exists() or not path.is_dir():
        return False
    try:
        child_dirs = {p.name for p in path.iterdir() if p.is_dir()}
    except OSError:
        return False
    return bool(child_dirs & _KNOWN_TRAIN_DATA_DIRS)


def _find_nested_subdir(base: Path, subdir_name: str, max_depth: int = 6) -> Path | None:
    """Find nested subdir_name under base with bounded depth, preferring valid TRAIN_DATA roots."""
    if not base.exists() or not base.is_dir():
        return None

    base_resolved = base.resolve()
    for root, dirs, _ in os.walk(base_resolved):
        rel = os.path.relpath(root, base_resolved)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        if subdir_name in dirs:
            candidate = Path(root) / subdir_name
            if _looks_like_train_data_dir(candidate):
                return candidate
    return None


def _resolve_datasets_train_dir() -> Path:
    """Resolve source dataset root. Default: datasets/TRAIN_DATA."""
    preferred_name = (os.environ.get("UAV_DATASET_SUBDIR", "TRAIN_DATA") or "TRAIN_DATA").strip()
    preferred_path = DATASETS_ROOT / preferred_name
    if preferred_path.is_dir():
        return preferred_path

    # Direct-root layout fallback:
    # datasets/{UAI_UAP,drone-vision-project,megaset}
    if _looks_like_train_data_dir(DATASETS_ROOT):
        return DATASETS_ROOT

    train_data_path = DATASETS_ROOT / "TRAIN_DATA"
    if preferred_name != "TRAIN_DATA" and train_data_path.is_dir():
        return train_data_path

    nested_preferred = _find_nested_subdir(DATASETS_ROOT, preferred_name)
    if nested_preferred is not None:
        return nested_preferred

    if preferred_name != "TRAIN_DATA":
        nested_train_data = _find_nested_subdir(DATASETS_ROOT, "TRAIN_DATA")
        if nested_train_data is not None:
            return nested_train_data

    # Default path for fresh setups before extraction.
    return preferred_path

# Source directory for build_dataset (audit must scan the same path).
# Primary layout: datasets/TRAIN_DATA.
DATASETS_TRAIN_DIR = _resolve_datasets_train_dir()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "uav_model"

# Do not create directories at import time.
# Directories are created in ensure_colab_config() to keep imports side-effect free.

def _get_module_version() -> str:
    """Read __version__ from uav_training/__init__.py without importing the package."""
    vf = Path(__file__).resolve().parent / "__init__.py"
    if vf.exists():
        try:
            for line in vf.read_text(encoding="utf-8").splitlines():
                if "__version__" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip("'\"")
        except Exception:
            pass
    return "dev"

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

# Canonical class mapping - matches build_dataset.py / dataset.yaml output.
# 0=vehicle, 1=human, 2=uap, 3=uai
TARGET_CLASSES = {
    "vehicle": 0,
    "car": 0,
    "tasit": 0,
    "arac": 0,
    "araba": 0,
    "human": 1,
    "person": 1,
    "pedestrian": 1,
    "people": 1,
    "insan": 1,
    "uap": 2,
    "uai": 3,
}

# Supported image extensions used by build/audit/inference/visualization utilities.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif")

# Hardware auto-detection is intentionally lazy.
# train.py calls ensure_colab_config() before training to avoid import-time side effects.

def setup_torch_backend() -> None:
    """Central TF32 configuration. Call before any CUDA ops."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def auto_detect_hardware() -> tuple:
    """
    Probe Colab (or any) GPU/RAM/CPU and return optimal TRAIN_CONFIG overrides.
    Returns (config_overrides: dict, hw_info: dict).

    Uses EXPLICIT batch sizes per GPU tier instead of autobatch (-1),
    because autobatch only fills ~60% of VRAM which wastes expensive cloud GPU.
    """
    import torch
    setup_torch_backend()

    info = {
        "gpu_name": "N/A",
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "cpu_count": os.cpu_count() or 2,
    }

    # Debug snapshot: actual GPU name and VRAM seen by PyTorch.
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)

    # Debug snapshot: total host RAM (psutil first, /proc fallback).
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

    # Explicit GPU-tier configuration.
    # autobatch (-1) only uses ~60% VRAM -> wastes 40% of expensive cloud GPU.
    # These are tested batch sizes that safely fill ~85-90% VRAM with AMP.
    vram = float(info["vram_gb"])
    ram = float(info["ram_gb"])
    cpus = int(info["cpu_count"])

    # Split RAM into Normal/High profiles for cache/worker tuning.
    is_high_ram = ram > 120
    info["is_high_ram"] = is_high_ram

    # TPU detection and hard warning path.
    # YOLO/Ultralytics requires CUDA. TPUs (v6e-1, v5e-1) use XLA which
    # is NOT compatible with YOLO's training loop. Detect and warn.
    is_tpu = os.environ.get("TPU_NAME") is not None or \
        os.environ.get("COLAB_TPU_ADDR") is not None or \
        os.path.exists("/dev/accel0")
    if is_tpu and vram == 0:
        print("\n" + "!" * 60, flush=True)
        print("  ⚠️ WARN  TPU RUNTIME DETECTED - YOLO EGITIMI ICIN UYGUN DEGIL!")
        print("  YOLO/Ultralytics CUDA (GPU) gerektirir, TPU/XLA desteklemez.")
        print("  Lutfen runtime'i GPU'ya degistirin:")
        print("     Runtime -> Change runtime type -> GPU (T4/L4/A100/H100)")
        print("!" * 60 + "\n", flush=True)
        # Fall back to CPU-like settings
        tier = "TPU (unsupported)"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 8
        info["gpu_name"] = f"TPU ({os.environ.get('TPU_NAME', 'unknown')})"

    elif is_tpu:
        # TPU detected with vram > 0 (e.g. Colab H100+TPU hybrid) - same fallback
        print("\n" + "!" * 60, flush=True)
        print("  ⚠️ WARN  TPU RUNTIME DETECTED - YOLO EGITIMI ICIN UYGUN DEGIL!")
        print("  YOLO/Ultralytics CUDA (GPU) gerektirir, TPU/XLA desteklemez.")
        print("  Lutfen runtime'i GPU'ya degistirin:")
        print("     Runtime -> Change runtime type -> GPU (T4/L4/A100/H100)")
        print("!" * 60 + "\n", flush=True)
        tier = "TPU (unsupported)"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 8
        info["gpu_name"] = f"TPU ({os.environ.get('TPU_NAME', 'unknown')})"

    elif vram >= 70:
        # H100 80GB / A100 80GB
        gpu_name = info.get("gpu_name", "") or ""
        if "A100" in gpu_name:
            tier = "A100-80GB"
        else:
            tier = "H100-80GB"
        model = "yolo11m.pt"
        imgsz = 1024        # High-res for small objects
        batch = 32          # Reduced from 64 due to 1024px peaks (fix for P-04)
    elif vram >= 35:
        # A100 40GB / A6000 48GB
        tier = "A100-40GB"
        model = "yolo11m.pt"
        imgsz = 1024        # High-res for small objects
        batch = 24          # Reduced from 28 to avoid TaskAlignedAssigner OOM
    elif vram >= 20:
        # L4 24GB
        tier = "L4-24GB"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 32           # yolo11m@640 -> ~12-15GB VRAM
    elif vram >= 14:
        # T4 16GB / V100 16GB
        tier = "T4-16GB"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 16           # yolo11m@640 -> ~10-12GB VRAM
    elif vram >= 7:
        # Various 8GB GPUs
        tier = "8GB GPU"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 8
    else:
        # Low VRAM / CPU
        tier = "Low VRAM / CPU"
        model = "yolo11m.pt"
        imgsz = 640
        batch = 4

    # CPU worker/thread configuration.
    # Keep thread limits conservative but not overly restrictive.
    torch_threads = max(1, min(4, cpus // 2))
    interop_threads = 1 if cpus <= 4 else 2
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(interop_threads)
    os.environ["OMP_NUM_THREADS"] = str(max(2, min(6, cpus // 2)))
    os.environ["OPENBLAS_NUM_THREADS"] = str(max(2, min(6, cpus // 2)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, min(2, cpus // 4)))

    # Multi-scale OFF - variable sizes cause OOM with large batches
    # High RAM (167GB): more workers for faster data prefetch; Normal (~80GB): conservative
    workers = min(max(cpus * 2, 4), 12) if (vram >= 35 and is_high_ram) else \
             min(max(cpus * 2, 4), 10) if vram >= 35 else min(max(cpus * 2, 4), 8)

    # Multi-scale OFF - variable sizes cause OOM with large batches
    multi_scale = False

    # Dynamic cache: High RAM (167GB) -> ram; Normal A100 (~80GB) -> disk; low RAM -> off
    # Colab: Normal ~80GB, High RAM ~167GB
    if is_high_ram or ram > 100:
        cache = "ram"
    elif ram > 20:
        cache = "disk"
    else:
        cache = False

    # AdamW base LR config.
    # AdamW uses ~1/10 of SGD lr; nbs=batch disables Ultralytics internal scaling
    adamw_lr0 = 0.001
    warmup_epochs = 5.0

    # Dynamo (torch.compile) supports Python 3.12 on recent versions
    compile_mode = "reduce-overhead" if vram >= 35 else False

    config_overrides = {
        "epochs": 65,            # 50 + 15 two-phase profile
        "phase1_epochs": 50,
        "phase2_epochs": 15,
        "phase2_imgsz": imgsz,
        "phase2_batch": batch,
        "phase2_mosaic": 0.3,
        "phase2_close_mosaic": 0,
        "phase2_lr0": adamw_lr0 * 0.1,
        "phase2_lrf": 0.01,
        "phase2_copy_paste": 0.3,
        "phase2_degrees": 0.0,
        "phase2_scale": 0.6,
        "phase2_hsv_s": 0.9,
        "phase2_hsv_v": 0.6,
        "batch": batch,
        "imgsz": imgsz,
        "device": 0,
        "model": model,
        "workers": workers,
        "amp": True,
        "cache": cache,
        "exist_ok": True,
        "patience": 30,
        "cos_lr": True,
        "close_mosaic": 5,
        "overlap_mask": True,

        # Augmentation profile (UAV and small-object optimized, top-down bias).
        # For top-down aerial videos, slightly lower mosaic and flipud
        # reduce unrealistic composites and extreme vertical flips.
        "mosaic": 0.7,
        "scale": 0.4,
        "copy_paste": 0.2,
        "copy_paste_mode": "flip",
        "flipud": 0.15,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "bgr": 0.05,
        "degrees": 0.0,

        # Optimizer and hyperparameters.
        "optimizer": "AdamW",
        "lr0": adamw_lr0,
        "lrf": 0.01,
        "momentum": 0.9,         # beta1 for AdamW
        "nbs": batch,            # disable Ultralytics internal LR scaling
        "warmup_epochs": warmup_epochs,
        "weight_decay": 0.0005,
        "box": 7.5,
        "cls": 1.0,
        "dfl": 1.5,
        "min_bbox_norm": 0.003,

        # Execution mechanics.
        "rect": False,
        "multi_scale": multi_scale,
        "deterministic": False,
        "compile": compile_mode,
        "save_period": 1,
    }

    # Print detected hardware summary (used for debugging wrong runtime selection).
    ram_mode = "High RAM (167GB)" if is_high_ram else "Normal (~80GB)"
    print(f"\n{'='*60}", flush=True)
    print("    AUTO HARDWARE DETECTION")
    print(f"{'='*60}")
    print(f"  GPU        : {info['gpu_name']}")
    print(f"  VRAM       : {info['vram_gb']} GB")
    print(f"  RAM        : {info['ram_gb']} GB  ({ram_mode})")
    print(f"  CPU Cores  : {info['cpu_count']}")
    print(f"  Tier       : {tier}")
    print(f"{'-'*60}")
    print(f"  Model      : {model}")
    print(f"  ImgSz      : {imgsz}")
    print(f"  Batch      : {batch}  (explicit, ~85-90% VRAM)")
    print(f"  Workers    : {workers}")
    print(f"  Threads    : torch={torch_threads}, interop={interop_threads}")
    print(f"  Cache      : {cache}")
    print(f"  Multi-Scale: {multi_scale}")
    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print(f"  BF16      : {'Supported' if bf16_ok else 'Not supported'}")
    print("  TF32      : Enabled (matmul + cuDNN)")
    print("  Epochs     : 65 (phase1=50, phase2=15)")
    print("  AMP        : True (BF16 auto-selected on Ampere+)")
    print(f"{'='*60}\n", flush=True)

    return config_overrides, info

# Training configuration (local defaults; Colab overrides applied in ensure_colab_config()).

# In Colab: train to LOCAL SSD for max speed, then copy results to Drive.
# Google Drive FUSE is extremely slow for frequent small writes (checkpoints, plots)
# which stalls the GPU between batches.
if is_colab():
    _project_dir = "/content/runs"  # Local SSD - fast I/O
else:
    _project_dir = os.environ.get("UAV_PROJECT_DIR", str(ARTIFACTS_DIR / "training_results"))

# Default config (local / fallback)
TRAIN_CONFIG = {
    "epochs": 65,
    "phase1_epochs": 50,
    "phase2_epochs": 15,
    "phase2_imgsz": 640,
    "phase2_batch": 4,
    "phase2_mosaic": 0.3,
    "phase2_close_mosaic": 0,
    "phase2_lr0": 0.0001,
    "phase2_lrf": 0.01,
    "phase2_copy_paste": 0.3,
    "phase2_degrees": 0.0,
    "phase2_scale": 0.6,
    "phase2_hsv_s": 0.9,
    "phase2_hsv_v": 0.6,
    "batch": 4,           # Conservative for local 6GB VRAM
    "imgsz": 640,
    "device": 0,
    "model": "yolo11m.pt",
    "project": Path(_project_dir),
    "name": f"uav_v3_optimized_v{_get_module_version()}",
    "workers": 8,
    "amp": True,
    "cache": "disk",
    "exist_ok": True,
    "patience": 30,
    "cos_lr": True,
    "close_mosaic": 5,
    "overlap_mask": True,
    "mosaic": 0.7,
    "scale": 0.4,
    "copy_paste": 0.2,
    "copy_paste_mode": "flip",
    "flipud": 0.15,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "bgr": 0.05,
    "degrees": 0.0,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.9,
    "nbs": 4,                 # match batch to disable internal LR scaling
    "warmup_epochs": 5.0,
    "weight_decay": 0.0005,
    "box": 7.5,
    "cls": 1.0,
    "dfl": 1.5,
    "min_bbox_norm": 0.003,
    "include_test_in_val": False,
    "remove_orphans": True,           # Remove images without labels and labels without images
    "remove_train_val_duplicates": True,  # Remove from val any image with same content as train (prevents leakage)
    "rect": False,
    "multi_scale": False,
    "deterministic": False,
    "compile": False,
    "save_period": 1,
}

# Lazy Colab config - avoids import side effect (torch/CUDA/psutil on every import).
# Call ensure_colab_config() from train.py before training; other modules use default config.
_colab_config_initialized = False

def ensure_colab_config() -> None:
    """Run auto_detect_hardware and update TRAIN_CONFIG when on Colab. Idempotent."""
    # Create artifacts directory here to keep module import side-effect free.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    global _colab_config_initialized
    if not is_colab() or _colab_config_initialized:
        return
    try:
        _overrides, _hw_info = auto_detect_hardware()
        _overrides["project"] = TRAIN_CONFIG["project"]
        _overrides["name"] = TRAIN_CONFIG["name"]
        TRAIN_CONFIG.update(_overrides)
        _colab_config_initialized = True
    except Exception as e:
        print(f"⚠️ WARN Auto hardware detection failed: {e}", flush=True)
        print("  Falling back to default config")

