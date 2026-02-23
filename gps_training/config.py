from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "gps_model"

# Ensure artifacts dir exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

import os

def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    return os.environ.get("COLAB_RELEASE_TAG") is not None or \
           os.path.exists("/content")

def auto_detect_hardware() -> tuple:
    import torch
    
    info = {
        "gpu_name": "N/A",
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "cpu_count": os.cpu_count() or 2,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        info["ram_gb"] = 12.0

    vram = float(info["vram_gb"])
    cpus = int(info["cpu_count"])

    if vram >= 70:
        tier = "H100-80GB"
        batch = 256
    elif vram >= 35:
        tier = "A100-40GB"
        batch = 128
    elif vram >= 20:
        tier = "L4-24GB"
        batch = 64
    elif vram >= 14:
        tier = "T4-16GB"
        batch = 32
    elif vram >= 7:
        tier = "8GB GPU"
        batch = 16
    else:
        tier = "Low VRAM / CPU"
        batch = 8

    workers = min(cpus, 8)

    config_overrides = {
        "batch_size": batch,
        "num_workers": workers,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"  🖥️  AUTO HARDWARE DETECTION")
    print(f"{'='*60}")
    print(f"  GPU        : {info['gpu_name']}")
    print(f"  VRAM       : {info['vram_gb']} GB")
    print(f"  RAM        : {info['ram_gb']} GB")
    print(f"  CPU Cores  : {info['cpu_count']}")
    print(f"  Tier       : {tier}")
    print(f"{'─'*60}")
    print(f"  Batch      : {batch}  (explicit, ~85-90% VRAM)")
    print(f"  Workers    : {workers}")
    print(f"  AMP (FP16) : True")
    print(f"{'='*60}\n", flush=True)

    return config_overrides, info

# Training Config
TRAIN_CONFIG = {
    "epochs": 100, 
    "batch_size": 16, # Default
    "img_size": (256, 256), 
    "device": "0", 
    "num_workers": 8, 
    "learning_rate": 1e-4,
    "lambda_traj": 0.1, 
    "mixed_precision": True, 
    "pin_memory": True,
    "frame_cache_size": 256,
}

if is_colab():
    try:
        _overrides, _hw_info = auto_detect_hardware()
        TRAIN_CONFIG.update(_overrides)
    except Exception as e:
        print(f"⚠️ Auto hardware detection failed: {e}", flush=True)
        print("  Falling back to default config")
