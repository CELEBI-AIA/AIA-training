# UAV Training — YOLO Object Detection

YOLO (Ultralytics) tabanlı İHA görüntülerinden nesne tespit modülü.

## Class Mapping

| ID | Name | Source Aliases |
|----|------|---------------|
| 0 | `vehicle` | car, taşıt, araç, araba |
| 1 | `human` | person, people, pedestrian, insan |
| 2 | `uap` | UAP, UAP- |
| 3 | `uai` | UAI, UAI- |

## Scripts

| Script | Purpose |
|--------|---------|
| `config.py` | All paths, class definitions, and training hyperparameters |
| `audit.py` | Scans `datasets/` directory, validates format, generates `audit_report.json` |
| `build_dataset.py` | Merges multiple datasets → unified YOLO format with class remapping |
| `train.py` | YOLOv8 training with auto-resume support |
| `inference.py` | Quick inference smoke test on sample images |
| `visualize_dataset.py` | Draws bounding boxes on random samples for visual verification |

## Pipeline

```bash
# 1. Audit datasets
python audit.py

# 2. Build unified dataset
python build_dataset.py

# 3. Train
python train.py --epochs 50 --batch 4 --device 0

# 4. Resume
python train.py --resume

# 5. Inference test
python inference.py --model path/to/best.pt --source path/to/images
```

## Config Reference (`config.py`)

```python
TRAIN_CONFIG = {
    "epochs": 10,
    "batch": 4,           # Optimized for 6GB VRAM
    "imgsz": 640,
    "device": 0,
    "model": "yolov8s.pt",
    "workers": 8,
    "amp": True,           # Mixed precision
    "cache": True,         # RAM cache
    "patience": 50,        # Early stopping
    "cos_lr": True,        # Cosine LR scheduler
    "mosaic": 0.5,         # Mosaic augmentation
}
```

## Dataset Structure (after `build_dataset.py`)

```
artifacts/uav_model/dataset_uap_uai/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

## Smart Sampling

For large datasets (e.g., `megaset` with 24k images):
- **100%** of images with `human` annotations are kept
- **10%** of vehicle-only images are kept
- This balances class distribution without manual curation
