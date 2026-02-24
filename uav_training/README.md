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
| `train.py` | YOLO11 training with auto-resume and optional two-phase profile |
| `inference.py` | Quick inference smoke test on sample images |
| `visualize_dataset.py` | Draws bounding boxes on random samples for visual verification |

## Pipeline

```bash
# 1. Audit datasets
python audit.py

# 2. Build unified dataset
python build_dataset.py

# 3. Train (single phase)
python train.py --epochs 65 --batch 4 --device 0

# 3b. Train (recommended two-phase profile: 50 + 15)
python train.py --two-phase --batch 4 --device 0

# 4. Resume
python train.py --resume

# 5. Inference test
python inference.py --model path/to/best.pt --source path/to/images
```

## Config Reference (`config.py`)

```python
TRAIN_CONFIG = {
    "epochs": 65,
    "phase1_epochs": 50,
    "phase2_epochs": 15,
    "batch": 4,           # Optimized for 6GB VRAM
    "imgsz": 640,
    "device": 0,
    "model": "yolo11m.pt",
    "workers": 8,
    "amp": True,          # Mixed precision (BF16 target on Ampere+)
    "cache": True,         # Local default, Colab uses dynamic RAM-based selection
    "patience": 30,        # Early stopping
    "cos_lr": True,        # Cosine LR scheduler
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 5.0,
    "weight_decay": 0.0005,
    "mosaic": 1.0,         # Phase-1 augmentation
    "close_mosaic": 5,
    "box": 7.5, "cls": 0.7, "dfl": 1.5,
    "min_bbox_norm": 0.004,
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
- Oversampling is applied only to `train`, not `val`
- `test` split is not merged into `val` by default (`include_test_in_val=False`)
- This balances class distribution without leaking validation quality
