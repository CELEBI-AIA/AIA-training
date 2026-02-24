# UAV Training — YOLO Object Detection (v0.8.20)

YOLO11m (Ultralytics) tabanlı İHA görüntülerinden nesne tespit modülü. Teknofest yarışması için optimize edilmiş.

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
| `config.py` | Paths, class definitions, hyperparameters. Colab'da `ensure_colab_config()` ile lazy hardware detection. |
| `audit.py` | `datasets/TRAIN` tarar, format doğrular, `audit_report.json` üretir. |
| `build_dataset.py` | Birden fazla dataset'i birleştirir → tek YOLO formatı, sınıf eşlemesi. |
| `train.py` | YOLO11m eğitimi, auto-resume, iki fazlı profil (50+15), leakage denetimi. |
| `inference.py` | Örnek görüntülerde hızlı inference testi. |
| `visualize_dataset.py` | Bounding box görselleştirme; `--split train|val|test` ile split seçimi. |

## Pipeline

```bash
# Proje kökünden (python -m) veya uav_training/ içinden çalıştırın
cd uav_training   # veya: cd AIA-training

# 1. Audit datasets
python audit.py
# veya: python -m uav_training.audit

# 2. Build unified dataset
python build_dataset.py

# 3. Train (single phase)
python train.py --epochs 65 --batch 4 --device 0

# 3b. Train (önerilen iki fazlı profil: 50 + 15)
python train.py --two-phase --batch 4 --device 0

# 4. Resume
python train.py --resume

# 5. Leakage denetimini atla (audit overlap varsa)
python train.py --allow-leakage --two-phase

# 6. Görselleştirme (train/val/test)
python visualize_dataset.py --split train --num 20
python visualize_dataset.py --split val --num 10

# 7. Inference test
python inference.py --model path/to/best.pt --source path/to/images
```

## Config Reference (`config.py`)

```python
TRAIN_CONFIG = {
    "epochs": 65,
    "phase1_epochs": 50,
    "phase2_epochs": 15,
    "phase2_imgsz": 896,
    "phase2_mosaic": 0.2,
    "phase2_lr0": 0.0001,   # Fine-tuning için Phase-1'den düşük
    "batch": 4,             # 6GB VRAM için; Colab'da auto-detect ile override
    "imgsz": 640,
    "device": 0,
    "model": "yolo11m.pt",
    "workers": 8,
    "amp": True,            # Mixed precision (BF16 Ampere+)
    "cache": "disk",       # Local default; Colab'da RAM/disk/False dinamik
    "patience": 30,
    "cos_lr": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 5.0,
    "weight_decay": 0.0005,
    "mosaic": 1.0,
    "close_mosaic": 5,
    "box": 7.5, "cls": 0.7, "dfl": 1.5,
    "min_bbox_norm": 0.004,
    "include_test_in_val": False,
}
```

**Colab:** `ensure_colab_config()` train başında çağrılır; GPU/RAM/CPU'ya göre batch, imgsz, workers otomatik ayarlanır. Import sırasında torch/CUDA yüklenmez (test edilebilirlik).

## Dataset Structure (after `build_dataset.py`)

```
artifacts/uav_model/dataset_uap_uai/   # veya Colab: /content/dataset_built
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/          # Varsa (megaset vb.)
    ├── images/
    └── labels/
```

Kaynak veriler: `datasets/TRAIN/` (audit ile aynı dizin).

## Smart Sampling

Büyük dataset'lerde (örn. `megaset` 24k görüntü):
- **%100** human etiketli görüntüler tutulur
- **%30** yalnızca vehicle görüntüleri tutulur
- Oversampling yalnızca `train` split'e uygulanır
- `test` varsayılan olarak `val`'e merge edilmez (`include_test_in_val=False`)
- Megaset için scene-based train/val split (`_frame_` formatı)
