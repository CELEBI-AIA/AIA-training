# UAV Training â€” YOLO Object Detection (v0.8.50)

YOLO11m (Ultralytics) tabanlÄ± Ä°HA gÃ¶rÃ¼ntÃ¼lerinden nesne tespit modÃ¼lÃ¼. Teknofest yarÄ±ÅŸmasÄ± iÃ§in optimize edilmiÅŸ.

## Class Mapping

| ID | Name | Source Aliases |
|----|------|---------------|
| 0 | `vehicle` | car, taÅŸÄ±t, araÃ§, araba |
| 1 | `human` | person, people, pedestrian, insan |
| 2 | `uap` | UAP, UAP- |
| 3 | `uai` | UAI, UAI- |

## Scripts

| Script | Purpose |
|--------|---------|
| `config.py` | Paths, class definitions, hyperparameters, IMAGE_EXTENSIONS. Colab'da `ensure_colab_config()` ile lazy hardware detection. |
| `audit.py` | `datasets/TRAIN` tarar, format doÄŸrular, `audit_report.json` Ã¼retir. |
| `build_dataset.py` | Birden fazla dataset'i birleÅŸtirir â†’ tek YOLO formatÄ±, sÄ±nÄ±f eÅŸlemesi, orphan/duplicate cleanup. |
| `train.py` | YOLO11m eÄŸitimi, auto-resume, iki fazlÄ± profil (50+15), leakage denetimi. |
| `val_utils.py` | Per-class AP50, temporal leakage check. |
| `inference.py` | Ã–rnek gÃ¶rÃ¼ntÃ¼lerde hÄ±zlÄ± inference testi. |
| `visualize_dataset.py` | Bounding box gÃ¶rselleÅŸtirme; `--split train|val|test` ile split seÃ§imi. |

## Pipeline

```bash
# Proje kÃ¶kÃ¼nden (python -m) veya uav_training/ iÃ§inden Ã§alÄ±ÅŸtÄ±rÄ±n
cd uav_training   # veya: cd AIA-training

# 1. Audit datasets
python audit.py
# veya: python -m uav_training.audit

# 2. Build unified dataset
python build_dataset.py

# 3. Train (single phase)
python train.py --epochs 65 --batch 4 --device 0

# 3b. Train (Ã¶nerilen iki fazlÄ± profil: 50 + 15)
python train.py --two-phase --batch 4 --device 0

# 4. Resume
python train.py --resume

# 5. Leakage denetimini atla (audit overlap varsa)
python train.py --allow-leakage --two-phase

# 6. GÃ¶rselleÅŸtirme (train/val/test)
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
    "phase2_mosaic": 0.0,
    "phase2_lr0": 0.0001,   # Fine-tuning iÃ§in Phase-1'den dÃ¼ÅŸÃ¼k
    "batch": 4,             # 6GB VRAM iÃ§in; Colab'da auto-detect ile override
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
    "min_bbox_norm": 0.002,
    "include_test_in_val": False,
    "remove_orphans": True,              # Labelsiz gÃ¶rÃ¼ntÃ¼leri ve gÃ¶rÃ¼ntÃ¼sÃ¼z label'larÄ± sil
    "remove_train_val_duplicates": True, # Val'deki train ile aynÄ± iÃ§erikli gÃ¶rÃ¼ntÃ¼leri sil (leakage Ã¶nleme)
}
```

**Colab:** `ensure_colab_config()` train baÅŸÄ±nda Ã§aÄŸrÄ±lÄ±r; GPU/RAM/CPU'ya gÃ¶re batch, imgsz, workers otomatik ayarlanÄ±r. Import sÄ±rasÄ±nda torch/CUDA yÃ¼klenmez (test edilebilirlik).

## Dataset Structure (after `build_dataset.py`)

```
artifacts/uav_model/dataset_uap_uai/   # veya Colab: /content/dataset_built
â”œâ”€â”€ dataset.yaml
â”œâ”€â”€ train/
â”‚   â”œâ”€â”€ images/
â”‚   â””â”€â”€ labels/
â”œâ”€â”€ val/
â”‚   â”œâ”€â”€ images/
â”‚   â””â”€â”€ labels/
â””â”€â”€ test/          # Varsa (megaset vb.)
    â”œâ”€â”€ images/
    â””â”€â”€ labels/
```

Kaynak veriler: `datasets/TRAIN/` (audit ile aynÄ± dizin).

**Desteklenen gÃ¶rÃ¼ntÃ¼ formatlarÄ±:** jpg, jpeg, png, webp, bmp, tiff, tif, gif (`IMAGE_EXTENSIONS`)

**Build sonrasÄ± otomatik temizlik:**
- Orphan cleanup: Labelsiz gÃ¶rÃ¼ntÃ¼ler ve gÃ¶rÃ¼ntÃ¼sÃ¼z label dosyalarÄ± silinir
- Train/val duplicate: Val'deki, train ile aynÄ± iÃ§erik (hash) olan gÃ¶rÃ¼ntÃ¼ler silinir

## Smart Sampling

BÃ¼yÃ¼k dataset'lerde (Ã¶rn. `megaset` 24k gÃ¶rÃ¼ntÃ¼):
- **%100** human etiketli gÃ¶rÃ¼ntÃ¼ler tutulur
- **%30** yalnÄ±zca vehicle gÃ¶rÃ¼ntÃ¼leri tutulur
- Oversampling yalnÄ±zca `train` split'e uygulanÄ±r
- `test` varsayÄ±lan olarak `val`'e merge edilmez (`include_test_in_val=False`)
- Megaset iÃ§in scene-based train/val split (`_frame_` formatÄ±)

