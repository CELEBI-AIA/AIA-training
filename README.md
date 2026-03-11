# ğŸ›©ï¸ UAV Training Pipeline â€” v0.8.50

YOLO11m tabanlÄ± Ä°HA (UAV) tespit eÄŸitim altyapÄ±sÄ±.
Teknofest yarÄ±ÅŸmasÄ± iÃ§in optimize edilmiÅŸ, Google Colab Ã¼zerinde tek hÃ¼cre ile Ã§alÄ±ÅŸÄ±r.

---

## ğŸ“ Repository Structure

```
.
â”œâ”€â”€ uav_training/              # YOLO object detection module
â”‚   â”œâ”€â”€ config.py              # Auto hardware detection & hyperparameters
â”‚   â”œâ”€â”€ train.py               # Training entrypoint
â”‚   â”œâ”€â”€ build_dataset.py       # Dataset unification, smart sampling, dedup, orphan/duplicate cleanup
â”‚   â”œâ”€â”€ audit.py               # Dataset audit & validation
â”‚   â”œâ”€â”€ inference.py           # Smoke test inference
â”‚   â”œâ”€â”€ val_utils.py           # Per-class validation, temporal leakage check
â”‚   â””â”€â”€ visualize_dataset.py   # Bounding-box visualization
â”‚
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ colab_bootstrap.py     # One-cell Colab training launcher (runs tests automatically)
â”‚   â”œâ”€â”€ run_all_tests.py       # Single entry point for pytest
â”‚   â”œâ”€â”€ colab_smoke_test.py    # Quick import/validation checks (Colab)
â”‚   â”œâ”€â”€ setup_hooks.py         # One-time setup: pre-commit install (tests on commit)
â”‚   â”œâ”€â”€ run_per_class_val.py   # Per-class AP50 validation (vehicle, human, uap, uai)
â”‚   â””â”€â”€ cleanup_checkpoints.py # Epoch checkpoint cleanup (best, last, son 3 epoch)
â”‚
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ train_colab.ipynb      # Open in Colab notebook
â”‚
â”œâ”€â”€ tests/                     # Unit tests (audit, build, config, val_utils, cleanup)
â”œâ”€â”€ documentation/             # System check prompts & dataset docs
â”‚
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ CHANGELOGS.md              # Version history
â”œâ”€â”€ LICENSE
â””â”€â”€ .github/workflows/         # CI (lint, compile, pytest)
```

---

## ğŸ¯ Detection Classes

| ID | Class     | Description                          |
|----|-----------|--------------------------------------|
| 0  | `vehicle` | AraÃ§ (car, taÅŸÄ±t, araÃ§, araba)      |
| 1  | `human`   | Ä°nsan (person, people, pedestrian)   |
| 2  | `uap`     | UAP iniÅŸ alanÄ±                      |
| 3  | `uai`     | UAI iniÅŸ alanÄ±                      |

---

## ğŸš€ Quick Start

### Google Colab (Ã–nerilen)

1. [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) dosyasÄ±nÄ± Colab'da aÃ§Ä±n
2. Dataset'in `Google Drive > MyDrive > AIA > datasets > mega.tar.gz` konumunda olduÄŸundan emin olun
3. HÃ¼creyi Ã§alÄ±ÅŸtÄ±rÄ±n â€” her ÅŸey otomatik:

```
Drive mount â†’ Repo clone â†’ Dependencies â†’ Hardware detect â†’
Dataset download (104 MB/s) â†’ pigz extract â†’ YOLO11m train â†’
best.pt rename & Drive upload
```

### Local Setup

```bash
git clone https://github.com/CELEBI-AIA/AIA-training.git
cd AIA-training
pip install -r requirements.txt

# Train (UAV Detection)
cd uav_training
python train.py --epochs 65 --batch 4 --device 0

# Recommended two-phase profile (50 + 15)
python train.py --two-phase --batch 4 --device 0

# Resume from checkpoint
python train.py --resume

# Custom model
python train.py --model yolo11m.pt --batch 16
```

### Otomatik Testler (Manuel Ã‡alÄ±ÅŸtÄ±rma Gerektirmez)

- **Colab**: `colab_bootstrap` Ã§alÄ±ÅŸtÄ±ÄŸÄ±nda testler otomatik Ã§alÄ±ÅŸÄ±r (dataset indirmeden Ã¶nce).
- **CI**: Her push/PR'da GitHub Actions testleri Ã§alÄ±ÅŸtÄ±rÄ±r.
- **Local (commit Ã¶ncesi)**: Tek seferlik `python scripts/setup_hooks.py` â€” sonrasÄ±nda her `git commit` Ã¶ncesi testler otomatik Ã§alÄ±ÅŸÄ±r.

---

## ğŸ“¦ UAV Training Module

YOLO11m (Ultralytics) tabanlÄ± nesne tespit eÄŸitimi.

### Pipeline

1. `audit.py` â€” `datasets/TRAIN/` klasÃ¶rÃ¼nÃ¼ tarar, audit raporu Ã¼retir
2. `build_dataset.py` â€” Birden fazla dataset'i birleÅŸtirir (class remapping, smart sampling, orphan/duplicate cleanup)
3. `train.py` â€” YOLO11m eÄŸitimi (auto-resume, torch.compile, mAP-based best.pt rename)
4. `inference.py` â€” Validation gÃ¶rÃ¼ntÃ¼lerinde smoke test
5. `visualize_dataset.py` â€” Bounding box'larla gÃ¶rsel doÄŸrulama

### Model & Config

| Parametre     | DeÄŸer                                      |
|---------------|--------------------------------------------|
| Model         | `yolo11m.pt` (20.1M params, 68 GFLOPs)    |
| Image Size    | 1024 (A100/H100) / 640 (T4/L4)             |
| Epochs        | 65 (Phase1: 50 + Phase2: 15, Patience: 30) |
| Optimizations | `optimizer=AdamW`, `lr0=0.001`, `close_mosaic=5` |
| Augmentations | `scale=0.4`, `copy_paste=0.3`, `flipud=0.5` |
| AMP (BF16)    | âœ… Enabled (`amp=True`, Ampere+ GPU'da Ultralytics otomatik BF16 seÃ§er) |
| torch.compile | âœ… VRAM â‰¥35GB **ve** Python <3.12 iÃ§in `reduce-overhead`, diÄŸer durumlarda kapalÄ± |
| Cache         | Dinamik (High RAM >120GB: `ram`, Normal ~80GB: `disk`, dÃ¼ÅŸÃ¼k: off) |
| Deterministic | âŒ Off (HÄ±zlÄ± CUDA kernels)                 |
| Save Period   | Her 1 epoch checkpoint + Drive sync         |
| Label Filter  | `min_bbox_norm=0.002` ile filtrelenir       |
| Image Formats | jpg, jpeg, png, webp, bmp, tiff, tif, gif   |
| Post-Build    | Orphan cleanup, train/val duplicate removal |

### Auto Hardware Detection (Colab)

**Sistem RAM:** High RAM (~167GB) vs Normal (~80GB) otomatik algÄ±lanÄ±r; High RAM'de `cache=ram` ve daha fazla worker kullanÄ±lÄ±r.

| GPU Tier   | Batch | ImgSz | VRAM Usage |
|------------|-------|-------|------------|
| H100 80GB  | 32    | 1024  | ~85-90%    |
| A100 80GB  | 32    | 1024  | ~85-90%    |
| A100 40GB  | 24    | 1024  | ~85-90%    |
| L4 24GB    | 32    | 640   | ~70-80%    |
| T4 16GB    | 16    | 640   | ~75%       |
| 8GB GPU    | 8     | 640   | ~80%       |

### Dataset (30,625 train / 12,217 val)

| Dataset                          | Oversample | Smart Sample |
|----------------------------------|------------|--------------|
| Uap-UaiAlanlariVeriSeti.v2i.yolov8 | 3x        | â€”            |
| Uap-UaiAlanlariVeriSeti          | 3x         | â€”            |
| drone-vision-project             | 3x         | â€”            |
| megaset (24k images)             | 2x         | âœ… 100% human, 30% vehicle |
| visdrone_yolo (~8k images)       | 3x         | âœ… 100% human, 30% vehicle |

---

## â˜ï¸ Colab Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   GitHub (Code)     â”‚â”€â”€â”€â”€â–¶â”‚  /content/repo       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                      â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Google Drive             â”‚â”€â–¶â”‚  /content/datasets_localâ”‚
â”‚ AIA/datasets/mega.tar.gz â”‚  â”‚  (NVMe SSD cache)      â”‚
â”‚ Python copy (104 MB/s)   â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚ + pigz extraction        â”‚            â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ Auto HW Detection   â”‚
                              â”‚ â†’ YOLO11m + batch    â”‚
                              â”‚ â†’ torch.compile      â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                        â”‚
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ Training (GPU)      â”‚
                              â”‚ /content/runs (SSD) â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                        â”‚
                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                   â”‚ Google Drive                            â”‚
                   â”‚ AIA/runs/ (training outputs)            â”‚
                   â”‚ AIA/models/<date>_yolo11m_mAP.../       â”‚
                   â”‚ AIA/best_mAP50-X_mAP50-95-Y.pt         â”‚
                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Key Optimizations

- **Local SSD Training** â€” Drive FUSE bypass, tÃ¼m I/O NVMe SSD'de
- **torch.compile** â€” %20-40 hÄ±z artÄ±ÅŸÄ± (ilk epoch derleme sÃ¼resi hariÃ§)
- **Thread Limiting** â€” OMP/OpenBLAS = cpus//2 (max 6), MKL = cpus//4 (max 2), torch threads = cpus//2 (max 4)
- **No Redundant Validation** â€” model.train() sonucu kullanÄ±lÄ±r, ekstra model.val() yok
- **Download Progress** â€” Python I/O loop ile gerÃ§ek zamanlÄ± %, MB/s, ETA
- **Disk Management** â€” Otomatik cleanup, tar.gz hemen silinir
- **Label Cache** â€” YOLO .cache dosyalarÄ± korunur (tekrar scan yok)
- **Set-based Dedup** â€” O(1) duplicate label filtreleme

---

## ğŸ“„ License

[MIT](LICENSE)

