# 🛩️ UAV Training Pipeline — v0.8.40

YOLO11m tabanlı İHA (UAV) tespit eğitim altyapısı.
Teknofest yarışması için optimize edilmiş, Google Colab üzerinde tek hücre ile çalışır.

---

## 📁 Repository Structure

```
.
├── uav_training/              # YOLO object detection module
│   ├── config.py              # Auto hardware detection & hyperparameters
│   ├── train.py               # Training entrypoint
│   ├── build_dataset.py       # Dataset unification, smart sampling, dedup, orphan/duplicate cleanup
│   ├── audit.py               # Dataset audit & validation
│   ├── inference.py           # Smoke test inference
│   ├── val_utils.py           # Per-class validation, temporal leakage check
│   └── visualize_dataset.py   # Bounding-box visualization
│
├── scripts/
│   ├── colab_bootstrap.py     # One-cell Colab training launcher (runs tests automatically)
│   ├── run_all_tests.py       # Single entry point for pytest
│   ├── colab_smoke_test.py    # Quick import/validation checks (Colab)
│   ├── setup_hooks.py         # One-time setup: pre-commit install (tests on commit)
│   ├── run_per_class_val.py   # Per-class AP50 validation (vehicle, human, uap, uai)
│   └── cleanup_checkpoints.py # Epoch checkpoint cleanup (best, last, son 3 epoch)
│
├── notebooks/
│   └── train_colab.ipynb      # Open in Colab notebook
│
├── tests/                     # Unit tests (audit, build, config, val_utils, cleanup)
├── documentation/             # System check prompts & dataset docs
│
├── requirements.txt
├── CHANGELOGS.md              # Version history
├── LICENSE
└── .github/workflows/         # CI (lint, compile, pytest)
```

---

## 🎯 Detection Classes

| ID | Class     | Description                          |
|----|-----------|--------------------------------------|
| 0  | `vehicle` | Araç (car, taşıt, araç, araba)      |
| 1  | `human`   | İnsan (person, people, pedestrian)   |
| 2  | `uap`     | UAP iniş alanı                      |
| 3  | `uai`     | UAI iniş alanı                      |

---

## 🚀 Quick Start

### Google Colab (Önerilen)

1. [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) dosyasını Colab'da açın
2. Dataset'in `Google Drive > MyDrive > AIA > datasets.tar.gz` konumunda olduğundan emin olun
3. Hücreyi çalıştırın — her şey otomatik:

```
Drive mount → Repo clone → Dependencies → Hardware detect →
Dataset download (104 MB/s) → pigz extract → YOLO11m train →
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

### Otomatik Testler (Manuel Çalıştırma Gerektirmez)

- **Colab**: `colab_bootstrap` çalıştığında testler otomatik çalışır (dataset indirmeden önce).
- **CI**: Her push/PR'da GitHub Actions testleri çalıştırır.
- **Local (commit öncesi)**: Tek seferlik `python scripts/setup_hooks.py` — sonrasında her `git commit` öncesi testler otomatik çalışır.

---

## 📦 UAV Training Module

YOLO11m (Ultralytics) tabanlı nesne tespit eğitimi.

### Pipeline

1. `audit.py` — `datasets/TRAIN/` klasörünü tarar, audit raporu üretir
2. `build_dataset.py` — Birden fazla dataset'i birleştirir (class remapping, smart sampling, orphan/duplicate cleanup)
3. `train.py` — YOLO11m eğitimi (auto-resume, torch.compile, mAP-based best.pt rename)
4. `inference.py` — Validation görüntülerinde smoke test
5. `visualize_dataset.py` — Bounding box'larla görsel doğrulama

### Model & Config

| Parametre     | Değer                                      |
|---------------|--------------------------------------------|
| Model         | `yolo11m.pt` (20.1M params, 68 GFLOPs)    |
| Image Size    | 1024 (A100/H100) / 640 (T4/L4)             |
| Epochs        | 65 (Phase1: 50 + Phase2: 15, Patience: 30) |
| Optimizations | `optimizer=AdamW`, `lr0=0.001`, `close_mosaic=5` |
| Augmentations | `scale=0.4`, `copy_paste=0.3`, `flipud=0.5` |
| AMP (BF16)    | ✅ Enabled (`amp=True`, Ampere+ GPU'da Ultralytics otomatik BF16 seçer) |
| torch.compile | ✅ VRAM ≥35GB **ve** Python <3.12 için `reduce-overhead`, diğer durumlarda kapalı |
| Cache         | Dinamik (High RAM >120GB: `ram`, Normal ~80GB: `disk`, düşük: off) |
| Deterministic | ❌ Off (Hızlı CUDA kernels)                 |
| Save Period   | Her 1 epoch checkpoint + Drive sync         |
| Label Filter  | `min_bbox_norm=0.002` ile filtrelenir       |
| Image Formats | jpg, jpeg, png, webp, bmp, tiff, tif, gif   |
| Post-Build    | Orphan cleanup, train/val duplicate removal |

### Auto Hardware Detection (Colab)

**Sistem RAM:** High RAM (~167GB) vs Normal (~80GB) otomatik algılanır; High RAM'de `cache=ram` ve daha fazla worker kullanılır.

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
| Uap-UaiAlanlariVeriSeti.v2i.yolov8 | 3x        | —            |
| Uap-UaiAlanlariVeriSeti          | 3x         | —            |
| drone-vision-project             | 3x         | —            |
| megaset (24k images)             | 2x         | ✅ 100% human, 30% vehicle |

---

## ☁️ Colab Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   GitHub (Code)     │────▶│  /content/repo       │
└─────────────────────┘     └──────────────────────┘
                                      │
┌──────────────────────────┐  ┌────────────────────────┐
│ Google Drive             │─▶│  /content/datasets_local│
│ AIA/datasets.tar.gz      │  │  (NVMe SSD cache)      │
│ Python copy (104 MB/s)   │  └────────────────────────┘
│ + pigz extraction        │            │
└──────────────────────────┘  ┌─────────▼──────────┐
                              │ Auto HW Detection   │
                              │ → YOLO11m + batch    │
                              │ → torch.compile      │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │ Training (GPU)      │
                              │ /content/runs (SSD) │
                              └─────────┬──────────┘
                                        │
                   ┌────────────────────▼────────────────────┐
                   │ Google Drive                            │
                   │ AIA/runs/ (training outputs)            │
                   │ AIA/models/<date>_yolo11m_mAP.../       │
                   │ AIA/best_mAP50-X_mAP50-95-Y.pt         │
                   └────────────────────────────────────────┘
```

### Key Optimizations

- **Local SSD Training** — Drive FUSE bypass, tüm I/O NVMe SSD'de
- **torch.compile** — %20-40 hız artışı (ilk epoch derleme süresi hariç)
- **Thread Limiting** — OMP/OpenBLAS = cpus//2 (max 6), MKL = cpus//4 (max 2), torch threads = cpus//2 (max 4)
- **No Redundant Validation** — model.train() sonucu kullanılır, ekstra model.val() yok
- **Download Progress** — Python I/O loop ile gerçek zamanlı %, MB/s, ETA
- **Disk Management** — Otomatik cleanup, tar.gz hemen silinir
- **Label Cache** — YOLO .cache dosyaları korunur (tekrar scan yok)
- **Set-based Dedup** — O(1) duplicate label filtreleme

---

## 📄 License

[MIT](LICENSE)
