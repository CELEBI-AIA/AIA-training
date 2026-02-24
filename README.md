# 🛩️ UAV Training Pipeline — v0.8.6

YOLO11m tabanlı İHA (UAV) tespit ve GPS tabanlı konum takibi eğitim altyapısı.
Teknofest yarışması için optimize edilmiş, Google Colab üzerinde tek hücre ile çalışır.

---

## 📁 Repository Structure

```
.
├── uav_training/              # YOLO object detection module
│   ├── config.py              # Auto hardware detection & hyperparameters
│   ├── train.py               # Training entrypoint (v0.8.6)
│   ├── build_dataset.py       # Dataset unification, smart sampling & dedup
│   ├── audit.py               # Dataset audit & validation
│   ├── inference.py           # Smoke test inference
│   └── visualize_dataset.py   # Bounding-box visualization
│
├── gps_training/              # Siamese GPS tracker module
│   ├── config.py              # Hyperparameters & paths
│   ├── train.py               # Training entrypoint
│   ├── model.py               # SiameseTracker (ResNet-18 backbone)
│   ├── dataset.py             # Frame-pair GPS dataset loader
│   └── audit_gps.py           # Trajectory CSV scanner & validator
│
├── scripts/
│   ├── colab_bootstrap.py     # One-cell Colab training launcher
│   └── cleanup.sh             # GPU memory & process cleanup
│
├── notebooks/
│   └── train_colab.ipynb      # Open in Colab notebook
│
├── requirements.txt
├── LICENSE
└── .github/workflows/         # CI (lint)
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

---

## 📦 UAV Training Module

YOLO11m (Ultralytics) tabanlı nesne tespit eğitimi.

### Pipeline

1. `audit.py` — `datasets/` klasörünü tarar, audit raporu üretir
2. `build_dataset.py` — Birden fazla dataset'i birleştirir (class remapping, smart sampling, duplicate label filtering)
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
| torch.compile | ✅ A100/H100 için `reduce-overhead`, düşük tier GPU'da kapalı |
| Cache         | Dinamik (RAM >60GB: `ram`, >20GB: `disk`, diğer: off) |
| Deterministic | ❌ Off (Hızlı CUDA kernels)                 |
| Save Period   | Her 5 epoch checkpoint (Colab güvenliği)   |
| Label Filter  | `min_bbox_norm=0.004` ile filtrelenir       |

### Auto Hardware Detection (Colab)

| GPU Tier   | Batch | VRAM Usage |
|------------|-------|------------|
| H100 80GB  | 64    | ~85-90%    |
| A100 40GB  | 32    | ~85-90%    |
| L4 24GB    | 32    | ~70-80%    |
| T4 16GB    | 16    | ~75%       |
| 8GB GPU    | 8     | ~80%       |

### Dataset (30,625 train / 12,217 val)

| Dataset                          | Oversample | Smart Sample |
|----------------------------------|------------|--------------|
| Uap-UaiAlanlariVeriSeti.v2i      | 3x         | —            |
| Uap-UaiAlanlariVeriSeti          | 3x         | —            |
| drone-vision-project             | 3x         | —            |
| megaset (24k images)             | 2x         | ✅ 100% human, 10% vehicle |

---

## 🛰️ GPS Training Module

Siamese network (ResNet-18) ile ardışık frame'lerden Δ(x, y, z) konum tahmini.

1. `audit_gps.py` — Trajectory CSV + video/image tarama
2. `dataset.py` — (frame_t, frame_t+1, Δxyz) çiftleri oluşturur
3. `model.py` — SiameseTracker: dual ResNet-18 → concat → regressor
4. `train.py` — OneCycleLR, AMP, checkpoint rotation, ONNX export

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
- **Thread Limiting** — OMP/OpenBLAS/MKL = 1 thread (DataLoader contention önleme)
- **No Redundant Validation** — model.train() sonucu kullanılır, ekstra model.val() yok
- **Download Progress** — Python I/O loop ile gerçek zamanlı %, MB/s, ETA
- **Disk Management** — Otomatik cleanup, tar.gz hemen silinir
- **Label Cache** — YOLO .cache dosyaları korunur (tekrar scan yok)
- **Set-based Dedup** — O(1) duplicate label filtreleme

---

## 🧹 Cleanup

```bash
bash scripts/cleanup.sh
```

---

## 📄 License

[MIT](LICENSE)
