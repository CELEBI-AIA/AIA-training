# 🛩️ UAV Training Pipeline

YOLO tabanlı İHA (UAV) tespit ve GPS tabanlı konum takibi eğitim altyapısı.  
A YOLO-based UAV detection and GPS-based position tracking training pipeline designed for Google Colab.

---

## 📁 Repository Structure

```
.
├── uav_training/          # YOLO object detection module
│   ├── config.py          # Hyperparameters & paths
│   ├── train.py           # Training entrypoint
│   ├── build_dataset.py   # Dataset unification & smart sampling
│   ├── audit.py           # Dataset audit & validation
│   ├── inference.py       # Smoke test inference
│   └── visualize_dataset.py # Bounding-box visualization
│
├── gps_training/          # Siamese GPS tracker module
│   ├── config.py          # Hyperparameters & paths
│   ├── train.py           # Training entrypoint
│   ├── model.py           # SiameseTracker (ResNet-18 backbone)
│   ├── dataset.py         # Frame-pair GPS dataset loader
│   └── audit_gps.py       # Trajectory CSV scanner & validator
│
├── scripts/
│   ├── colab_bootstrap.py # One-cell Colab training launcher
│   └── cleanup.sh         # GPU memory & process cleanup
│
├── notebooks/
│   └── train_colab.ipynb  # Open in Colab notebook
│
├── requirements.txt
├── LICENSE
└── .github/workflows/     # CI (lint)
```

---

## 🎯 Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `vehicle` | Araç (car, taşıt, araç, araba) |
| 1 | `human` | İnsan (person, people, pedestrian) |
| 2 | `uap` | UAP alanı |
| 3 | `uai` | UAI alanı |

---

## 🚀 Quick Start

### Google Colab (Önerilen / Recommended)

1. Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab
2. Set `REPO_URL` to your GitHub repo URL
3. Ensure your dataset is at `Google Drive > MyDrive > train`
4. Run the cell — it handles everything automatically:
   - Drive mount → Repo clone → Dependency install → Dataset streaming → Training

### Local Setup

```bash
# Clone
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO

# Install
pip install -r requirements.txt

# Train (UAV Detection)
cd uav_training
python train.py --epochs 50 --batch 4 --device 0

# Train (GPS Tracker)
cd gps_training
python train.py

# Resume from checkpoint
python uav_training/train.py --resume
```

---

## 📦 UAV Training Module

YOLO (Ultralytics) tabanlı nesne tespit eğitimi.

**Pipeline:**
1. `audit.py` — Scans `datasets/` and generates audit report
2. `build_dataset.py` — Unifies multiple datasets with class remapping & smart sampling
3. `train.py` — Trains YOLOv8 with configurable params + auto-resume
4. `inference.py` — Quick smoke test on validation images
5. `visualize_dataset.py` — Visual verification with bounding boxes

**Key Config (`uav_training/config.py`):**
- Model: `yolov8s.pt`
- Image size: 640
- AMP: enabled
- RAM cache: enabled
- Smart sampling: 100% humans, 10% vehicles (for megaset)

See [`uav_training/README.md`](uav_training/README.md) for details.

---

## 🛰️ GPS Training Module

Siamese network (ResNet-18) ile ardışık frame'lerden Δ(x, y, z) konum tahmini.

**Pipeline:**
1. `audit_gps.py` — Scans for trajectory CSVs + matching video/images
2. `dataset.py` — Creates (frame_t, frame_t+1, Δxyz) pairs
3. `model.py` — SiameseTracker: dual ResNet-18 → concat → regressor
4. `train.py` — Training with OneCycleLR, AMP, checkpoint rotation, ONNX export

See [`gps_training/README.md`](gps_training/README.md) for details.

---

## ☁️ Colab Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   GitHub (Code)     │────▶│  /content/repo       │
└─────────────────────┘     └──────────────────────┘
                                      │
┌─────────────────────┐     ┌──────────────────────┐
│ Google Drive        │────▶│  /content/train_local│
│ MyDrive/train       │ tar │  (SSD cache)         │
└─────────────────────┘ +pv └──────────────────────┘
                                      │
                            ┌──────────────────────┐
                            │  Training (GPU)      │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │ Google Drive          │
                            │ MyDrive/runs          │
                            │ (persistent outputs)  │
                            └──────────────────────┘
```

- **Code**: GitHub → cloned fresh each session
- **Dataset**: Drive → streamed to local SSD via `tar + pv`
- **Outputs**: Saved directly to Drive (survives runtime restarts)
- **Resume**: Auto-detects `last.pt` checkpoint in Drive

---

## 🧹 Cleanup

```bash
# Kill all training processes and free GPU memory
bash scripts/cleanup.sh
```

---

## 📄 License

[MIT](LICENSE)
