# GPS Training — Siamese Visual Odometry

Ardışık drone frame'lerinden Δ(x, y, z) konum değişimi tahmin eden Siamese ağ modülü.

## Architecture

```
Frame t ──► ResNet-18 ──► f₁ (512-d)  ──┐
                                          ├──► Concat (1024-d) ──► MLP ──► Δ(x,y,z)
Frame t+1 ──► ResNet-18 ──► f₂ (512-d) ──┘
```

- **Backbone**: ResNet-18 (ImageNet pretrained, weight-shared)
- **Head**: MLP regressor (1024 → 512 → 256 → 3)
- **Output**: Translation delta `[Δx, Δy, Δz]`

## Scripts

| Script | Purpose |
|--------|---------|
| `config.py` | Paths and training hyperparameters |
| `audit_gps.py` | Scans for trajectory CSVs, finds matching media, validates |
| `dataset.py` | `GPSDataset` — creates `(frame_t, frame_t+1, Δxyz)` pairs |
| `model.py` | `SiameseTracker` network definition |
| `train.py` | Full training loop with AMP, checkpoint rotation, ONNX export |

## Pipeline

```bash
# 1. Audit — find valid trajectory sequences
python audit_gps.py

# 2. Train
python train.py

# 3. Resume from checkpoint
python train.py --resume
```

## Config Reference (`config.py`)

```python
TRAIN_CONFIG = {
    "epochs": 5,
    "batch_size": 16,       # Adjusted for 6GB VRAM
    "img_size": (256, 256),
    "num_workers": 8,
    "learning_rate": 1e-4,
    "mixed_precision": True, # AMP
}
```

## Data Format

Expects trajectory CSV files with columns:
- `frame_numbers` — Frame index
- `translation_x`, `translation_y`, `translation_z` — Absolute position

Media can be:
- **Video**: `.mp4` file alongside CSV
- **Images**: Directory of `.webp` / `.jpg` / `.png` frames

## Checkpoint Strategy

- `last_model.pt` — Latest checkpoint (rotated with `.bak` backup)
- `best_model.pt` — Best validation loss (raw state_dict)
- `gps_model.onnx` — Exported ONNX model (post-training)
