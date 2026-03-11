# Dataset Reference

## Archive & Path Layout

**TRAIN_DATA.tar.gz** (`MyDrive/AIA/datasets/TRAIN_DATA.tar.gz`): İçerik doğrudan üç dataset klasörüdür; üst TRAIN_DATA klasörü yoktur.

```
TRAIN_DATA.tar.gz
├── UAI_UAP/
├── drone-vision-project/
└── megaset/
```

- **Colab:** Tar extract edilir → `/content/datasets_local/{UAI_UAP, drone-vision-project, megaset}`. `repo/datasets` symlink ile `/content/datasets_local`'e bağlanır. `DATASETS_TRAIN_DIR` bu kök dizini kullanır.
- **Local:** `datasets/` altında bu üç klasör olmalı (örn. `AIA-training/datasets/UAI_UAP/`). Alternatif olarak `datasets/TRAIN_DATA/` altında da olabilir (`UAV_DATASET_SUBDIR` ile yapılandırılır).

## Source Datasets

Desteklenen klasörler: **UAI_UAP**, **drone-vision-project**, **megaset**.

### 1. UAI_UAP

| Index | Original Name | Maps To |
|-------|--------------|---------|
| 0     | UAI          | uai (3) |
| 1     | UAP          | uap (2) |

UAI/UAP iniş alanı dataset. `TRAIN_DATA.tar.gz` içinde gelir.

### 2. drone-vision-project

| Index | Original Name |
|-------|--------------|
| 0     | car          |
| 1     | pedestrian   |

Aerial drone footage — vehicle and pedestrian classes only.

### 3. megaset

| Index | Original Name |
|-------|--------------|
| 0     | vehicle      |
| 1     | pedestrian   |

Large-scale dataset (~24k images). Smart sampling: 100% human, 30% vehicle-only images kept.

## UAI- / UAP- Semantics

- **UAI** — Unmanned Aerial Vehicle Integration area, suitable for landing.
- **UAI-** — Same area type, but **not suitable** for landing (iniş yapmaya uygun olmayan).
- **UAP** — Unmanned Aerial Vehicle Platform, suitable for landing.
- **UAP-** — Same area type, but **not suitable** for landing (iniş yapmaya uygun olmayan).

During dataset build, the suitable and unsuitable variants are **merged** into the same target class because the detection objective is to locate these areas regardless of suitability.

## Unified Target Mapping

All source classes are remapped to 4 target classes by `build_dataset.py`:

| Target ID | Target Name | Source Classes Mapped                         |
|-----------|-------------|-----------------------------------------------|
| 0         | vehicle     | vehicle, car, tasit, arac, araba              |
| 1         | human       | pedestrian, people, person, human, insan      |
| 2         | uap         | UAP, UAP-                                     |
| 3         | uai         | UAI, UAI-                                     |

The canonical mapping is defined in `uav_training/config.py` (`TARGET_CLASSES`) and enforced by the `MAPPINGS` dict in `uav_training/build_dataset.py`.

The final `dataset.yaml` produced by the build step uses:

```yaml
names:
  0: vehicle
  1: human
  2: uap
  3: uai
```
