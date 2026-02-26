# Dataset Reference

Training datasets are located under `datasets/TRAIN/` (relative to project root). The code uses `DATASETS_TRAIN_DIR = PROJECT_ROOT / "datasets" / "TRAIN"`.

- **Local:** `datasets/TRAIN/` (e.g. `AIA-training/datasets/TRAIN/`)
- **Colab:** The bootstrap symlinks `repo/datasets` → `/content/datasets_local`, so `DATASETS_TRAIN_DIR` resolves to `/content/datasets_local/TRAIN/` after extraction.

## Source Datasets

### 1. megaset

| Index | Original Name |
|-------|--------------|
| 0     | vehicle      |
| 1     | pedestrian   |

Large-scale dataset (~24k images). Smart sampling is applied during build: 100% of human annotations are kept while 30% of vehicle-only images are kept to reduce class imbalance.

### 2. drone-vision-project

| Index | Original Name |
|-------|--------------|
| 0     | car          |
| 1     | pedestrian   |

Aerial drone footage — vehicle and pedestrian classes only.

### 3. Uap-UaiAlanlariVeriSeti.v2i.yolov8

| Index | Original Name | Description                                         |
|-------|--------------|-----------------------------------------------------|
| 0     | UAI          | İniş yapmaya **uygun** UAI alanı (suitable)         |
| 1     | UAI-         | İniş yapmaya **uygun olmayan** UAI alanı (unsuitable)|
| 2     | UAP          | İniş yapmaya **uygun** UAP alanı (suitable)         |
| 3     | UAP-         | İniş yapmaya **uygun olmayan** UAP alanı (unsuitable)|
| 4     | car          | Araç                                                |
| 5     | people       | İnsan                                               |

### 4. Uap-UaiAlanlariVeriSeti

| Index | Original Name | Description                                         |
|-------|--------------|-----------------------------------------------------|
| 0     | UAI          | İniş yapmaya **uygun** UAI alanı (suitable)         |
| 1     | UAI-         | İniş yapmaya **uygun olmayan** UAI alanı (unsuitable)|
| 2     | UAP          | İniş yapmaya **uygun** UAP alanı (suitable)         |
| 3     | UAP-         | İniş yapmaya **uygun olmayan** UAP alanı (unsuitable)|
| 4     | car          | Araç                                                |
| 5     | people       | İnsan                                               |

### 5. visdrone_yolo

| Index | Original Name    | Maps To  |
|-------|------------------|----------|
| 0     | pedestrian       | human    |
| 1     | people           | human    |
| 2     | bicycle          | vehicle  |
| 3     | car              | vehicle  |
| 4     | van              | vehicle  |
| 5     | truck            | vehicle  |
| 6     | tricycle         | vehicle  |
| 7     | awning-tricycle  | vehicle  |
| 8     | bus              | vehicle  |
| 9     | motor            | vehicle  |

VisDrone aerial detection dataset (~8k+ images). UAP/UAI sınıfları yok; sadece vehicle ve human hedeflerine map edilir. **İnsanlara öncelik:** Smart sampling ile %100 human, %30 vehicle tutulur; pure-human görüntülere ekstra oversample uygulanır. Oversample: 3x.

**Klasör yapısı:** `datasets/TRAIN/visdrone_yolo/` altında `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels` beklenir.

**data.yaml örneği** (kök dizinde `data.yaml`). Örnek dosya: [`visdrone_yolo_data.yaml.example`](visdrone_yolo_data.yaml.example) — `datasets/TRAIN/visdrone_yolo/data.yaml` olarak kopyalayın.

```yaml
train: train/images
val: valid/images
test: test/images

nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```

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
