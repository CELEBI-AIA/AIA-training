# Training Audit: Small Object Threshold & Loss Balance

## 1. Current Implementation Summary

### Small-object filtering: where it happens

- Filtering is implemented in `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/build_dataset.py`.
- Threshold source:
  - `MIN_BBOX_NORM = float(TRAIN_CONFIG.get("min_bbox_norm", 0.004))` at `uav_training/build_dataset.py:76`.
- Actual filtering condition:
  - `if MIN_BBOX_NORM < w <= 1.0 and MIN_BBOX_NORM < h <= 1.0:` at `uav_training/build_dataset.py:263`.
- Bounding boxes are clamped to `[0,1]` before thresholding (`uav_training/build_dataset.py:257-260`).

### Loss weights: where defined and how forwarded

- Loss weights are defined in config:
  - `box`, `cls`, `dfl` in `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/config.py:195-197` (auto-detect overrides) and `config.py:274-276` (default `TRAIN_CONFIG`).
- Forwarding into YOLO training call:
  - Optional train args list includes `box`, `cls`, `dfl` at `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/train.py:205-211`.
  - Forwarding loop copies keys from `TRAIN_CONFIG` to `train_args` at `train.py:212-214`.
  - Effective arguments are printed by `print_training_config(train_args)` at `train.py:219`.
  - Training is called via `model.train(**train_args)` at `train.py:220`.

### Are these parameters active or silently ignored?

- In the current code, `box`, `cls`, `dfl`, `lr0`, `lrf`, `warmup_epochs`, `weight_decay`, augmentation knobs (`scale`, `copy_paste`, `copy_paste_mode`, `flipud`, `bgr`) are in `optional_params` and are forwarded if present in `TRAIN_CONFIG`.
- Therefore, in the present revision they are **active** (not silently ignored), and visibility is provided by pre-train config logging (`train.py:219`).

### Oversampling and class imbalance interaction

- Oversampling is configured per dataset using `oversample` multipliers in `MAPPINGS` (`build_dataset.py:27,41,55,70`).
- Oversampling is now train-only:
  - `oversample_count = base_oversample_count if target_split == "train" else 1` at `build_dataset.py:144`.
- Smart sampling (for `megaset`) is train-only:
  - `split_smart_sample = smart_sample and target_split == "train"` at `build_dataset.py:145`.
  - Logic: keep all images with human, and keep 10% of vehicle-only images (`build_dataset.py:273-281`).

### Per-class frequency statistics: do they exist?

- No robust per-class frequency table is computed for the built unified train/val set.
- Existing signals are partial:
  - `audit.py` tracks class-name presence flags (`present`) rather than per-class instance/image counts (`uav_training/audit.py:182-187`).
  - `build_dataset.py` logs smart-sampling counters only for human/vehicle decisions, not complete class frequencies (`build_dataset.py:176-178`, `310-311`).

---

## 2. Risks Identified

### Small object deletion risk

- Current threshold is global and static (`min_bbox_norm`), applied uniformly across all classes/scenes.
- Risk:
  - Legitimate very-small targets (especially distant UAP/UAI) may be dropped before training.
  - This can depress small-object recall while appearing to improve label cleanliness.

### Validation metric distortion risk

- Positive: current code explicitly prevents routine `test -> val` merge by default (`include_test_in_val=False` in `config.py:278`, used in `build_dataset.py:126-127`).
- Residual risk:
  - If `include_test_in_val` is manually enabled, test contamination can return.
  - There is no hard guardrail/warning in build output that this toggle changes evaluation purity.

### Oversampling side effects

- Dataset-level oversampling and smart sampling alter effective class/image priors.
- Risks:
  - Calibration drift between training distribution and real validation/test distribution.
  - Potential overfitting to duplicated samples (`_copy` variants) in train.
  - For smart sampling, balancing is binary (human vs vehicle-only) and may under-represent other classes (`uap`, `uai`) in a principled way.

### Loss weight masking imbalance risk

- `box/cls/dfl` are fixed constants from config.
- Risks:
  - Improvements in one metric can hide degradation in minority classes.
  - Without class-wise AP and frequency context, tuning may become trial-and-error and overfit to aggregate mAP.

### Silent config mismatch risk (current status)

- Historical risk: keys present in config but not forwarded.
- Current status: this specific risk is reduced because forwarding list now includes key training knobs (`train.py:205-214`) and logs effective args (`train.py:219`).
- Remaining risk: no explicit assertion that all expected keys are consumed; unknown/typo keys in config still fail silently.

---

## 3. Required Observability Improvements

### Bounding box normalized size histogram

- Add pre-train reporting for bbox size distributions from unified dataset labels:
  - `w`, `h`, `area=w*h`, class-wise and global.
  - Suggested bins: `<0.0025`, `0.0025-0.005`, `0.005-0.01`, `0.01-0.02`, `>0.02`.
- Output as table in logs and optionally CSV artifact under `artifacts/uav_model/`.

### Per-class frequency computation

- Add counts for both:
  - instance count per class (total labels),
  - image count per class (images containing at least one class id).
- Compute separately for `train` and `val` after dataset build.

### Explicit logging of effective training hyperparameters

- Keep existing `print_training_config(train_args)` and extend with:
  - explicit phase label (phase1/phase2),
  - explicit print of `min_bbox_norm` and `include_test_in_val` used during build,
  - warning if optional config keys are present but not in forwarding allowlist.

### Why this is needed

- Current logs are enough to confirm many train args are active, but insufficient to explain why small-object AP changes.
- No distributional evidence exists today to separate “cleaning effect” from “data loss effect.”

### 3.4 Effective Training Config Verification

- Printing `train_args` before `model.train(...)` is necessary but not sufficient: downstream defaults/overrides inside the trainer can still diverge from intended values.
- Runtime must log the final effective configuration exactly as applied by Ultralytics for the current run.
- Minimum expected runtime block:

```text
[CONFIG EFFECTIVE]
epochs=100
lr0=0.005
lrf=0.01
box=7.5
cls=0.7
dfl=1.5
min_bbox_norm=0.004
two_phase=True
```

> Any unknown config key must trigger a warning and not silently pass.

### 3.5 Dataset Snapshot Artifacts

- Standardized artifacts are required at:
  - `artifacts/uav_model/dataset_stats_train.csv`
  - `artifacts/uav_model/dataset_stats_val.csv`
- Each file must contain:
  - `image_count`
  - `instance_count`
  - per-class instance counts
  - normalized bbox width histogram bins
  - normalized bbox height histogram bins

Minimal table shape (example):

| split | image_count | instance_count | class_0_instances | class_1_instances | class_2_instances | class_3_instances | w_bin_lt_0.0025 | w_bin_0.0025_0.005 | w_bin_0.005_0.01 | w_bin_0.01_0.02 | w_bin_gt_0.02 | h_bin_lt_0.0025 | h_bin_0.0025_0.005 | h_bin_0.005_0.01 | h_bin_0.01_0.02 | h_bin_gt_0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 30625 | 182340 | 90500 | 74200 | 9300 | 8340 | 420 | 1830 | 9470 | 38420 | 132200 | 390 | 1760 | 9210 | 37680 | 133300 |

> These artifacts are required before any ablation run is considered valid.

---

## 4. Recommended Action Plan

1. Add bbox histogram extraction script (read YOLO labels from unified train/val and print class-wise size bins).
2. Add per-class statistics print before training starts (instances + images per class, train/val).
3. Enforce explicit train_args forwarding policy:
   - keep allowlist,
   - add warning for unexpected keys in `TRAIN_CONFIG`.
4. Run controlled ablations (one variable at a time):
   - `min_bbox_norm`: `0.003`, `0.004`, `0.005`
   - loss weights: baseline vs adjusted `cls` and `box`
   - oversampling on/off variants (train-only) with fixed seed.

Scientific tuning criteria (instead of heuristic-only):
- every change must include dataset distribution snapshot + class-wise AP deltas,
- single-variable ablation table with fixed seed/split/config,
- report both aggregate and minority-class outcomes.

### 4.4 Two-Phase Training Validation Logic

- Phase-1 is the generalization phase (baseline feature learning and robustness).
- Phase-2 is the high-resolution refinement phase; `imgsz=896` is expected to improve small-object AP.
- Before and after Phase-2, compare:
  - `mAP50-95`
  - `AP_small` (if available)
  - class-wise AP for small-object-sensitive classes (for this repo: `uap`/`uai`)
  - small bbox histogram distribution

If Phase-2 increases overall mAP but decreases AP_small, the resolution gain is not translating into true small-object learning.

---

## 5. Acceptance Criteria

- Histogram printed before training (global + class-wise normalized bbox size bins).
- Class frequency table logged (instance and image counts for train/val).
- Config values confirmed active in logs (effective `train_args` dump includes `box/cls/dfl` and related hyperparameters).
- Ablation comparison table added (at least baseline + 3 controlled variants).

---

## Verified File References

- `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/build_dataset.py`
- `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/train.py`
- `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/config.py`
- `/Users/emreilhan/Desktop/HavaciliktaYapayZeka/AIA-training/uav_training/audit.py`
