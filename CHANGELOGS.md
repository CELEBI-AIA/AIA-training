# CHANGELOGS

## 0.0.04 - 2026-02-21
- **Colab Auto Hardware Detection**: `uav_training/config.py` now auto-detects GPU/RAM/CPU on Colab and maximizes batch/workers/imgsz/model size.
- **Dataset `.tar.gz` Support**: `scripts/colab_bootstrap.py` now extracts `datasets.tar.gz` from Drive using `pigz` (parallel decompression) with local SSD caching.
- **Post-Training Export**: `uav_training/train.py` renames `best.pt` with mAP50 and mAP50-95 scores appended and uploads to `Drive/AIA/`.
- **GitHub URL Updated**: All references now point to `CELEBI-AIA/AIA-training`.
- **Notebook Synced**: `notebooks/train_colab.ipynb` updated with 7-step auto-tuned pipeline.

## 0.0.01 - 2026-02-21
- Fixed resume robustness in `gps_training/train.py` to avoid KeyError when `scheduler_state_dict` is missing in older/incomplete checkpoints.
- Added guarded loading for both optimizer and scheduler states with warning-based fallback.

## 0.0.02 - 2026-02-21
- Added process-level file locking for `gps_training/train.py` checkpoint flow to prevent parallel run overwrite/race on `last_model.pt` and backup files.
- Added process-level file locking for `uav_training/build_dataset.py` to prevent concurrent dataset rebuild race on `dataset_uap_uai/`.

## 0.0.03 - 2026-02-21
- Optimized `gps_training/dataset.py` video loading with worker-local `VideoCapture` reuse and LRU frame cache to reduce repeated open/seek/decode overhead.
- Added pair-read fast path for consecutive frames (`t`, `t+1`) to prefetch both frames in one decode flow.
- Added safe resource cleanup (`VideoCapture.release`) via `atexit` and dataset destructor to prevent handle leaks when worker processes exit.
- Added `frame_cache_size` to `gps_training/config.py` for cache tuning.

## 0.0.05 - 2026-02-22
- Training config now defaults to a 100-epoch profile with explicit phase settings (`phase1=85`, `phase2=15`) and tuned optimizer/loss parameters (`lr0`, `lrf`, `warmup_epochs`, `weight_decay`, `box/cls/dfl`).
- `uav_training/train.py` now forwards all key hyperparameters/augment settings from `TRAIN_CONFIG` to `model.train(...)`, removing silent config no-op risk.
- Added `--two-phase` flag in `uav_training/train.py` to run a short two-stage training flow (85+15) with high-resolution phase-2 fine-tune.
- `uav_training/build_dataset.py` now enforces cleaner validation data: oversampling/smart sampling are train-only, and `test->val` merge is disabled by default.
- Added configurable small-object bbox threshold via `min_bbox_norm` (default `0.004`) instead of hard-coded `0.005`.
- Updated documentation (`README.md`, `uav_training/README.md`) and module version to `0.8.0`.

## 0.0.06 - 2026-02-22
- Added internal training audit documentation at `documentation/TRAINING_AUDIT_SMALL_OBJECT_AND_LOSS_BALANCE.md`.
- Documented verified code paths for small-object thresholding (`min_bbox_norm`) and loss-weight forwarding (`box/cls/dfl`) with file-level references.
- Added risk analysis for validation distortion, oversampling side effects, and observability gaps, plus acceptance criteria for scientific ablation tracking.

## 0.0.07 - 2026-02-22
- Extended `documentation/TRAINING_AUDIT_SMALL_OBJECT_AND_LOSS_BALANCE.md` with `3.4 Effective Training Config Verification` and explicit unknown-key warning rule.
- Added `3.5 Dataset Snapshot Artifacts` contract with required CSV paths and required columns/hist bins.
- Added `4.4 Two-Phase Training Validation Logic` clarifying phase intent and required pre/post Phase-2 comparisons.

## 0.0.08 - 2026-02-22
- Prepended a non-technical Turkish summary to `documentation/TRAINING_AUDIT_SMALL_OBJECT_AND_LOSS_BALANCE.md` for broader team readability.

## 0.0.09 - 2026-02-22
- Upgraded `.github/workflows/lint.yml` from lint-only to a broader CI quality gate on `push` and `pull_request` for `main`.
- CI now enforces syntax-focused `flake8` checks (`E9,F63,F7,F82`) to catch parser/name-critical failures early.
- Added `python -m compileall -q .` compile validation across the repository to surface syntax regressions before merge.
- Removed heavy runtime dependency installation from CI path for faster, more reliable checks in a non-GPU runner context.

## 0.0.10 - 2026-02-22
- Updated `scripts/colab_bootstrap.py` version banner from `v0.7.3` to `v0.8.0` to align with current training module release and reduce operator confusion.
- Reduced dependency-install log noise in Colab bootstrap by capturing pip output and printing concise install summaries (`updated packages` vs `already satisfied`).

## 0.0.11 - 2026-02-22
- Made bootstrap version banner dynamic in `scripts/colab_bootstrap.py` by reading `uav_training/__init__.py::__version__` instead of using a hard-coded `VERSION` string.
- Added a post-repo-sync version refresh message to confirm the active bootstrap/module version in runtime logs.
