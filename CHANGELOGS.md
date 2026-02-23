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

## 0.0.12 - 2026-02-23
- Bumped UAV module and script versions from `0.8.0` to `0.8.1`.
- `gps_training/model.py`: Replaced deprecated `pretrained=True` with `weights='DEFAULT'` for ResNet18 to avoid future deprecation errors.
- `uav_training/build_dataset.py`: Fixed PEP-8 indentation block alignment in validation split configuration.

## 0.0.13 - 2026-02-23
- **Architectural Alignment (GPS)**: Refactored `gps_training` module to match standard optimizations from `uav_training`.
- `gps_training/config.py`: Added `auto_detect_hardware()` and `is_colab()` for dynamic VRAM-based batch sizing.
- `gps_training/train.py`: Added GPU memory cleanup hooks (`kill_gpu_hogs`), training config styling banners, and post-training Google Drive sync (`_sync_results_to_drive`).
- `gps_training/dataset.py` & `audit_gps.py`: Refactored I/O logic to check and prefer local SSD extractions (`/content/datasets_local`) in Colab, eliminating Drive FUSE latency bottlenecks.

## 0.0.14 - 2026-02-23
- Bumped all module and script versions across the repository from `0.8.1` to `0.8.2` following global GPS & UAV architectural alignment.

## 0.0.15 - 2026-02-23
- **perf(ml-pipeline)**: Applied static analysis recommendations to improve training robustness and speed.
- `uav_training/config.py`: Decreased scale augmentation from `0.2` to `0.05` to prevent small object deletion.
- `uav_training/train.py`, `gps_training/train.py`: Implemented `_is_checkpoint_valid` guard to prevent `EOFError` when resuming from corrupted checkpoints.
- `uav_training/train.py`: Added fallback mechanism for phase 2 transition to load `last.pt` if `best.pt` is missing.
- `uav_training/build_dataset.py`: Replaced `shutil.copy2` with `os.link` to eliminate disk overhead and accelerate I/O in Colab VMs.

## 0.0.16 - 2026-02-23
- Added Drive path resolution in `scripts/colab_bootstrap.py` to recover from dangling `MyDrive` shortcuts by resolving symlink targets automatically.
- Added periodic in-training sync (`rsync` every 3 minutes) from `/content/runs` to Drive to reduce checkpoint loss risk on Colab disconnects/restarts.
- **perf(training)**: Updated epoch limit to 65 (phase1=50, phase2=15).
- **fix(gps_training)**: Added `collate_drop_none` to DataLoader to discard dummy samples dynamically.
- **fix(gps_training)**: Added `torch.isfinite(loss)` check in training loop to fail fast on divergence.
- **log(uav_training)**: Explicitly log `amp_dtype: bf16` in `auto_detect_hardware()`.

## 0.0.17 - 2026-02-23
- **fix(uav_training, gps_training)**: Replaced bare except blocks with `Exception` to prevent swallowing critical errors and improve logging.
- **fix(uav_training)**: Hardened model checkpoint loading in `uav_training/train.py` with `_is_checkpoint_valid` to prevent `EOFError` during two-phase fallback.
- **fix(gps_training)**: Resolved a `VideoCapture` resource leak in `gps_training/dataset.py` by ensuring handles are properly released inside a `finally` block.
- **security(scripts)**: Mitigated shell injection vulnerability in `scripts/colab_bootstrap.py` by replacing `shell=True` execution with `shell=False`.
- **perf(gps_training)**: Optimized DataLoader by moving RGB tensor float normalization to the GPU training loop, significantly reducing PCI-e transfer overhead.
- **refactor(notebooks)**: Cleaned up `notebooks/train_colab.ipynb` by externalizing all bootstrap logic, nullifying notebook state corruption risks.
- **test**: Initialized testing infrastructure with `pytest` unit coverage for validation scripts.
