# CHANGELOGS

Version format: UAV module uses `0.8.XX` (e.g. 0.8.35). Historical entries may use `0.0.XX`; new entries should use `0.8.XX`.

---

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

## 0.0.18 - 2026-02-23
- **feat(gps_training)**: Aligned scheduler LR policy with config by introducing `resolve_scheduler_max_lr()` and replacing hardcoded `OneCycleLR(max_lr=1e-3)` usage.
- **feat(uav_training/build_dataset)**: Added strict split mapping via `resolve_target_split()` with explicit `test` handling (`test -> test` by default, `test -> val` only when `include_test_in_val=True`).
- **feat(uav_training/build_dataset)**: Enabled native `test` split output directories and writes (`test/images`, `test/labels`) plus conditional `test: test/images` export in `dataset.yaml`.
- **feat(uav_training/build_dataset)**: Reworked smart sampling to class-aware keep probabilities across all target classes and added warnings for out-of-range/unmapped class IDs.
- **feat(uav_training/audit)**: Added split-level image/label counts and overlap risk reporting (`train-val-test`) for early leakage detection.
- **test**: Added targeted unit tests for split mapping policy, scheduler max_lr validation, and audit overlap/count reporting.

## 0.0.19 - 2026-02-23
- **feat(uav_training/build_dataset)**: Tuned TEKNOFEST smart-sampling default for `vehicle` class from `0.10` to `0.30` to reduce missed-vehicle risk while keeping dataset balancing.

## 0.0.20 - 2026-02-23
- **fix(uav_training/train)**: Added OOM-safe retry flow in training phase execution: retry with `compile=False`, then progressively lower batch/imgsz only when CUDA OOM occurs.
- **fix(uav_training/train)**: Disabled BF16 monkey patch by default to avoid conflicts with Ultralytics AMP safety checks; patch can still be enabled via `FORCE_BF16_PATCH=1`.
- **chore(uav_training/train)**: Removed duplicate `threading` import.

## 0.0.21 - 2026-02-24
- **fix(uav_training/resume)**: Hardened resume path resolution in `uav_training/train.py` with a 3-step fallback chain (CLI `--model` -> local runs -> Drive runs) to prevent accidental fresh starts after Colab reset.
- **fix(scripts/colab_bootstrap)**: Resume launch now passes explicit checkpoint path (`--model <last.pt> --resume`) so training state is restored deterministically.
- **perf(uav_training/seed-tf32)**: Refactored `setup_seed()` in `uav_training/train.py` to honor config-driven determinism, re-enable `cudnn.benchmark` in speed mode, and apply `torch.set_float32_matmul_precision("high")`.
- **perf(uav_training/optimizer)**: Switched training defaults to explicit `AdamW` in `uav_training/config.py` and forwarded `optimizer/momentum/nbs` in `uav_training/train.py` to avoid Ultralytics `optimizer=auto` overriding LR policy.
- **perf(uav_training/bf16)**: Added Ampere+ compute capability detection in `scripts/colab_bootstrap.py` to auto-set `FORCE_BF16_PATCH=1`; startup logs now show GPU SM capability and BF16 patch state in `uav_training/train.py`.
- **fix(uav_training/build_dataset)**: Replaced silent bbox clamping with strict validation (NaN/out-of-range/too-small counters) and added `[BBOX AUDIT]` summary logs to surface label quality issues before training.
- **perf(scripts/colab_bootstrap)**: Increased periodic Drive sync interval from 180s to 300s to reduce I/O contention during training.
- **perf(uav_training/config)**: Raised A100-tier DataLoader worker cap from 8 to 10 for better GPU feed stability.
- **chore(deps)**: Added upper bounds in `requirements.txt` (`ultralytics<9.0.0`, `torch<3.0.0`, `torchvision<1.0.0`) for better reproducibility against upstream breaking changes.

## 0.0.22 - 2026-02-24
- **log(uav_training/precision)**: Strengthened startup precision proof in `uav_training/train.py` with a single `[PRECISION]` line that now includes `gpu_capability`, `bf16_patch`, and `amp_dtype` (plus TF32/compile state) for deterministic BF16 observability.
- **perf(scripts/colab_bootstrap)**: Added optional quiet-window control via `UAV_SYNC_QUIET_WINDOW_SEC` in periodic checkpoint sync flow so recent `last.pt` writes can settle before rsync.
- **log(scripts/colab_bootstrap)**: Periodic sync log now includes checkpoint `mtime` and still syncs only when newest `last.pt` changes, improving I/O jitter diagnostics.
- **fix(uav_training/inference)**: Updated default model in `uav_training/inference.py` from `yolov8n.pt` to `yolo11m.pt` to avoid operational model-selection confusion.

## 0.0.23 - 2026-02-24
- **docs(training-config)**: Updated `README.md` and `uav_training/README.md` to match current training defaults (`epochs=65`, `phase1=50`, `phase2=15`, AdamW/BF16-target settings, and A100 compile profile notes).
- **fix(uav_training/build_dataset)**: Made dataset build file-locking cross-platform by adding a Windows lock fallback (`msvcrt`) while preserving Linux/Colab lock behavior (`fcntl`).
- **docs(audit)**: Added `documentation/rapor_uyum_dogrulama.md` with report-to-code mapping (`UYGULANDI/KISMİ/EKSİK`) and runtime proof checklist for Colab A100 validation.

## 0.0.24 - 2026-02-24
- **fix(gps_training/precision)**: `gps_training/train.py` updated with CUDA-guarded autocast (`nullcontext` CPU fallback) in both train and validation loops to prevent CPU fallback crashes and enable consistent BF16 validation path.
- **fix(gps_training/reproducibility)**: Added global seed/determinism chain in `gps_training/train.py` (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, `cuda`) and wired it into training startup.
- **fix(gps_training/dataset)**: Replaced silent `return None` failure path in `gps_training/dataset.py` with structured logging and fail-fast `RuntimeError`.
- **fix(uav_training/bf16)**: `uav_training/train.py` now forwards `amp_dtype="bf16"` into `train_args` when CUDA BF16 support is available; `uav_training/config.py` default now explicitly includes `amp_dtype`.
- **fix(uav_training/sync)**: Added single-flight checkpoint sync guard in `uav_training/train.py` to reduce overlapping Drive sync races and I/O spikes.
- **perf(scripts/colab_bootstrap)**: Reworked DataLoader thread limiter in `scripts/colab_bootstrap.py` with CPU-aware dynamic thread/env settings instead of fixed low-thread caps.
- **chore(deps)**: Tightened core dependency ranges in `requirements.txt` (`ultralytics`, `torch`, `torchvision`, `numpy`) to reduce long-term reproducibility drift.
- **docs(reporting)**: Rebuilt `coderapor.md` scorecard and bug status mapping from current code state (`FIXED/PARTIAL/OPEN`) with updated risk narrative.
- **release**: Bumped module/script version from `0.8.4` to `0.8.5`.

## 0.0.25 - 2026-02-24
- **perf(uav_training/tf32)**: Enabled TF32 matmul and cuDNN flags (`allow_tf32=True`) inside `auto_detect_hardware()` in `uav_training/config.py` for ~15-25% speedup on Ampere+ GPUs.
- **perf(uav_training/cache)**: Replaced hardcoded `cache=False` with dynamic RAM-based cache selection (`ram`/`disk`/`False`) using `psutil` in `uav_training/config.py`, targeting GPU utilization from ~65% to 90%+ on A100 Colab.
- **fix(uav_training/batch)**: Removed dead-code VRAM batch formula that silently overrode all tier-based batch values in `uav_training/config.py`; empirically tested tier batch sizes (A100=32, L4=32, T4=16) are now used directly.
- **fix(uav_training/amp_dtype)**: Removed invalid `amp_dtype: "bf16"` key from both `config_overrides` and `TRAIN_CONFIG` in `uav_training/config.py` and its forwarding logic in `uav_training/train.py`; Ultralytics silently ignored this parameter.
- **fix(uav_training/audit)**: Fixed `is_sample` dead code in `uav_training/audit.py` — sample/inference-only datasets now correctly receive `SKIP` status instead of being included in training.
- **fix(uav_training/audit)**: Fixed class count logic in `uav_training/audit.py` — `result[key] = 1` replaced with proper `+= 1` accumulator for accurate class match reporting.
- **log(uav_training/precision)**: Added BF16 hardware support verification and TF32 status to `auto_detect_hardware()` output in `uav_training/config.py`; cleaned up `amp_dtype` reference in `_log_precision_policy()` in `uav_training/train.py`.
- **release**: Bumped module/script version from `0.8.5` to `0.8.6`.

## 0.0.29 - 2026-02-24
- **fix(uav_training/config)**: Raised RAM cache threshold from 60GB to 100GB so A100 Colab (83.5GB RAM) correctly falls back to `cache="disk"` instead of silently skipping all caching; Ultralytics 1.5x safety margin made RAM caching impossible at 78.5GB available.
- **fix(uav_training/config)**: Removed deprecated `label_smoothing` from `config_overrides` and `TRAIN_CONFIG` dicts.
- **fix(uav_training/train)**: Removed `label_smoothing` from `optional_params` forwarding list and deleted the stale `smoothing` -> `label_smoothing` migration shim.
- **release**: Bumped module/script version from `0.8.7` to `0.8.8`.

## 0.0.29 - 2026-02-24
- **docs(root)**: Created `statik_denetim_raporu.md` — comprehensive Turkish static audit report covering both UAV and GPS training pipelines with 14 findings across 7 categories.
- **release**: Bumped module/script version from `0.8.8` to `0.8.9`.

## 0.0.29 - 2026-02-24
- **fix(gps_training/train)**: Replaced unconditional `import fcntl` with platform-aware import (`msvcrt` on Windows, `fcntl` on POSIX). Also updated `_acquire_file_lock` / `_release_file_lock` to use `msvcrt.locking` on Windows, fixing `ModuleNotFoundError` crash.
- **fix(gps_training/train)**: Fixed `OneCycleLR` resume incompatibility — scheduler now uses `epochs=remaining_epochs` instead of total epochs, so LR profile stays correct after resume even if batch size or data changed.
- **fix(gps_training/train)**: Added `GradScaler` for pre-Ampere GPUs (T4, V100). Auto-detects GPU capability: BF16 on Ampere+, FP16 + GradScaler on older hardware. Prevents silent gradient underflow/divergence.
- **fix(gps_training/train)**: Applied atomic write pattern (tmp + `os.replace`) to `best_model.pt` — same as `last_model.pt` — to prevent half-written files on crash.
- **fix(gps_training/train)**: Replaced bare `except` in `torch.compile` with `except Exception` and gated behind `sys.version_info < (3, 12)` to avoid swallowing meaningful errors and Dynamo incompatibility.
- **fix(gps_training/train)**: `_is_checkpoint_valid` now uses `weights_only=True` to avoid loading full tensors into RAM, with `gc.collect()` fallback for older PyTorch.
- **fix(uav_training/train)**: `_is_checkpoint_valid` now uses `weights_only=True` + explicit `gc.collect()` to reduce transient memory pressure during resume search (up to 3 checkpoint loads).
- **fix(uav_training/config)**: Aligned `TARGET_CLASSES` mapping with `build_dataset.py` / `dataset.yaml` output (0=vehicle, 1=human, 2=uap, 3=uai). Previous mapping was inverted and would cause wrong class assignments in any code referencing `TARGET_CLASSES`.
- **feat(gps_training/dataset)**: Added Siamese-aware data augmentation (horizontal flip with delta negation, color jitter, Gaussian blur) applied consistently to both frames during training. Reduces overfitting risk.
- **fix(uav_training/train)**: Hardened `checkpoint_guard` with `_LAST_SYNC_EPOCH` dedup guard to prevent duplicate sync dispatches within the same epoch. Drive sync now uses atomic write (tmp + replace) to avoid partially-written checkpoint files.
- **feat(.github/workflows/lint.yml)**: CI now installs full dependencies and runs `pytest tests/` alongside existing flake8 and compileall checks.
- **release**: Bumped module/script version from `0.8.9` to `0.8.10`.

## 0.0.30 - 2026-02-24
- **fix(uav_training/train)**: Fixed `_is_checkpoint_valid` regression — `weights_only=True` rejects YOLO checkpoints containing `DetectionModel` custom class, falsely marking valid checkpoints as corrupt. Now uses explicit `weights_only=False` with `gc.collect()` cleanup.
- **fix(gps_training/train)**: Same `_is_checkpoint_valid` fix for GPS module.
- **fix(gps_training/train)**: Platform-aware `fcntl`/`msvcrt` import and file locking for Windows compatibility.
- **fix(gps_training/train)**: `OneCycleLR` now uses `epochs=remaining_epochs` for correct resume LR profile.
- **fix(gps_training/train)**: Added `GradScaler` for pre-Ampere GPUs (FP16 + scaler on T4/V100, BF16 on Ampere+).
- **fix(gps_training/train)**: Atomic write pattern for `best_model.pt` (tmp + `os.replace`).
- **fix(gps_training/train)**: `torch.compile` gated behind Python <3.12 with proper `except Exception`.
- **fix(uav_training/config)**: Aligned `TARGET_CLASSES` with `dataset.yaml` (0=vehicle, 1=human, 2=uap, 3=uai).
- **feat(gps_training/dataset)**: Siamese-aware augmentation (horizontal flip, color jitter, Gaussian blur).
- **fix(uav_training/train)**: `checkpoint_guard` dedup via `_LAST_SYNC_EPOCH`; atomic Drive sync writes.
- **feat(.github/workflows/lint.yml)**: CI now runs `pytest tests/`.
- **release**: Bumped module/script version from `0.8.11` to `0.8.12`.

## 0.0.29 - 2026-02-24
- **docs(uav_training/build_dataset)**: Added inline comments to MAPPINGS explaining UAI-/UAP- semantics (unsuitable landing areas) and why they are merged with their suitable counterparts.
- **docs(documentation/datasets.md)**: Created dataset reference documenting all 4 source datasets, their original class names, UAI-/UAP- meanings, and the unified 4-class target mapping.
- **release**: Bumped module/script version from `0.8.10` to `0.8.11`.

## 0.0.28 - 2026-02-24
- **perf(uav_training/config)**: Reduced A100-40GB batch from 32 to 28 at 1024px to provide ~6GB VRAM headroom for TaskAlignedAssigner dynamic allocations, eliminating repeated OOM CPU fallbacks.
- **perf(uav_training/train)**: Added `max_split_size_mb:512` to `PYTORCH_CUDA_ALLOC_CONF` to reduce VRAM fragmentation.
- **release**: Bumped module/script version from `0.8.6` to `0.8.7`.

## 0.0.27 - 2026-02-24
- **fix(uav_training/train)**: Removed BF16 monkey patch (`FORCE_BF16_PATCH`) that broke Ultralytics AMP validation check, causing AMP to be silently disabled and forcing FP32 (~2x VRAM), which cascaded into CUDA OOM on A100-40GB at batch=32/1024px. Native AMP now handles BF16 on Ampere+ GPUs automatically.
- **fix(scripts/colab_bootstrap)**: Replaced `FORCE_BF16_PATCH=1` env var with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce VRAM fragmentation on Ampere+ GPUs.
- **fix(uav_training/config)**: Gated `torch.compile` (`reduce-overhead`) behind `sys.version_info < (3, 12)` check since Dynamo is unsupported on Python 3.12+ with torch 2.x; prevents wasting an OOM recovery attempt on a known failure.
- **fix(uav_training/train)**: OOM fallback now also sets `nbs=batch` when halving batch size, preventing Ultralytics from silently halving the learning rate via `lr = lr0 * batch/nbs`.
- **perf(uav_training/train)**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as fallback in `train.py` for non-bootstrap launches.
- **perf(uav_training/train)**: Improved `kill_gpu_hogs()` with `torch.cuda.synchronize()` before `empty_cache()` and added `del model` in OOM retry loop for more reliable VRAM cleanup between attempts.

## 0.0.26 - 2026-02-24
- **fix(uav_training/config)**: Renamed deprecated `smoothing` key to `label_smoothing` in both auto-detect and fallback config dicts in `uav_training/config.py` to fix Ultralytics 8.3+ `SyntaxError: 'smoothing' is not a valid YOLO argument`.
- **fix(uav_training/train)**: Removed stale `smoothing` from `optional_params` forwarding list and reversed the backward-compat shim to migrate legacy `smoothing` -> `label_smoothing` instead of dropping the valid key.
- **chore(deps)**: Relaxed `requirements.txt` upper-bound constraints (`<8.4.0`, `<2.3.0`, `<0.18.0`, `<2.0.0`) to minimum-only (`>=`) to prevent Colab pre-installed package conflicts.
- **fix(scripts/colab_bootstrap)**: Added `/content/repo` fallback in version reader so the initial banner shows the actual version instead of `vdev` when the repo was already cloned from a previous Colab run.

## 0.0.31 - 2026-02-24
- **fix(scripts/colab_bootstrap)**: Disabled periodic runs sync by default (`UAV_SYNC_INTERVAL_SEC=0`) so minute-level "Periodic checkpoint sync completed" log spam stops unless explicitly re-enabled.
- **fix(uav_training/config)**: Changed `save_period` default from `5` to `1` for epoch-by-epoch checkpoint sync.
- **fix(uav_training/train)**: Updated `checkpoint_guard` to 1-based epoch indexing and removed extra debug-file writes to reduce I/O overhead during training.

## 0.0.32 - 2026-02-24
- **docs(statik_denetim_raporu)**: Added comprehensive static code audit report covering both UAV and GPS training pipelines with 7 mandatory sections: findings summary, critical risks, performance evaluation, training stability analysis, MLOps maturity assessment, uncertainties, and health score (6.7/10).
- **release**: Bumped module/script version from `0.8.12` to `0.8.13`.

## 0.0.33 - 2026-02-24
- **docs(README)**: Fixed version references `v0.8.6` → `v0.8.13` in title and repo structure.
- **docs(README)**: Corrected A100 batch size `32` → `28` to match actual `config.py` value.
- **docs(README)**: Fixed cache thresholds from `>60GB: ram` to `>100GB: ram, >20GB: disk` per current `auto_detect_hardware()`.
- **docs(README)**: Updated save period from `Her 5 epoch` to `Her 1 epoch` to reflect `save_period=1` change.
- **docs(README)**: Corrected megaset smart sampling vehicle keep rate from `10%` to `30%` per `DEFAULT_CLASS_KEEP_PROB`.
- **docs(README)**: Fixed thread limiting description to match dynamic OMP/OpenBLAS/MKL values in `config.py`.
- **docs(README)**: Added Python `<3.12` gate to `torch.compile` description.
- **docs(README)**: Added missing `tests/`, `documentation/`, `CHANGELOGS.md` to repository structure.
- **docs(README)**: Added ImgSz column to Auto Hardware Detection table.

## 0.0.34 - 2026-02-24
- **fix(gps_training/dataset)**: `__getitem__` now returns `None` on error instead of raising `RuntimeError`, enabling `collate_drop_none` to gracefully skip corrupt samples instead of crashing training (KR-01).
- **fix(gps_training/config)**: Added `max_lr: 1e-3` to `TRAIN_CONFIG` so `OneCycleLR` performs its warmup/anneal cycle instead of behaving as constant LR (KR-03).
- **fix(gps_training/train)**: Removed stale `scheduler_state_dict` restoration on resume; `OneCycleLR` is now always created fresh for remaining epochs to avoid `total_steps` mismatch (KR-02).
- **fix(uav_training/build_dataset)**: Added standard `nc: 4` key to generated `dataset.yaml` and removed non-standard `values` key for Ultralytics compatibility (KR-04).
- **fix(deps)**: Added major-version upper bounds to `requirements.txt` (`ultralytics<9`, `torch<3`, `torchvision<1`, `numpy<3`) and minimum versions for previously unpinned packages (KR-05).
- **perf(gps_training/train)**: Added `persistent_workers` and `prefetch_factor=4` to `val_loader`, eliminating per-epoch worker respawn overhead (PD-03).
- **perf(gps_training/train)**: Updated ONNX export `opset_version` from 11 to 17 for modern operator fusion support (PD-05).
- **perf(gps_training/config)**: Reduced default `frame_cache_size` from 256 to 128, halving per-worker memory footprint in multi-worker DataLoader (PD-01).
- **chore(uav_training/train)**: Removed all 4 `#region agent log` debug blocks that wrote to `debug-4e729f.log`, eliminating production I/O overhead and code clutter (PD-08).
- **feat(gps_training/train)**: Added early stopping with configurable `patience` (default 20 epochs) to prevent overfitting and GPU waste (ES-01).
- **feat(gps_training/model)**: Added `LayerNorm(1024)` after backbone feature concatenation in `SiameseTracker` to stabilize regressor input distribution (ES-02).
- **fix(uav_training/train)**: Wrapped `torch.cuda.manual_seed` calls in `setup_seed` with `torch.cuda.is_available()` guard, consistent with GPS module (ES-03).
- **fix(gps_training/train)**: Added `torch.isfinite(output)` check before loss computation to catch NaN model outputs early under AMP (ES-07).
- **feat(gps_training/__init__)**: Added `__version__ = "0.8.14"` to GPS module for version tracking parity with UAV (MO-05).
- **fix(gps_training/train)**: Replaced `subprocess.run(shell=True)` rsync call with explicit arg list to eliminate shell injection risk (MO-07).
- **fix(uav_training/train)**: Resume path now auto-triggers `build_dataset()` when `dataset.yaml` is missing, preventing hard-fail (`Error: Dataset config not found at /content/dataset_built/dataset.yaml`) after Colab runtime resets.
- **release**: Bumped module/script version from `0.8.13` to `0.8.14`.

## 0.0.35 - 2026-02-24
- **feat(gps_training/train)**: Added CUDA OOM recovery with automatic batch halving (max 2 retries), DataLoader rebuild, and scheduler reset — prevents training crashes on memory spikes (PD-02).
- **fix(gps_training/train)**: Added explicit `weights_only=False` to resume `torch.load` calls to suppress `FutureWarning` on PyTorch 2.2-2.5 and ensure consistent behavior across versions (MO-02).
- **feat(gps_training/train)**: Training config is now persisted to `train_config.json` in artifacts directory at startup for post-hoc experiment comparison (MO-06).
- **perf(scripts/colab_bootstrap)**: Added fast-path dataset readiness check (`/content/datasets_local/dataset.yaml` + file count) to skip extraction tooling, Drive→SSD copy, and tar extraction when local SSD dataset is already prepared in the current runtime.
- **release**: Bumped module/script version from `0.8.14` to `0.8.15`.

## 0.0.36 - 2026-02-24
- **docs(statik_denetim_raporu)**: Güncellenmiş statik kod denetim raporu; önceki denetimde tespit edilen kritik risklerin giderildiği, kalan orta/yüksek risklerin ve genel sağlık skorunun (7.5/10) belgelenmesi.
- **release**: Bumped module/script version from `0.8.15` to `0.8.16`.

## 0.0.37 - 2026-02-24
- **chore(archive)**: GPS training modülü `archive_gps/` klasörüne taşındı; UAV eğitim hattı tamamlanana kadar kullanılmayacak. Taşınan: `gps_training/`, `tests/test_gps_config.py`, `tests/test_gps_train_scheduler.py`.
- **docs(README)**: GPS bölümü kaldırıldı, repository yapısı güncellendi.
- **chore(cleanup.sh)**: GPS train.py pkill satırı kaldırıldı.
- **release**: Bumped module/script version from `0.8.16` to `0.8.17`.

## 0.0.38 - 2026-02-24
- **chore(remove)**: GPS ile ilgili tüm dosyalar silindi (`archive_gps/` klasörü). Yerel yedek alındı, artık projede GPS modülü yok.
- **docs(README)**: archive_gps referansı kaldırıldı.
- **release**: Bumped module/script version from `0.8.17` to `0.8.18`.

## 0.0.39 - 2026-02-24
- **fix(uav_training/train)**: PYTORCH_CUDA_ALLOC_CONF artık CUDA erişiminden önce ayarlanıyor (KR-5).
- **fix(uav_training/config)**: TPU + VRAM > 0 senaryosunda NameError önlendi; `elif is_tpu:` dalı eklendi (KR-1).
- **fix(uav_training/train)**: `optional_params` listesine `fliplr`, `hsv_h`, `hsv_s`, `hsv_v` eklendi (KR-2).
- **fix(uav_training/train)**: Phase2 `phase_overrides` içinde `nbs` güncelleniyor (KR-4).
- **fix(uav_training/config)**: Config drift giderildi; `phase2_close_mosaic` ve `phase2_lr0` donanım profillerinden kaldırıldı (ES-1, ES-2).
- **fix(uav_training)**: `DATASETS_TRAIN_DIR` eklendi; audit ve build_dataset aynı dizini kullanıyor (KR-3, MLO-1).
- **feat(uav_training/train)**: Resume öncesi `_resume_preflight_check()` ile dataset path ve split doğrulaması.
- **fix(uav_training/train)**: Son epoch'ta checkpoint sync senkron yapılıyor; daemon thread kesintisi riski azaltıldı (P-3).
- **feat(uav_training/train)**: Leakage denetimi yaptırıma bağlandı; `--allow-leakage` ile override (P1).
- **fix(uav_training/audit)**: YOLO_FLAT formatı için `split_overlap` alanı eklendi (ES-6).
- **chore(uav_training/audit)**: `OUTPUT_REPORT` dead code kaldırıldı (MLO-2).
- **refactor(uav_training)**: `__version__` tek kaynak olarak `__init__.py`'da (MLO-4).
- **refactor(uav_training)**: TF32 ayarları `setup_torch_backend()` ile merkezileştirildi (MLO-5).
- **perf(uav_training/build_dataset)**: Cross-device hardlink denemesi `st_dev` ile atlanıyor (P-1).
- **fix(uav_training/build_dataset)**: Megaset sahne bölümlemesi deterministik; `set_seed(42)` shuffle öncesi (P-4).
- **fix(uav_training/train)**: OOM kurtarma zincirinde `imgsz > 640` ise `imgsz=640` (P-5).
- **docs(uav_training)**: `DEFAULT_CLASS_KEEP_PROB` ile dokümantasyon uyumu; %30 (ES-3).
- **chore(requirements)**: `requirements-dev.txt` oluşturuldu; `pytest` oraya taşındı (MLO-3).
- **docs(datasets.md)**: Colab yolu ve `DATASETS_TRAIN_DIR` açıklaması güncellendi (MLO-6).
- **feat(uav_training/build_dataset)**: Dışlanan etiket satırları (short_format, invalid_coords) raporlanıyor.
- **fix(uav_training/inference)**: CLI ve fonksiyon varsayılan path tutarlı; `DEFAULT_INFER_SOURCE`.
- **release**: Bumped module/script version from `0.8.18` to `0.8.19`.

## 0.0.40 - 2026-02-24
- **refactor(uav_training/config)**: Import side effect kaldırıldı; `auto_detect_hardware()` lazy `ensure_colab_config()` ile sadece train başında çağrılıyor (sistem_denetim_raporu §2.1).
- **feat(uav_training/visualize_dataset)**: `--split train|val|test` ve `--num` CLI argümanları eklendi; val/test split görselleştirmesi destekleniyor (sistem_denetim_raporu §6.5).
- **release**: Bumped module/script version from `0.8.19` to `0.8.20`.

## 0.0.41 - 2026-02-24
- **fix(uav_training/build_dataset)**: Megaset label çözümleme hatası düzeltildi; `_execute_megaset_process` içinde `split` parametresi ile `src_split_path` doğru tayin ediliyor (KRTK-01).
- **fix(uav_training/build_dataset)**: `MIN_BBOX_NORM` ve `INCLUDE_TEST_IN_VAL` artık modül seviyesi sabit yerine `build_dataset()` çağrıldığında dinamik okunuyor; Colab config senkronizasyon riski giderildi (KRTK-02).
- **fix(uav_training/train)**: Leakage denetimi artık built dataset (`DATASET_DIR`) üzerinde de yapılıyor; audit raporu yanı sıra gerçek split overlap kontrolü (KRTK-03).
- **docs(UAV_Pipeline_Audit_Report)**: Statik kod denetim raporu eklendi; sağlık skoru 9.0/10.
- **release**: Bumped module/script version from `0.8.20` to `0.8.21`.

## 0.0.42 - 2026-02-24
- **chore(requirements)**: ultralytics/torch/torchvision sabit versiyonlara çekildi (8.3.0, 2.5.1, 0.20.1); Colab reprodüksiyon.
- **feat(colab_bootstrap)**: `TWO_PHASE_TRAINING` ve `FORCE_FRESH_START` bayrakları eklendi (M-01, B-05).
- **fix(uav_training/build_dataset)**: KR-2 — `target_id` sadece tüm bbox geçerlilik kontrolleri sonrası `present_target_ids`'e ekleniyor.
- **fix(uav_training/build_dataset)**: Lock release'de `os.close(fd)` OSError koruması.
- **fix(uav_training/build_dataset)**: `data.yaml` path `'.'` olarak ayarlandı; test yoksa `val/images` fallback.
- **fix(uav_training/config)**: H100 batch 32'ye düşürüldü (P-04); workers `cpus*2` formülü; phase2_lr0 düşürüldü.
- **fix(uav_training/train)**: M-02 sync lock ile race condition önlendi; M-03 checkpoint zipfile ile doğrulama.
- **feat(uav_training/train)**: M-05 full_attempt_args.yaml kaydı; E-02/B-04 phase2 batch phase1'den alınıyor.
- **feat(uav_training/train)**: M-06/E-05 audit raporu yoksa audit.py otomatik çalıştırılıyor.
- **release**: Bumped module/script version from `0.8.21` to `0.8.22`.

## 0.0.43 - 2026-02-25
- **fix(uav_training/train)**: Ultralytics `compile` arg — only pass when version >= 8.3.196; avoids `SyntaxError: 'compile' is not a valid YOLO argument` on 8.3.0.
- **chore(requirements)**: ultralytics pin updated from `8.3.0` to `>=8.3.196` for torch.compile support.
- **feat(uav_training/config)**: Run name now versioned — `uav_v3_optimized_v{version}` from `__init__.py`; runs no longer overwrite across versions.
- **feat(colab_bootstrap)**: Log filename now versioned — `log_v{VERSION}_%Y-%m-%d_%H-%M.txt` for easier log identification.
- **release**: Bumped module/script version from `0.8.22` to `0.8.23`.

## 0.0.44 - 2026-02-25
- **fix(uav_training/train)**: Resume across Ultralytics version upgrade (8.3→8.4) — on optimizer state mismatch (`parameter group` ValueError), retry with `resume=False` to load model weights only and restart optimizer.
- **release**: Bumped module/script version from `0.8.23` to `0.8.24`.

## 0.0.45 - 2026-02-25
- **perf(colab_bootstrap)**: Parallel Drive→SSD download — 8 workers copy disjoint byte ranges concurrently; tunable via `UAV_DOWNLOAD_WORKERS` env var.
- **release**: Bumped module/script version from `0.8.24` to `0.8.25`.

## 0.0.46 - 2026-02-25
- **perf(colab_bootstrap)**: Download optimization — progress in separate thread (no lock contention); 8MB chunks; default 4 workers to avoid Drive FUSE throttling; workers only hold lock for counter update.
- **fix(colab_bootstrap)**: Banner moved to after repo sync — `UAV Training Bootstrap v{VERSION}` now always shows version from pulled repo.
- **docs(.cursor/rules)**: Added bootstrap version reminder to changelogs rule.
- **release**: Bumped module/script version from `0.8.25` to `0.8.26`.

## 0.0.47 - 2026-02-25
- **perf(colab_bootstrap)**: Extraction optimization — 64MB pipe buffer (stdbuf), eatmydata (skip fsync), checkpoint 50k; install eatmydata for faster tar writes.
- **release**: Bumped module/script version from `0.8.26` to `0.8.27`.

## 0.0.48 - 2026-02-25
- **fix(uav_training/build_dataset)**: Made sampling deterministic with `set_seed(42)` for non-megaset datasets (R-01).
- **fix(uav_training/train)**: Synced `nbs` updates with batch size dynamically during OOM recovery to preserve LR scaling (R-02).
- **refactor(uav_training/audit)**: Updated `is_sample` keyword detection logic strictly via content scan instead of filename constraints (R-04).
- **perf(scripts/colab_bootstrap)**: Limited periodic rsync exclusively to the `weights/` subdirectory to reduce unnecessary I/O latency (P-01).
- **feat(uav_training/train)**: Explicitly inherited `lrf` configuration inside `phase2_overrides` instead of dropping it entirely (S-01).
- **feat(uav_training/train)**: Exported `git_commit` and `audit_md5` to `full_attempt_args.yaml` for complete tracking and experiment integrity (M-01).
- **fix(uav_training/config)**: Moved `ARTIFACTS_DIR.mkdir()` inside `ensure_colab_config()` to avoid breaking simple test-level imports via side-effect creation (M-02).
- **release**: Bumped module/script version from `0.8.27` to `0.8.28`.

## 0.0.49 - 2026-02-25
- **fix(scripts/colab_bootstrap)**: Mitigated shell injection — all `_run()` interpolations now use `shlex.quote()` for paths and env vars (KR-2).
- **fix(scripts/colab_bootstrap)**: Parallel download uses `os.pwrite()` with per-worker fd to avoid seek/write race (KR-3).
- **fix(scripts/colab_bootstrap)**: Fast-path dataset skip now requires `CACHE_MARKER` file; partial extraction no longer falsely triggers skip (KR-5).
- **fix(scripts/colab_bootstrap)**: Periodic sync default changed from `UAV_SYNC_INTERVAL_SEC=0` to `300` to reduce checkpoint loss on Colab disconnect (KR-1).
- **fix(uav_training/build_dataset)**: `resolve_target_split` unknown split now raises `ValueError` instead of silently mapping to `val` (R-04).
- **fix(uav_training/train)**: Resume checkpoint search filters by version (`v{__version__}`) to avoid loading incompatible checkpoints.
- **fix(scripts/colab_bootstrap)**: `find_latest_checkpoint` filters by version for consistent resume across version bumps.
- **docs(system_check)**: Restructured static audit prompt — performance/quality focus, mandatory headings, evidence-based findings.
- **docs(reports)**: Replaced `ml_pipeline_denetim_raporu.md` with `UAV_Pipeline_Statik_Denetim_Raporu.md`.
- **docs(README)**: Version references updated to v0.8.28.
- **release**: Bumped module/script version from `0.8.28` to `0.8.29`.

## 0.0.50 - 2026-02-25
- **feat(uav_training)**: UAV Training Optimization (uav_training_optimizasyon.md) — Phase 2 config (phase2_mosaic=0, close_mosaic=0, lrf=0.1, copy_paste=0.5, degrees=10, scale=0.6, hsv_s/hsv_v), min_bbox_norm=0.002.
- **feat(uav_training/val_utils)**: New module — run_per_class_val, print_per_class_report, check_temporal_leakage.
- **feat(uav_training/train)**: Automatic per-class validation after Phase 1; best_map50.pt callback; checkpoint cleanup (keep best, last, last 3 epoch); Drive sync includes best_map50.pt.
- **feat(scripts/run_per_class_val)**: Per-class AP50 script (vehicle, human, uap, uai) with target thresholds.
- **feat(scripts/cleanup_checkpoints)**: Standalone checkpoint cleanup script.
- **fix(uav_training/audit)**: Megaset audit — fallback for split_path when train/images/ missing (images/labels in split root).
- **fix(uav_training/build_dataset)**: Automatic temporal leakage check after build.
- **fix(requirements.txt)**: ultralytics==8.4.16 (version pinned).
- **fix(scripts)**: Colab paths — run_per_class_val and cleanup_checkpoints prefer /content/runs on Colab.
- **release**: Bumped module/script version from `0.8.29` to `0.8.30`.

## 0.8.31 - 2026-02-25
- **fix(scripts/run_per_class_val)**: Local runs now use `TRAIN_CONFIG["project"]` instead of `PROJECT_ROOT/runs` so models are found after local training.
- **fix(uav_training/build_dataset)**: `min_bbox_norm` fallback aligned to 0.002 (config consistency).
- **fix(docs)**: README and uav_training/README `min_bbox_norm` updated to 0.002.
- **feat(uav_training/config)**: Added `IMAGE_EXTENSIONS` for webp, bmp, tiff, tif, gif support.
- **feat(uav_training)**: build_dataset, audit, val_utils, inference, visualize_dataset now support all IMAGE_EXTENSIONS (webp, bmp, tiff, gif).
- **fix(uav_training/train)**: Corrected raise indentation in phase2 fallback block.
- **fix(uav_training/val_utils)**: `check_temporal_leakage` accepts str or Path; defensive `Path(dataset_dir)`.
- **release**: Bumped module/script version from `0.8.30` to `0.8.31`.

## 0.8.32 - 2026-02-25
- **fix(uav_training/build_dataset)**: Megaset unique_name now uses `target_split` instead of `None` when split is synthetic (fixes `megaset_None_copy0_*.jpg` filenames).
- **fix(uav_training/build_dataset)**: Split display in logs uses `target_split` when split is None (megaset scene-based splits).
- **release**: Bumped module/script version from `0.8.31` to `0.8.32`.

## 0.8.33 - 2026-02-25
- **feat(uav_training/build_dataset)**: Post-build cleanup: remove orphan images (no label) and orphan labels (no image).
- **feat(uav_training/build_dataset)**: Post-build train/val duplicate removal by content hash — removes from val any image identical to train (prevents data leakage).
- **feat(uav_training/config)**: `remove_orphans` and `remove_train_val_duplicates` config flags (default True).
- **release**: Bumped module/script version from `0.8.32` to `0.8.33`.

## 0.8.34 - 2026-02-25
- **test**: Added `tests/conftest.py` for cross-platform sys.path setup; fixed imports to `from uav_training.config import` for package compatibility.
- **test**: New tests: `test_config`, `test_val_utils`, `test_cleanup_checkpoints`, `test_inference`; cross-platform fixes (tmp_path, Path.is_relative_to, Colab skip).
- **feat(scripts)**: `run_all_tests.py` and `colab_smoke_test.py`; `scripts/__init__.py` for package.
- **feat(scripts/colab_bootstrap)**: Full pytest runs automatically before dataset download (check=True).
- **feat(.pre-commit-config.yaml)**: `pre-commit install` runs tests before each commit; `scripts/setup_hooks.py` for one-time setup.
- **feat(.github/workflows/lint.yml)**: CI now runs on all branches (push/PR).
- **chore(requirements.txt)**: Added pytest>=7.0.0; `requirements-dev.txt`: flake8, pre-commit.
- **docs(README)**: "Otomatik Testler" section — Colab, CI, local pre-commit.
- **release**: Bumped module/script version from `0.8.33` to `0.8.34`.

## 0.8.35 - 2026-02-25
- **chore(scripts)**: Removed `cleanup.sh` — Linux-only bash script, Windows-incompatible; checkpoint cleanup via `cleanup_checkpoints.py` only.
- **chore(tests)**: Removed `test_audit.py` — redundant with `test_uav_audit.py` (more comprehensive coverage).
- **docs(README)**: Removed cleanup.sh references; updated uav_training/README version.
- **release**: Bumped module/script version from `0.8.34` to `0.8.35`.

## 0.8.36 - 2026-02-25
- **feat(.flake8)**: Added exclude for .git, __pycache__, venv, .venv, artifacts, datasets; max-line-length 120.
- **perf(CI)**: Compile job no longer installs requirements.txt — syntax-only check, faster CI.
- **docs(CHANGELOGS)**: Version format note — UAV uses 0.8.XX; new entries follow this format.
- **release**: Bumped module/script version from `0.8.35` to `0.8.36`.

## 0.8.37 - 2026-02-25
- **feat(uav_training/config)**: High RAM vs Normal RAM detection — Colab A100 (~167GB vs ~80GB).
- **feat(uav_training/config)**: A100 80GB tier (separate from H100); High RAM: cache=ram, workers up to 12.
- **docs(README)**: RAM detection and A100 80GB tier documentation.
- **release**: Bumped module/script version from `0.8.36` to `0.8.37`.

## 0.8.38 - 2026-02-25
- **perf(colab_bootstrap)**: Drive download — pv sequential 128MB buffer (default), parallel fallback 2 workers/32MB chunks; Drive FUSE throttles parallel reads.
- **perf(colab_bootstrap)**: Drive mount — UAV_DRIVE_FORCE_REMOUNT=1 for stale connection fix.
- **perf(colab_bootstrap)**: Extraction — 128MB pipe buffer, tar -b 10240, checkpoint 100k; UAV_EXTRACT_PIPE_BUF, UAV_PV_BUFFER env vars.
- **release**: Bumped module/script version from `0.8.37` to `0.8.38`.

## 0.8.39 - 2026-02-25
- **fix(colab_bootstrap)**: pv progress output — use decode+write instead of sys.stdout.buffer (Colab OutStream has no .buffer).
- **release**: Bumped module/script version from `0.8.38` to `0.8.39`.

## 0.8.40 - 2026-02-25
- **fix(colab_bootstrap)**: _write_stdout_bytes helper — hasattr check for .buffer, works in Colab and regular Python.
- **release**: Bumped module/script version from `0.8.39` to `0.8.40`.

## 0.8.41 - 2026-02-25
- **fix(colab_bootstrap)**: pv progress — file-size polling thread (pv stderr invisible in Colab); shows % GB MB/s ETA.
- **feat(notebook)**: Clone repo first, run bootstrap from repo — always latest, no wget cache.
- **chore(colab_bootstrap)**: Removed unused _write_stdout_bytes.
- **release**: Bumped module/script version from `0.8.40` to `0.8.41`.

## 0.8.42 - 2026-02-25
- **perf(colab_bootstrap)**: Parallel download default — 4 workers (UAV_DOWNLOAD_METHOD=parallel); set pv for sequential.
- **release**: Bumped module/script version from `0.8.41` to `0.8.42`.

## 0.8.43 - 2026-02-25
- **fix(colab_bootstrap)**: Cleanup output — 🧹→• (avoid red in Colab), print_output=False for pkill/find/rm.
- **fix(notebook)**: git pull 2>&1 — merge stderr to stdout so output is not red.
- **perf(colab_bootstrap)**: UAV_DOWNLOAD_WORKERS default 4→8.
- **release**: Bumped module/script version from `0.8.42` to `0.8.43`.

## 0.8.44 - 2026-02-25
- **refactor(notebook)**: wget bootstrap only — no git in notebook; bootstrap clones repo inside.
- **fix(colab_bootstrap)**: git fetch/reset/clone/log print_output=False — no red stderr in Colab.
- **release**: Bumped module/script version from `0.8.43` to `0.8.44`.
