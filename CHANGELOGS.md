# CHANGELOGS

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
