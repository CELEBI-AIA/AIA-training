# CHANGELOGS

## 0.0.01 - 2026-02-21
- Fixed resume robustness in `gps_training/train.py` to avoid KeyError when `scheduler_state_dict` is missing in older/incomplete checkpoints.
- Added guarded loading for both optimizer and scheduler states with warning-based fallback.
