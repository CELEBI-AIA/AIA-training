#!/usr/bin/env python3
"""
Colab smoke test — quick import and logic checks after clone + pip install.
No GPU, Drive, or dataset required. Run before training to catch build issues.

Usage (Colab):
  !python scripts/colab_smoke_test.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def smoke_test():
    """Run minimal import and logic checks."""
    errors = []

    # 1. Import checks
    try:
        from uav_training.config import TRAIN_CONFIG, IMAGE_EXTENSIONS, is_colab
    except Exception as e:
        errors.append(f"uav_training.config: {e}")
        return errors

    try:
        from uav_training.build_dataset import resolve_target_split
    except Exception as e:
        errors.append(f"uav_training.build_dataset: {e}")
        return errors

    try:
        from uav_training.audit import read_yaml, read_txt_classes
    except Exception as e:
        errors.append(f"uav_training.audit: {e}")
        return errors

    try:
        from uav_training.val_utils import check_temporal_leakage, TARGET_THRESHOLDS
    except Exception as e:
        errors.append(f"uav_training.val_utils: {e}")
        return errors

    # 2. Quick logic checks
    if resolve_target_split("train", include_test_in_val=False) != "train":
        errors.append("resolve_target_split(train) != train")
    if resolve_target_split("test", include_test_in_val=True) != "val":
        errors.append("resolve_target_split(test, include_test_in_val=True) != val")

    if "epochs" not in TRAIN_CONFIG:
        errors.append("TRAIN_CONFIG missing 'epochs'")
    if ".jpg" not in IMAGE_EXTENSIONS:
        errors.append("IMAGE_EXTENSIONS missing .jpg")

    return errors


def main():
    print("Colab smoke test - checking imports and logic...", flush=True)
    errors = smoke_test()
    if errors:
        for e in errors:
            print(f"  FAIL: {e}", flush=True)
        print("Smoke test FAILED.", flush=True)
        sys.exit(1)
    print("  OK: All imports and logic checks passed.", flush=True)
    print("Smoke test PASSED.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
