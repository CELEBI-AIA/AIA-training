#!/usr/bin/env python3
"""
Per-class validation for UAV model - vehicle, human, uap, uai AP50.

Run after Phase 1 to check per-class metrics. Otomatik olarak two-phase
training sirasinda da calistirilir.

Usage:
  python scripts/run_per_class_val.py [model_path] [dataset_path]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from uav_training.config import DATASET_DIR  # noqa: E402
from uav_training.val_utils import run_per_class_val, print_per_class_report  # noqa: E402

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-class validation for UAV model")
    parser.add_argument("model_path", nargs="?", default=None, help="Path to best.pt")
    parser.add_argument("data_path", nargs="?", default=str(DATASET_DIR), help="Path to dataset")
    parser.add_argument("--split", default="val", help="Split to validate")
    args = parser.parse_args()

    if not args.model_path:
        # Colab: egitim /content/runs'ta; local: TRAIN_CONFIG["project"] (artifacts/uav_model/training_results)
        from uav_training.config import is_colab, TRAIN_CONFIG
        runs = Path("/content/runs") if is_colab() else Path(TRAIN_CONFIG.get("project", "."))
        if not runs.exists() and runs != Path("/content/runs"):
            runs = Path("/content/runs")
        if runs.exists():
            candidates = list(runs.rglob("phase1/weights/best.pt"))
            if not candidates:
                candidates = list(runs.rglob("weights/best.pt"))
            if candidates:
                args.model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        if not args.model_path:
            print("Error: model_path required. No phase1/weights/best.pt found.")
            sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)

    print(f"\nPer-class validation: {args.model_path}")
    print(f"Dataset: {args.data_path}\n")

    result = run_per_class_val(args.model_path, args.data_path, args.split, verbose=True)
    print_per_class_report(result)
    return result

if __name__ == "__main__":
    main()

