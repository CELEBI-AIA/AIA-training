#!/usr/bin/env python3
"""
Per-class validation for UAV model - vehicle, human, uap, uai AP50.

Run after Phase 1 to check per-class metrics. Otomatik olarak two-phase
training sirasinda da calistirilir.

Usage:
  python scripts/run_per_class_val.py [model_path] [dataset_path]
"""
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from uav_training.config import DATASET_DIR  # noqa: E402
from uav_training.val_utils import run_per_class_val, print_per_class_report  # noqa: E402

def _render_report(result: dict, model_path: str, data_path: str, split: str) -> str:
    lines = []
    lines.append("# Per-Class Validation Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Dataset: `{data_path}`")
    lines.append(f"- Split: `{split}`")
    lines.append("")
    lines.append("| Class | AP50 | AP50-95 |")
    lines.append("|---|---:|---:|")
    for cls, metrics in result.items():
        ap50 = float(metrics.get("ap50", 0.0)) if isinstance(metrics, dict) else float(metrics)
        ap50_95 = float(metrics.get("ap50_95", 0.0)) if isinstance(metrics, dict) else 0.0
        lines.append(f"| {cls} | {ap50:.4f} | {ap50_95:.4f} |")
    lines.append("")
    return "\n".join(lines)
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-class validation for UAV model")
    parser.add_argument("model_path", nargs="?", default=None, help="Path to best.pt")
    parser.add_argument("data_path", nargs="?", default=str(DATASET_DIR), help="Path to dataset")
    parser.add_argument("--split", default="val", help="Split to validate")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path (.md/.txt). If omitted, only prints to stdout.",
    )
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

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            _render_report(result, args.model_path, args.data_path, args.split),
            encoding="utf-8",
        )
        print(f"\nSaved report: {out_path}", flush=True)
    return result

if __name__ == "__main__":
    main()
