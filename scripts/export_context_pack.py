#!/usr/bin/env python3
"""
Export a review-ready context pack for dataset/model analysis.

This script is designed for Google Colab or local usage and produces:
- dataset_summary.md
- train_run_summary.md
- model/manifest.md
- copied run artifacts (results.csv/png, confusion matrix, PR curve, checkpoints)
- class samples and heuristic hard-case samples
- optional inference prediction samples on validation images

Usage example:
  python scripts/export_context_pack.py \
      --dataset-root /content/dataset_built \
      --output-dir /content/context_pack \
      --model /content/runs/detect/train/weights/best.pt \
      --run-dir /content/runs/detect/train \
      --gpu-name "A100 40GB" \
      --train-command "python train.py --two-phase --batch 24 --device 0"
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_training.emoji_logs import install_emoji_print  # noqa: E402

install_emoji_print(globals())

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    YOLO = None


CLASS_NAMES = {
    0: "vehicle",
    1: "human",
    2: "uap",
    3: "uai",
}

SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "test": "test",
}


@dataclass
class ImageRecord:
    split: str
    image_path: Path
    label_path: Optional[Path]
    class_ids: List[int]
    box_count: int
    is_negative: bool
    brightness_mean: Optional[float] = None
    blur_score: Optional[float] = None
    has_small_human: bool = False
    is_crowded: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model + dataset context pack")
    parser.add_argument("--dataset-root", required=True, help="Built dataset root containing train/val/test")
    parser.add_argument("--output-dir", required=True, help="Context pack output directory")
    parser.add_argument("--model", default=None, help="Path to best.pt")
    parser.add_argument("--run-dir", default=None, help="Training run dir containing results artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=25)
    parser.add_argument("--hard-case-count", type=int, default=20)
    parser.add_argument("--prediction-samples", type=int, default=20)
    parser.add_argument("--train-command", default="")
    parser.add_argument("--epochs", default="")
    parser.add_argument("--batch", default="")
    parser.add_argument("--imgsz", default="")
    parser.add_argument("--gpu-name", default="")
    parser.add_argument("--early-stopping", default="")
    parser.add_argument("--dataset-build-name", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def resolve_split_dirs(dataset_root: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    split_dirs: Dict[str, Dict[str, Optional[Path]]] = {}
    for split_name in ("train", "val", "valid", "test"):
        split_path = dataset_root / split_name
        if not split_path.exists():
            continue
        canonical = SPLIT_ALIASES[split_name]
        images_dir = split_path / "images" if (split_path / "images").exists() else split_path
        labels_dir = split_path / "labels" if (split_path / "labels").exists() else None
        split_dirs[canonical] = {"images": images_dir, "labels": labels_dir}
    return split_dirs


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    return sorted(p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in exts)


def parse_label_file(label_path: Path) -> Tuple[List[int], List[List[float]]]:
    class_ids: List[int] = []
    boxes: List[List[float]] = []
    if not label_path.exists():
        return class_ids, boxes

    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xywh = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        class_ids.append(cls_id)
        boxes.append(xywh)
    return class_ids, boxes


def compute_image_metrics(image_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if cv2 is None:
        return None, None
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, blur


def build_records(dataset_root: Path) -> Tuple[List[ImageRecord], dict]:
    split_dirs = resolve_split_dirs(dataset_root)
    if not split_dirs:
        raise FileNotFoundError(
            f"Could not find train/val/test directories under dataset root: {dataset_root}"
        )

    records: List[ImageRecord] = []
    split_stats = {
        split: {"images": 0, "negative_images": 0, "labels": 0}
        for split in ("train", "val", "test")
    }
    class_counts = {
        split: Counter() for split in ("train", "val", "test")
    }

    for split, paths in split_dirs.items():
        images_dir = paths["images"]
        labels_dir = paths["labels"]
        if images_dir is None or not images_dir.exists():
            continue
        for image_path in list_images(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt" if labels_dir else None
            class_ids, boxes = parse_label_file(label_path) if label_path else ([], [])
            brightness, blur = compute_image_metrics(image_path)

            has_small_human = any(
                cls_id == 1 and (xywh[2] * xywh[3]) <= 0.01
                for cls_id, xywh in zip(class_ids, boxes)
            )
            is_crowded = len(class_ids) >= 5
            is_negative = not class_ids

            record = ImageRecord(
                split=split,
                image_path=image_path,
                label_path=label_path if label_path and label_path.exists() else None,
                class_ids=class_ids,
                box_count=len(class_ids),
                is_negative=is_negative,
                brightness_mean=brightness,
                blur_score=blur,
                has_small_human=has_small_human,
                is_crowded=is_crowded,
            )
            records.append(record)

            split_stats[split]["images"] += 1
            split_stats[split]["labels"] += len(class_ids)
            if is_negative:
                split_stats[split]["negative_images"] += 1
            for cls_id in class_ids:
                if cls_id in CLASS_NAMES:
                    class_counts[split][CLASS_NAMES[cls_id]] += 1

    return records, {"split_stats": split_stats, "class_counts": class_counts}


def copy_records(records: Iterable[ImageRecord], out_dir: Path, limit: int) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for idx, record in enumerate(records):
        if idx >= limit:
            break
        dest = out_dir / record.image_path.name
        shutil.copy2(record.image_path, dest)
        copied.append(dest.name)
    return copied


def export_samples(records: List[ImageRecord], output_dir: Path, seed: int, samples_per_class: int, hard_case_count: int) -> dict:
    rng = random.Random(seed)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    class_buckets: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        seen = set()
        for cls_id in record.class_ids:
            if cls_id in CLASS_NAMES and cls_id not in seen:
                class_buckets[CLASS_NAMES[cls_id]].append(record)
                seen.add(cls_id)

    sample_index = {}
    for class_name, bucket in class_buckets.items():
        rng.shuffle(bucket)
        sample_index[class_name] = copy_records(
            bucket, samples_dir / class_name, samples_per_class
        )

    hard_case_index = {}
    small_humans = [r for r in records if r.has_small_human]
    crowded = [r for r in records if r.is_crowded]
    low_light = [r for r in records if r.brightness_mean is not None and r.brightness_mean < 60]
    blur = [r for r in records if r.blur_score is not None and r.blur_score < 80]
    negatives = [r for r in records if r.is_negative]

    heuristics = {
        "small_human": small_humans,
        "crowded": crowded,
        "low_light": low_light,
        "blur": blur,
        "negative_images": negatives,
    }
    for name, bucket in heuristics.items():
        rng.shuffle(bucket)
        hard_case_index[name] = copy_records(
            bucket, samples_dir / "hard_cases" / name, hard_case_count
        )

    return {
        "class_samples": sample_index,
        "hard_cases": hard_case_index,
    }


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def copy_if_exists(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def export_run_artifacts(run_dir: Optional[Path], model_path: Optional[Path], output_dir: Path) -> dict:
    model_dir = output_dir / "model"
    run_files_dir = output_dir / "run_files"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_files_dir.mkdir(parents=True, exist_ok=True)

    copied = {"model": [], "run_files": []}

    if model_path and model_path.exists():
        dest = model_dir / model_path.name
        shutil.copy2(model_path, dest)
        copied["model"].append(dest.name)

    if run_dir and run_dir.exists():
        artifact_names = [
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
        ]
        for name in artifact_names:
            if copy_if_exists(run_dir / name, run_files_dir / name):
                copied["run_files"].append(name)

        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            for name in ("best.pt", "last.pt"):
                if copy_if_exists(weights_dir / name, model_dir / name):
                    copied["model"].append(name)

    return copied


def write_dataset_summary(output_dir: Path, stats: dict, sample_index: dict, dataset_root: Path) -> None:
    total_images = sum(item["images"] for item in stats["split_stats"].values())
    total_labels = sum(item["labels"] for item in stats["split_stats"].values())
    total_negatives = sum(item["negative_images"] for item in stats["split_stats"].values())

    lines = [
        "# Dataset Summary",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Total images: **{total_images}**",
        f"- Total annotations: **{total_labels}**",
        f"- Negative images: **{total_negatives}**",
        "",
        "## Split Breakdown",
        "",
        "| Split | Images | Annotations | Negative Images |",
        "|-------|--------|-------------|-----------------|",
    ]

    for split in ("train", "val", "test"):
        item = stats["split_stats"][split]
        lines.append(
            f"| {split} | {item['images']} | {item['labels']} | {item['negative_images']} |"
        )

    lines.extend(
        [
            "",
            "## Class Counts",
            "",
            "| Split | vehicle | human | uap | uai |",
            "|-------|---------|-------|-----|-----|",
        ]
    )
    for split in ("train", "val", "test"):
        counts = stats["class_counts"][split]
        lines.append(
            f"| {split} | {counts.get('vehicle', 0)} | {counts.get('human', 0)} | "
            f"{counts.get('uap', 0)} | {counts.get('uai', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Semantics Note",
            "",
            "- `UAP` and `UAP-` are merged into the same target class: `uap`.",
            "- `UAI` and `UAI-` are merged into the same target class: `uai`.",
            "- Suitability is treated as a downstream landing-status concern, not as a separate detection class.",
            "",
            "## Exported Samples",
            "",
        ]
    )

    for class_name, files in sample_index["class_samples"].items():
        lines.append(f"- `{class_name}` sample count: {len(files)}")
    for bucket, files in sample_index["hard_cases"].items():
        lines.append(f"- hard case `{bucket}` count: {len(files)}")

    (output_dir / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_train_run_summary(output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Training Run Summary",
        "",
        f"- Repo commit: `{git_commit_hash()}`",
        f"- Dataset build name: `{args.dataset_build_name or 'not_provided'}`",
        f"- Train command: `{args.train_command or 'not_provided'}`",
        f"- Epochs: `{args.epochs or 'not_provided'}`",
        f"- Batch: `{args.batch or 'not_provided'}`",
        f"- ImgSz: `{args.imgsz or 'not_provided'}`",
        f"- GPU: `{args.gpu_name or 'not_provided'}`",
        f"- Early stopping: `{args.early_stopping or 'not_provided'}`",
        "",
        "## Notes",
        "",
        args.notes or "- No extra notes provided.",
        "",
        "## Fill Later If Needed",
        "",
        "1. Last 3-5 runs comparison",
        "2. What improved",
        "3. What regressed",
        "4. Known failure patterns",
    ]
    (output_dir / "train_run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_manifest(output_dir: Path, args: argparse.Namespace, copied_artifacts: dict) -> None:
    lines = [
        "# Model Manifest",
        "",
        f"- Repo commit: `{git_commit_hash()}`",
        f"- Dataset build: `{args.dataset_build_name or 'not_provided'}`",
        f"- Training command: `{args.train_command or 'not_provided'}`",
        f"- GPU: `{args.gpu_name or 'not_provided'}`",
        f"- Epochs: `{args.epochs or 'not_provided'}`",
        f"- Batch: `{args.batch or 'not_provided'}`",
        f"- ImgSz: `{args.imgsz or 'not_provided'}`",
        "",
        "## Copied Model Files",
        "",
    ]
    for name in copied_artifacts["model"] or ["none"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Copied Run Artifacts", ""])
    for name in copied_artifacts["run_files"] or ["none"]:
        lines.append(f"- `{name}`")
    (output_dir / "model" / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


def export_prediction_samples(dataset_root: Path, model_path: Optional[Path], output_dir: Path, prediction_samples: int, seed: int) -> bool:
    if YOLO is None or model_path is None or not model_path.exists():
        return False

    split_dirs = resolve_split_dirs(dataset_root)
    val_images_dir = split_dirs.get("val", {}).get("images")
    if val_images_dir is None or not val_images_dir.exists():
        return False

    images = list_images(val_images_dir)
    if not images:
        return False

    rng = random.Random(seed)
    chosen = images[:]
    rng.shuffle(chosen)
    chosen = chosen[:prediction_samples]

    model = YOLO(str(model_path))
    predictions_dir = output_dir / "samples" / "val_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model.predict(
        [str(p) for p in chosen],
        save=True,
        project=str(predictions_dir.parent),
        name=predictions_dir.name,
        exist_ok=True,
        verbose=False,
    )
    return True


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve() if args.model else None
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)

    records, stats = build_records(dataset_root)
    sample_index = export_samples(
        records=records,
        output_dir=output_dir,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        hard_case_count=args.hard_case_count,
    )
    copied_artifacts = export_run_artifacts(run_dir=run_dir, model_path=model_path, output_dir=output_dir)
    predictions_exported = export_prediction_samples(
        dataset_root=dataset_root,
        model_path=model_path,
        output_dir=output_dir,
        prediction_samples=args.prediction_samples,
        seed=args.seed,
    )

    write_dataset_summary(output_dir, stats, sample_index, dataset_root)
    write_train_run_summary(output_dir, args)
    write_model_manifest(output_dir, args, copied_artifacts)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "predictions_exported": predictions_exported,
        "copied_artifacts": copied_artifacts,
        "class_sample_counts": {k: len(v) for k, v in sample_index["class_samples"].items()},
        "hard_case_counts": {k: len(v) for k, v in sample_index["hard_cases"].items()},
    }
    (output_dir / "context_pack_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nContext pack ready: {output_dir}")


if __name__ == "__main__":
    main()
