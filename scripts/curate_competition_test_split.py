#!/usr/bin/env python3
"""
Curate a competition-like test split from a mixed YOLO dataset.

Primary goal:
- separate likely competition-like samples from clearly external stock aerial data
- sanitize label files to strict 5-column detection format
- emit a summary report so the operator can review what was kept/excluded

Heuristics are intentionally conservative:
- filenames starting with "7sbt-" are treated as external stock aerial imagery
- filenames starting with "Ornek_Veri_" are treated as competition-like
- any sample containing UAP/UAI labels (class 2 or 3) is treated as competition-like
- everything else is kept in a review bucket unless explicitly forced
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
EXTERNAL_PREFIXES = ("7sbt-",)
COMPETITION_PREFIXES = ("Ornek_Veri_",)
CLASS_NAMES = {0: "vehicle", 1: "human", 2: "uap", 3: "uai"}


@dataclass
class LabelParseResult:
    kept_lines: list[str]
    malformed_lines: int
    duplicate_lines: int
    class_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate a competition-like YOLO test split")
    parser.add_argument("--dataset-root", required=True, help="Root containing train/valid/test or split folders")
    parser.add_argument("--split", default="test", help="Source split to curate (default: test)")
    parser.add_argument("--output-root", required=True, help="Output dataset root")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize curated files (default: copy)",
    )
    parser.add_argument(
        "--keep-review",
        action="store_true",
        help="Keep ambiguous non-stock samples in review/ instead of excluding them",
    )
    parser.add_argument(
        "--competition-only",
        action="store_true",
        help="Keep only clearly competition-like samples in curated test",
    )
    parser.add_argument(
        "--report-name",
        default="curation_report.json",
        help="Summary report file name written under output root",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_label_file(label_path: Path) -> LabelParseResult:
    kept_lines: list[str] = []
    seen = set()
    malformed = 0
    duplicates = 0
    class_ids: list[int] = []

    if not label_path.exists():
        return LabelParseResult(kept_lines=[], malformed_lines=0, duplicate_lines=0, class_ids=[])

    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            malformed += 1
            continue
        try:
            cls_id = int(float(parts[0]))
            floats = [float(v) for v in parts[1:5]]
        except ValueError:
            malformed += 1
            continue

        if cls_id not in CLASS_NAMES:
            malformed += 1
            continue

        x, y, w, h = floats
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            malformed += 1
            continue

        normalized = f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
        kept_lines.append(normalized)
        class_ids.append(cls_id)

    return LabelParseResult(
        kept_lines=kept_lines,
        malformed_lines=malformed,
        duplicate_lines=duplicates,
        class_ids=class_ids,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        dst.symlink_to(src)


def decide_bucket(name: str, class_ids: Iterable[int], competition_only: bool) -> str:
    if name.startswith(EXTERNAL_PREFIXES):
        return "excluded_external_stock"
    if name.startswith(COMPETITION_PREFIXES):
        return "competition_like"
    class_id_set = set(class_ids)
    if 2 in class_id_set or 3 in class_id_set:
        return "competition_like"
    if competition_only:
        return "excluded_non_competition"
    return "review"


def write_yaml(output_root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_root}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "nc: 4",
            "names:",
            "  0: vehicle",
            "  1: human",
            "  2: uap",
            "  3: uai",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_dir = dataset_root / args.split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected split directories at {images_dir} and {labels_dir}"
        )

    output_root = Path(args.output_root).expanduser().resolve()
    curated_images = output_root / "test" / "images"
    curated_labels = output_root / "test" / "labels"
    review_images = output_root / "review" / "images"
    review_labels = output_root / "review" / "labels"
    excluded_images = output_root / "excluded_external_stock" / "images"
    excluded_labels = output_root / "excluded_external_stock" / "labels"

    ensure_dir(curated_images)
    ensure_dir(curated_labels)
    ensure_dir(excluded_images)
    ensure_dir(excluded_labels)
    if args.keep_review:
        ensure_dir(review_images)
        ensure_dir(review_labels)

    summary = {
        "source_dataset_root": str(dataset_root),
        "source_split": args.split,
        "copy_mode": args.copy_mode,
        "competition_only": args.competition_only,
        "keep_review": args.keep_review,
        "totals": Counter(),
        "class_counts": {"competition_like": Counter(), "review": Counter(), "excluded": Counter()},
        "samples": {
            "competition_like": [],
            "review": [],
            "excluded_external_stock": [],
            "excluded_non_competition": [],
        },
    }

    for image_path in list_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        parsed = parse_label_file(label_path)
        bucket = decide_bucket(image_path.name, parsed.class_ids, args.competition_only)

        summary["totals"]["images_seen"] += 1
        summary["totals"]["labels_malformed"] += parsed.malformed_lines
        summary["totals"]["labels_duplicate_removed"] += parsed.duplicate_lines
        if not parsed.kept_lines:
            summary["totals"]["empty_after_sanitize"] += 1

        if bucket == "competition_like":
            dest_img = curated_images / image_path.name
            dest_lbl = curated_labels / label_path.name
            materialize(image_path, dest_img, args.copy_mode)
            dest_lbl.write_text("\n".join(parsed.kept_lines) + ("\n" if parsed.kept_lines else ""), encoding="utf-8")
            summary["totals"]["competition_like_kept"] += 1
            for cls_id in parsed.class_ids:
                summary["class_counts"]["competition_like"][CLASS_NAMES[cls_id]] += 1
        elif bucket == "review" and args.keep_review:
            dest_img = review_images / image_path.name
            dest_lbl = review_labels / label_path.name
            materialize(image_path, dest_img, args.copy_mode)
            dest_lbl.write_text("\n".join(parsed.kept_lines) + ("\n" if parsed.kept_lines else ""), encoding="utf-8")
            summary["totals"]["review_kept"] += 1
            for cls_id in parsed.class_ids:
                summary["class_counts"]["review"][CLASS_NAMES[cls_id]] += 1
        else:
            dest_img = excluded_images / image_path.name
            dest_lbl = excluded_labels / label_path.name
            materialize(image_path, dest_img, args.copy_mode)
            dest_lbl.write_text("\n".join(parsed.kept_lines) + ("\n" if parsed.kept_lines else ""), encoding="utf-8")
            key = "excluded_external_stock" if bucket == "excluded_external_stock" else "excluded_non_competition"
            summary["totals"][key] += 1
            for cls_id in parsed.class_ids:
                summary["class_counts"]["excluded"][CLASS_NAMES[cls_id]] += 1

        if len(summary["samples"].get(bucket, [])) < 20:
            summary["samples"].setdefault(bucket, []).append(image_path.name)

    write_yaml(output_root)

    report_path = output_root / args.report_name
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=lambda x: dict(x)),
        encoding="utf-8",
    )

    print("\nCurated competition-like test split ready.")
    print(f"Output root : {output_root}")
    print(f"Report      : {report_path}")
    print(f"Kept test   : {summary['totals'].get('competition_like_kept', 0)}")
    print(f"Review      : {summary['totals'].get('review_kept', 0)}")
    print(f"Excluded    : {summary['totals'].get('excluded_external_stock', 0) + summary['totals'].get('excluded_non_competition', 0)}")
    print(f"Malformed labels removed : {summary['totals'].get('labels_malformed', 0)}")
    print(f"Duplicate labels removed : {summary['totals'].get('labels_duplicate_removed', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
