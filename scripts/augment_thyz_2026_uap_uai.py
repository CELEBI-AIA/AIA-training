#!/usr/bin/env python3
"""
No-argument optimized offline augmentation script for YOLO-format datasets.

How to use:
    1) Put this file inside your YOLO dataset folder, or run it from that folder.
    2) Run only:
           python augment_yolo_optimized.py

Default behavior:
    - Source dataset: the current working directory.
    - If you run inside train/images or images, the script automatically detects the dataset root.
    - Output dataset: a sibling folder named <dataset_name>_augmented.
    - Original train/val/test files are copied unchanged first.
    - Only training images are augmented by default; val/test stay clean.
    - Each original training image gets 4 fixed high-value augmentation variants.

Fixed augmentation profile:
    aug00_affine_photo      -> small affine + bbox reprojection + photometric jitter
    aug01_zoomout_shadow    -> zoom-out / higher altitude + bbox reprojection + shadow + photometric jitter
    aug02_perspective_photo -> subtle perspective tilt + bbox reprojection + photometric jitter + mild blur
    aug03_crop_cutout       -> focused crop when safe + bbox reprojection + safe cutout + photometric jitter

Why this profile:
    It keeps the strongest UAV/YOLO augmentations and removes excessive always-on blur, fog,
    darkening, and overexposure variants that can bloat the dataset and hurt validation realism.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  FIXED SETTINGS — no command-line input needed
# ─────────────────────────────────────────────

SEED = 20260503
VARIANTS_PER_IMAGE = 4
JPEG_QUALITY = 92
COPY_ORIGINAL_DATASET = True
CLEAN_OUTPUT_FIRST = True
AUGMENT_EVAL_SPLITS = False  # keep val/test clean for honest evaluation
OUTPUT_SUFFIX = "_augmented"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EVAL_SPLIT_NAMES = {"val", "valid", "validation", "test", "testing"}
TRAIN_SPLIT_NAMES = {"train", "training"}

AUGMENTED_STEM_RE = re.compile(
    r"("
    r"_aug\d+.*|"
    r"_blur|_motionblur|_noise|_clahe|_shadow|_gammadark|_overexp|_fog|"
    r"_perspective\d*|_zoomout\d*|_affine|_crop|_cutout|_photo|_haze"
    r")$",
    re.IGNORECASE,
)

FIXED_VARIANT_NAMES = [
    "affine_photo",
    "zoomout_shadow",
    "perspective_photo",
    "crop_cutout",
]


@dataclass(frozen=True)
class YoloLabel:
    cls: str
    x: float
    y: float
    w: float
    h: float
    extras: tuple[str, ...] = ()


@dataclass
class AugmentStats:
    source_root: str = ""
    output_root: str = ""
    original_images_total: int = 0
    train_images_augmented: int = 0
    eval_images_kept_clean: int = 0
    images_without_labels_skipped: int = 0
    unreadable_images_skipped: int = 0
    augmented_images_written: int = 0
    labels_written: int = 0
    dropped_empty_augments: int = 0


# ─────────────────────────────────────────────
#  PATH / DATASET HELPERS
# ─────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_augmented_stem(stem: str) -> bool:
    return AUGMENTED_STEM_RE.search(stem) is not None


def has_path_part(path: Path, names: set[str]) -> bool:
    return any(part.lower() in names for part in path.parts)


def is_eval_split(path: Path) -> bool:
    return has_path_part(path, EVAL_SPLIT_NAMES)


def is_train_split(path: Path) -> bool:
    return has_path_part(path, TRAIN_SPLIT_NAMES)


def detect_source_root() -> Path:
    """Use cwd, but recover the YOLO dataset root if user runs from images/ or train/images/."""
    cwd = Path.cwd().resolve()

    if cwd.name.lower() == "images" and (cwd.parent / "labels").exists():
        # dataset/train/images -> dataset
        if cwd.parent.name.lower() in TRAIN_SPLIT_NAMES | EVAL_SPLIT_NAMES:
            return cwd.parent.parent.resolve()
        # dataset/images -> dataset
        return cwd.parent.resolve()

    if cwd.name.lower() in TRAIN_SPLIT_NAMES | EVAL_SPLIT_NAMES:
        if (cwd / "images").exists() and (cwd / "labels").exists():
            return cwd.parent.resolve()

    return cwd


def default_output_root(source_root: Path) -> Path:
    return source_root.with_name(f"{source_root.name}{OUTPUT_SUFFIX}")


def safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def destination_for(source_path: Path, source_root: Path, output_root: Path) -> Path:
    return output_root / safe_relpath(source_path, source_root)


def should_skip_file_on_copy(path: Path) -> bool:
    if path.name == Path(__file__).name:
        return True
    if is_augmented_stem(path.stem):
        return True
    return False


def copy_original_dataset(source_root: Path, output_root: Path) -> None:
    """Copy original dataset files first so val/test and metadata stay intact."""
    if not COPY_ORIGINAL_DATASET:
        return
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_file_on_copy(path):
            continue
        dest = destination_for(path, source_root, output_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


# ─────────────────────────────────────────────
#  YOLO LABEL HELPERS
# ─────────────────────────────────────────────

def read_yolo_labels(path: Path) -> list[YoloLabel]:
    labels: list[YoloLabel] = []
    if not path.exists():
        return labels

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.replace("\x00", "").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            labels.append(YoloLabel(parts[0], x, y, w, h, tuple(parts[5:])))
    return labels


def write_yolo_labels(path: Path, labels: list[YoloLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for label in labels:
        extra = " " + " ".join(label.extras) if label.extras else ""
        lines.append(f"{label.cls} {label.x:.6f} {label.y:.6f} {label.w:.6f} {label.h:.6f}{extra}\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_label_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for label_path in root.rglob("*.txt"):
        if label_path.name.lower() in {"classes.txt", "labels.txt"}:
            continue
        if is_augmented_stem(label_path.stem):
            continue
        index.setdefault(label_path.stem, []).append(label_path)
    return index


def candidate_label_paths_for_image(image_path: Path) -> list[Path]:
    candidates: list[Path] = [image_path.with_suffix(".txt")]

    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            replaced = parts.copy()
            replaced[i] = "labels"
            candidates.append(Path(*replaced).with_suffix(".txt"))

    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def find_label_path(image_path: Path, label_index: dict[str, list[Path]]) -> Path | None:
    for candidate in candidate_label_paths_for_image(image_path):
        if candidate.exists():
            return candidate

    matches = label_index.get(image_path.stem, [])
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    image_parts = set(part.lower() for part in image_path.parts)
    matches = sorted(
        matches,
        key=lambda p: len(image_parts.intersection(part.lower() for part in p.parts)),
        reverse=True,
    )
    return matches[0]


# ─────────────────────────────────────────────
#  BBOX TRANSFORM HELPERS
# ─────────────────────────────────────────────

def yolo_to_xyxy(label: YoloLabel, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = (label.x - label.w / 2.0) * width
    y1 = (label.y - label.h / 2.0) * height
    x2 = (label.x + label.w / 2.0) * width
    y2 = (label.y + label.h / 2.0) * height
    return x1, y1, x2, y2


def xyxy_to_yolo(
    cls: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    *,
    extras: tuple[str, ...] = (),
    original_area: float | None = None,
    min_visibility: float = 0.35,
    min_size_px: float = 3.0,
    min_area_px: float = 12.0,
) -> YoloLabel | None:
    original_area = original_area if original_area is not None else max(0.0, x2 - x1) * max(0.0, y2 - y1)

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh

    if bw < min_size_px or bh < min_size_px or area < min_area_px:
        return None
    if original_area > 0 and area / original_area < min_visibility:
        return None

    aspect = bw / max(1e-6, bh)
    if aspect < 0.12 or aspect > 8.0:
        return None

    return YoloLabel(
        cls=cls,
        x=(x1 + x2) / 2.0 / width,
        y=(y1 + y2) / 2.0 / height,
        w=bw / width,
        h=bh / height,
        extras=extras,
    )


def transform_labels_affine(
    labels: list[YoloLabel],
    matrix: np.ndarray,
    width: int,
    height: int,
    *,
    min_visibility: float = 0.35,
) -> list[YoloLabel]:
    new_labels: list[YoloLabel] = []
    for label in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(label, width, height)
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        transformed = corners @ matrix.T
        converted = xyxy_to_yolo(
            label.cls,
            float(transformed[:, 0].min()),
            float(transformed[:, 1].min()),
            float(transformed[:, 0].max()),
            float(transformed[:, 1].max()),
            width,
            height,
            extras=label.extras,
            original_area=original_area,
            min_visibility=min_visibility,
        )
        if converted is not None:
            new_labels.append(converted)
    return new_labels


def transform_labels_perspective(
    labels: list[YoloLabel],
    matrix: np.ndarray,
    width: int,
    height: int,
    *,
    min_visibility: float = 0.35,
) -> list[YoloLabel]:
    new_labels: list[YoloLabel] = []
    for label in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(label, width, height)
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        converted = xyxy_to_yolo(
            label.cls,
            float(transformed[:, 0].min()),
            float(transformed[:, 1].min()),
            float(transformed[:, 0].max()),
            float(transformed[:, 1].max()),
            width,
            height,
            extras=label.extras,
            original_area=original_area,
            min_visibility=min_visibility,
        )
        if converted is not None:
            new_labels.append(converted)
    return new_labels


def bbox_intersects(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


# ─────────────────────────────────────────────
#  FIXED HIGH-VALUE AUGMENTATIONS
# ─────────────────────────────────────────────

def apply_affine(image: np.ndarray, labels: list[YoloLabel], rng: random.Random) -> tuple[np.ndarray, list[YoloLabel]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-6.0, 6.0)
    scale = rng.uniform(0.92, 1.10)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += [rng.uniform(-0.04, 0.04) * width, rng.uniform(-0.04, 0.04) * height]
    border = tuple(int(v) for v in cv2.mean(image)[:3])
    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return warped, transform_labels_affine(labels, matrix, width, height, min_visibility=0.35)


def apply_zoom_out(image: np.ndarray, labels: list[YoloLabel], rng: random.Random) -> tuple[np.ndarray, list[YoloLabel]]:
    height, width = image.shape[:2]
    scale = rng.uniform(0.64, 0.82)
    new_w, new_h = max(2, int(width * scale)), max(2, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    pad_x = rng.randint(0, width - new_w) if width > new_w else 0
    pad_y = rng.randint(0, height - new_h) if height > new_h else 0
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)
    return canvas, transform_labels_affine(labels, matrix, width, height, min_visibility=0.40)


def apply_perspective(image: np.ndarray, labels: list[YoloLabel], rng: random.Random) -> tuple[np.ndarray, list[YoloLabel]]:
    height, width = image.shape[:2]
    margin = int(min(height, width) * rng.uniform(0.035, 0.075))
    if margin < 2:
        return image, labels
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([
        [rng.randint(0, margin), rng.randint(0, margin)],
        [width - rng.randint(0, margin), rng.randint(0, margin)],
        [width - rng.randint(0, margin), height - rng.randint(0, margin)],
        [rng.randint(0, margin), height - rng.randint(0, margin)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    return warped, transform_labels_perspective(labels, matrix, width, height, min_visibility=0.35)


def apply_focused_crop(image: np.ndarray, labels: list[YoloLabel], rng: random.Random) -> tuple[np.ndarray, list[YoloLabel]]:
    if not labels:
        return image, labels

    height, width = image.shape[:2]
    chosen = rng.choice(labels)
    x1, y1, x2, y2 = yolo_to_xyxy(chosen, width, height)

    pad_x = rng.uniform(0.22, 0.45) * width
    pad_y = rng.uniform(0.22, 0.45) * height
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))

    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    if crop_w < width * 0.45 or crop_h < height * 0.45:
        return image, labels

    crop = image[cy1:cy2, cx1:cx2]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    new_labels: list[YoloLabel] = []

    for label in labels:
        bx1, by1, bx2, by2 = yolo_to_xyxy(label, width, height)
        ix1, iy1 = max(bx1, cx1), max(by1, cy1)
        ix2, iy2 = min(bx2, cx2), min(by2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        original_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        visible_area = (ix2 - ix1) * (iy2 - iy1)
        if visible_area / original_area < 0.55:
            continue
        converted = xyxy_to_yolo(
            label.cls,
            (ix1 - cx1) / crop_w * width,
            (iy1 - cy1) / crop_h * height,
            (ix2 - cx1) / crop_w * width,
            (iy2 - cy1) / crop_h * height,
            width,
            height,
            extras=label.extras,
            original_area=original_area,
            min_visibility=0.55,
        )
        if converted is not None:
            new_labels.append(converted)

    return resized, new_labels


def apply_photometric(image: np.ndarray, rng: random.Random, *, clahe: bool = False, noise: bool = True) -> np.ndarray:
    out = image.astype(np.float32)
    out = out * rng.uniform(0.84, 1.20) + rng.uniform(-20.0, 20.0)
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.75, 1.28)
    hsv[:, :, 2] *= rng.uniform(0.84, 1.16)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if noise:
        noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        noise_arr = noise_rng.normal(0, rng.uniform(2.5, 7.5), out.shape)
        out = np.clip(out.astype(np.float32) + noise_arr, 0, 255).astype(np.uint8)

    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhancer = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 2.4), tileGridSize=(8, 8))
        l = enhancer.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return out


def apply_shadow(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    band_w = int(width * rng.uniform(0.18, 0.34))
    if band_w <= 1:
        return image

    x_start = rng.randint(0, max(0, width - band_w))
    tilt = int(height * rng.uniform(-0.12, 0.12))
    pts = np.array(
        [
            [x_start, 0],
            [x_start + band_w, 0],
            [x_start + band_w + tilt, height],
            [x_start + tilt, height],
        ],
        dtype=np.int32,
    )

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    blur_size = max(21, int(min(height, width) * 0.045) | 1)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    factor = rng.uniform(0.16, 0.32)
    out = image.astype(np.float32) * (1.0 - mask[:, :, None] * factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_small_motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    size = rng.choice([3, 5])
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        kernel[:, size // 2] = 1.0
    return cv2.filter2D(image, -1, kernel / size)


def apply_safe_cutout(image: np.ndarray, labels: list[YoloLabel], rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    boxes = [yolo_to_xyxy(label, width, height) for label in labels]
    out = image.copy()

    for _ in range(rng.randint(1, 2)):
        cut_w = rng.randint(max(8, width // 32), max(12, width // 11))
        cut_h = rng.randint(max(8, height // 32), max(12, height // 11))
        for _attempt in range(20):
            x1 = rng.randint(0, max(0, width - cut_w))
            y1 = rng.randint(0, max(0, height - cut_h))
            patch_box = (x1, y1, x1 + cut_w, y1 + cut_h)
            if any(bbox_intersects(patch_box, box) for box in boxes):
                continue
            fill = np.array([int(v) for v in cv2.mean(out)[:3]], dtype=np.int16)
            jitter = rng.randint(-14, 14)
            out[y1 : y1 + cut_h, x1 : x1 + cut_w] = np.clip(fill + jitter, 0, 255).astype(np.uint8)
            break

    return out


def augment_fixed_variant(
    image: np.ndarray,
    labels: list[YoloLabel],
    rng: random.Random,
    variant_index: int,
) -> tuple[np.ndarray, list[YoloLabel], list[str]]:
    """Apply one of the four fixed, high-value augmentation recipes."""
    out = image.copy()
    out_labels = list(labels)
    ops: list[str] = []

    recipe = FIXED_VARIANT_NAMES[variant_index % len(FIXED_VARIANT_NAMES)]

    if recipe == "affine_photo":
        out, out_labels = apply_affine(out, out_labels, rng)
        ops.append("affine_bbox_reproject")
        if not out_labels:
            return out, out_labels, ops
        out = apply_photometric(out, rng, clahe=False, noise=True)
        ops.append("brightness_contrast_hsv_noise")

    elif recipe == "zoomout_shadow":
        out, out_labels = apply_zoom_out(out, out_labels, rng)
        ops.append("zoom_out_bbox_reproject")
        if not out_labels:
            return out, out_labels, ops
        out = apply_shadow(out, rng)
        ops.append("soft_shadow")
        out = apply_photometric(out, rng, clahe=False, noise=True)
        ops.append("brightness_contrast_hsv_noise")

    elif recipe == "perspective_photo":
        out, out_labels = apply_perspective(out, out_labels, rng)
        ops.append("perspective_bbox_reproject")
        if not out_labels:
            return out, out_labels, ops
        out = apply_photometric(out, rng, clahe=True, noise=True)
        ops.append("brightness_contrast_hsv_noise_clahe")
        out = apply_small_motion_blur(out, rng)
        ops.append("small_motion_blur")

    elif recipe == "crop_cutout":
        out, out_labels = apply_focused_crop(out, out_labels, rng)
        ops.append("focused_crop_bbox_reproject_if_safe")
        if not out_labels:
            return out, out_labels, ops
        out = apply_photometric(out, rng, clahe=False, noise=True)
        ops.append("brightness_contrast_hsv_noise")
        out = apply_safe_cutout(out, out_labels, rng)
        ops.append("bbox_safe_cutout")

    return out, out_labels, ops


# ─────────────────────────────────────────────
#  REVIEW SHEET
# ─────────────────────────────────────────────

def draw_labels(image: np.ndarray, labels: list[YoloLabel]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for label in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(label, width, height)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            out,
            str(label.cls),
            (int(x1), max(12, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def write_review_sheet(output_root: Path, records: list[dict], max_samples: int = 48) -> None:
    if not records:
        return

    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(records) // max_samples)
    samples = records[::step][:max_samples]
    thumbs: list[np.ndarray] = []

    for record in samples:
        image = cv2.imread(record["image"])
        if image is None:
            continue
        labels = [YoloLabel(str(item["cls"]), *item["xywh"], tuple(item.get("extras", []))) for item in record["labels"]]
        image = draw_labels(image, labels)
        height, width = image.shape[:2]
        scale = min(300 / width, 168 / height)
        thumb = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        canvas = np.full((205, 322, 3), 255, np.uint8)
        y = (168 - thumb.shape[0]) // 2
        x = (322 - thumb.shape[1]) // 2
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(canvas, record["variant"], (6, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(canvas)

    if not thumbs:
        return

    rows: list[np.ndarray] = []
    for i in range(0, len(thumbs), 3):
        row = thumbs[i : i + 3]
        while len(row) < 3:
            row.append(np.full((205, 322, 3), 255, np.uint8))
        rows.append(np.hstack(row))

    cv2.imwrite(str(review_dir / "augmented_labels_sheet.jpg"), np.vstack(rows))


# ─────────────────────────────────────────────
#  MAIN AUGMENTATION FLOW
# ─────────────────────────────────────────────

def collect_original_images(root: Path) -> tuple[list[Path], int]:
    all_images: list[Path] = []
    eval_count = 0

    has_train_folder = any(path.is_dir() and path.name.lower() in TRAIN_SPLIT_NAMES for path in root.iterdir())

    for path in root.rglob("*"):
        if not path.is_file() or not is_image(path):
            continue
        if is_augmented_stem(path.stem):
            continue
        if is_eval_split(path):
            eval_count += 1
            if not AUGMENT_EVAL_SPLITS:
                continue
        if has_train_folder and not is_train_split(path):
            # In standard YOLO datasets, augment only train split.
            continue
        all_images.append(path)

    return sorted(all_images), eval_count


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    cv2.imwrite(str(path), image, params)


def augment_dataset(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")
    if source_root == output_root:
        raise ValueError("Output directory must be separate from source directory.")

    if CLEAN_OUTPUT_FIRST and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copy_original_dataset(source_root, output_root)

    label_index = build_label_index(source_root)
    image_paths, eval_count = collect_original_images(source_root)
    rng = random.Random(SEED)

    stats = AugmentStats(
        source_root=str(source_root),
        output_root=str(output_root),
        original_images_total=len(image_paths),
        eval_images_kept_clean=eval_count if not AUGMENT_EVAL_SPLITS else 0,
    )
    records: list[dict] = []

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Training images selected for augmentation: {len(image_paths)}")
    print(f"Fixed variants per image: {VARIANTS_PER_IMAGE} -> {', '.join(FIXED_VARIANT_NAMES)}")

    for index, image_path in enumerate(image_paths, start=1):
        label_path = find_label_path(image_path, label_index)
        if label_path is None:
            stats.images_without_labels_skipped += 1
            continue

        labels = read_yolo_labels(label_path)
        if not labels:
            stats.images_without_labels_skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            stats.unreadable_images_skipped += 1
            continue

        dest_image_original = destination_for(image_path, source_root, output_root)
        dest_label_original = destination_for(label_path, source_root, output_root)
        stats.train_images_augmented += 1

        for variant_index in range(VARIANTS_PER_IMAGE):
            local_rng = random.Random(rng.randint(0, 2**31 - 1))
            augmented, new_labels, ops = augment_fixed_variant(image, labels, local_rng, variant_index)
            if not new_labels:
                stats.dropped_empty_augments += 1
                continue

            variant_name = FIXED_VARIANT_NAMES[variant_index]
            suffix = f"_aug{variant_index:02d}_{variant_name}"
            out_image_path = dest_image_original.with_name(f"{dest_image_original.stem}{suffix}.jpg")
            out_label_path = dest_label_original.with_name(f"{dest_label_original.stem}{suffix}.txt")

            imwrite(out_image_path, augmented)
            write_yolo_labels(out_label_path, new_labels)

            stats.augmented_images_written += 1
            stats.labels_written += len(new_labels)
            records.append(
                {
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "image": str(out_image_path),
                    "label": str(out_label_path),
                    "variant": out_image_path.stem,
                    "ops": ops,
                    "labels": [
                        {"cls": label.cls, "xywh": [label.x, label.y, label.w, label.h], "extras": list(label.extras)}
                        for label in new_labels
                    ],
                }
            )

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_paths)} training images...")

    manifest = {
        "stats": asdict(stats),
        "settings": {
            "seed": SEED,
            "variants_per_image": VARIANTS_PER_IMAGE,
            "jpeg_quality": JPEG_QUALITY,
            "augment_eval_splits": AUGMENT_EVAL_SPLITS,
            "copy_original_dataset": COPY_ORIGINAL_DATASET,
            "clean_output_first": CLEAN_OUTPUT_FIRST,
        },
        "fixed_augmentation_profile": {
            "aug00_affine_photo": [
                "small rotation/scale/translation",
                "bbox reprojection",
                "brightness/contrast/HSV jitter",
                "mild Gaussian sensor noise",
            ],
            "aug01_zoomout_shadow": [
                "zoom-out / higher altitude simulation",
                "bbox reprojection",
                "soft diagonal shadow",
                "brightness/contrast/HSV jitter",
                "mild Gaussian sensor noise",
            ],
            "aug02_perspective_photo": [
                "subtle drone tilt / perspective warp",
                "bbox reprojection",
                "brightness/contrast/HSV jitter",
                "light CLAHE",
                "small motion blur",
            ],
            "aug03_crop_cutout": [
                "focused crop only when bbox visibility is safe",
                "bbox reprojection",
                "brightness/contrast/HSV jitter",
                "bbox-safe background cutout",
            ],
        },
        "removed_from_default_profile": [
            "always-on heavy 15x15 Gaussian blur",
            "always-on dark+noise output",
            "always-on overexposure output",
            "always-on fog output",
            "unbounded 10x/12x dataset expansion",
            "class/color-specific copy-paste that only fits special UAP/UAI marker assumptions",
        ],
        "records": records,
    }

    (output_root / "augmentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_review_sheet(output_root, records)
    return manifest


def main() -> None:
    source_root = detect_source_root()
    output_root = default_output_root(source_root)
    manifest = augment_dataset(source_root, output_root)

    print("\n[SUCCESS] Augmentation completed.")
    print(json.dumps(manifest["stats"], indent=2))
    print(f"Manifest: {output_root / 'augmentation_manifest.json'}")
    print(f"Review sheet: {output_root / 'review' / 'augmented_labels_sheet.jpg'}")


if __name__ == "__main__":
    main()
