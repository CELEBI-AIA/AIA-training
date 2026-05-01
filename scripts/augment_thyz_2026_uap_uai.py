import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE = Path("datasets/THYZ_2026_UAP_UAI_auto_labeled")
DEFAULT_OUTPUT = Path("datasets/THYZ_2026_UAP_UAI_augmented")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_PASTED_MARKER_PIXELS = 40


def _read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if cls in {2, 3} and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            labels.append((cls, x, y, w, h))
    return labels


def _write_yolo_labels(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for cls, x, y, w, h in labels]
    path.write_text("".join(lines), encoding="utf-8")


def _yolo_to_xyxy(label: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, float, float, float, float]:
    cls, x, y, w, h = label
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    return cls, x1, y1, x2, y2


def _xyxy_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    original_w = max(0.0, x2 - x1)
    original_h = max(0.0, y2 - y1)
    original_area = original_w * original_h
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 6 or bh < 6:
        return None
    if original_area > 0 and (bw * bh) / original_area < 0.55:
        return None
    aspect = bw / bh
    if aspect < 0.25 or aspect > 4.0:
        return None
    return (cls, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, bw / width, bh / height)


def _bbox_intersects(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _marker_mask_for_class(patch: np.ndarray, cls: int) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if cls == 2:
        mask = cv2.inRange(hsv, np.array([88, 55, 55]), np.array([128, 255, 255]))
    else:
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255])),
            cv2.inRange(hsv, np.array([168, 60, 60]), np.array([179, 255, 255])),
        )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _paste_marker(
    background: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    dx1: int,
    dy1: int,
) -> np.ndarray:
    out = background.copy()
    height, width = patch.shape[:2]
    roi = out[dy1 : dy1 + height, dx1 : dx1 + width]
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.82
    blended = roi.astype(np.float32) * (1.0 - alpha) + patch.astype(np.float32) * alpha
    out[dy1 : dy1 + height, dx1 : dx1 + width] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _placement_looks_like_ground(background: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    height, width = background.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    x2 = min(width, x2)
    y2 = min(height, y2)
    roi = background[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green_ratio = np.mean((hue >= 35) & (hue <= 95) & (sat >= 35) & (val >= 35))
    earth_ratio = np.mean((hue >= 8) & (hue <= 35) & (sat >= 18) & (val >= 45) & (val <= 235))
    low_sat_ground_ratio = np.mean((sat < 55) & (val >= 55) & (val <= 190))
    too_bright_ratio = np.mean(val > 238)
    return (green_ratio + earth_ratio + low_sat_ground_ratio) >= 0.42 and too_bright_ratio < 0.35


def _scaled_label_box_in_patch(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sx1: int,
    sy1: int,
    scale_x: float,
    scale_y: float,
    dx1: int,
    dy1: int,
) -> tuple[float, float, float, float]:
    return (
        dx1 + (x1 - sx1) * scale_x,
        dy1 + (y1 - sy1) * scale_y,
        dx1 + (x2 - sx1) * scale_x,
        dy1 + (y2 - sy1) * scale_y,
    )


def _safe_cutout(image: np.ndarray, labels: list[tuple[int, float, float, float, float]], rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    boxes = [tuple(_yolo_to_xyxy(label, width, height)[1:]) for label in labels]
    out = image.copy()
    for _ in range(rng.randint(1, 3)):
        cut_w = rng.randint(max(12, width // 25), max(24, width // 9))
        cut_h = rng.randint(max(12, height // 25), max(24, height // 9))
        for _attempt in range(20):
            x1 = rng.randint(0, max(0, width - cut_w))
            y1 = rng.randint(0, max(0, height - cut_h))
            patch_box = (x1, y1, x1 + cut_w, y1 + cut_h)
            if any(_bbox_intersects(patch_box, box) for box in boxes):
                continue
            fill = np.array([int(v) for v in cv2.mean(out)[:3]], dtype=np.int16)
            noise = rng.randint(-18, 18)
            out[y1 : y1 + cut_h, x1 : x1 + cut_w] = np.clip(fill + noise, 0, 255).astype(np.uint8)
            break
    return out


def _photometric(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)
    out = out * rng.uniform(0.78, 1.28) + rng.uniform(-28, 28)
    if rng.random() < 0.55:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.65, 1.35)
        hsv[:, :, 2] *= rng.uniform(0.75, 1.25)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.45:
        noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        out += noise_rng.normal(0, rng.uniform(3, 12), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur_or_sharpen(image: np.ndarray, rng: random.Random) -> np.ndarray:
    choice = rng.choice(["motion", "defocus", "sharpen", "none"])
    if choice == "motion":
        size = rng.choice([3, 5, 7])
        kernel = np.zeros((size, size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[size // 2, :] = 1.0
        else:
            kernel[:, size // 2] = 1.0
        return cv2.filter2D(image, -1, kernel / size)
    if choice == "defocus":
        return cv2.GaussianBlur(image, (rng.choice([3, 5]), rng.choice([3, 5])), 0)
    if choice == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    return image


def _affine(image: np.ndarray, labels: list[tuple[int, float, float, float, float]], rng: random.Random):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(-7.0, 7.0), rng.uniform(0.88, 1.14))
    matrix[:, 2] += [rng.uniform(-0.045, 0.045) * width, rng.uniform(-0.045, 0.045) * height]
    border = tuple(int(v) for v in cv2.mean(image)[:3])
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    new_labels = []
    for label in labels:
        cls, x1, y1, x2, y2 = _yolo_to_xyxy(label, width, height)
        points = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        transformed = points @ matrix.T
        converted = _xyxy_to_yolo(
            cls,
            transformed[:, 0].min(),
            transformed[:, 1].min(),
            transformed[:, 0].max(),
            transformed[:, 1].max(),
            width,
            height,
        )
        if converted:
            new_labels.append(converted)
    return warped, new_labels


def _focused_crop(image: np.ndarray, labels: list[tuple[int, float, float, float, float]], rng: random.Random):
    height, width = image.shape[:2]
    if not labels:
        return image, labels
    _, x1, y1, x2, y2 = _yolo_to_xyxy(rng.choice(labels), width, height)
    pad_x = rng.uniform(0.15, 0.38) * width
    pad_y = rng.uniform(0.15, 0.38) * height
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if (cx2 - cx1) < width * 0.35 or (cy2 - cy1) < height * 0.35:
        return image, labels
    crop = image[cy1:cy2, cx1:cx2]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    new_labels = []
    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    for label in labels:
        cls, bx1, by1, bx2, by2 = _yolo_to_xyxy(label, width, height)
        ix1, iy1 = max(bx1, cx1), max(by1, cy1)
        ix2, iy2 = min(bx2, cx2), min(by2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visibility = ((ix2 - ix1) * (iy2 - iy1)) / max(1.0, (bx2 - bx1) * (by2 - by1))
        if visibility < 0.55:
            continue
        converted = _xyxy_to_yolo(
            cls,
            (ix1 - cx1) / crop_w * width,
            (iy1 - cy1) / crop_h * height,
            (ix2 - cx1) / crop_w * width,
            (iy2 - cy1) / crop_h * height,
            width,
            height,
        )
        if converted:
            new_labels.append(converted)
    return resized, new_labels


def _copy_paste_object(image: np.ndarray, labels: list[tuple[int, float, float, float, float]], rng: random.Random):
    if not labels or rng.random() > 0.65:
        return image, labels
    height, width = image.shape[:2]
    cls, x1, y1, x2, y2 = _yolo_to_xyxy(rng.choice(labels), width, height)
    pad = rng.randint(3, 10)
    sx1, sy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    sx2, sy2 = min(width, int(x2) + pad), min(height, int(y2) + pad)
    patch = image[sy1:sy2, sx1:sx2].copy()
    if patch.size == 0:
        return image, labels
    scale = rng.uniform(0.75, 1.25)
    new_w, new_h = max(8, int(patch.shape[1] * scale)), max(8, int(patch.shape[0] * scale))
    if new_w >= width or new_h >= height:
        return image, labels
    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = _marker_mask_for_class(patch, cls)
    if cv2.countNonZero(mask) < MIN_PASTED_MARKER_PIXELS:
        return image, labels
    scale_x = new_w / max(1, sx2 - sx1)
    scale_y = new_h / max(1, sy2 - sy1)
    existing = [tuple(_yolo_to_xyxy(label, width, height)[1:]) for label in labels]
    for _ in range(30):
        dx1, dy1 = rng.randint(0, width - new_w), rng.randint(0, height - new_h)
        new_box = _scaled_label_box_in_patch(x1, y1, x2, y2, sx1, sy1, scale_x, scale_y, dx1, dy1)
        if any(_bbox_intersects(new_box, box) for box in existing):
            continue
        if not _placement_looks_like_ground(image, new_box):
            continue
        out = _paste_marker(image, patch, mask, dx1, dy1)
        copied = _xyxy_to_yolo(cls, *new_box, width, height)
        if copied:
            return out, labels + [copied]
    return image, labels


def _copy_paste_between_frames(
    source_image: np.ndarray,
    source_labels: list[tuple[int, float, float, float, float]],
    background_image: np.ndarray,
    background_labels: list[tuple[int, float, float, float, float]],
    rng: random.Random,
):
    if not source_labels:
        return source_image, source_labels, False
    height, width = background_image.shape[:2]
    src_h, src_w = source_image.shape[:2]
    source_label = rng.choice(source_labels)
    cls, x1, y1, x2, y2 = _yolo_to_xyxy(source_label, src_w, src_h)
    pad = rng.randint(4, 14)
    sx1, sy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    sx2, sy2 = min(src_w, int(x2) + pad), min(src_h, int(y2) + pad)
    patch = source_image[sy1:sy2, sx1:sx2].copy()
    if patch.size == 0:
        return source_image, source_labels, False

    # Keep the marker realistic: same-order size with moderate scale jitter.
    scale = rng.uniform(0.82, 1.18)
    new_w, new_h = max(8, int(patch.shape[1] * scale)), max(8, int(patch.shape[0] * scale))
    if new_w >= width * 0.30 or new_h >= height * 0.30:
        return source_image, source_labels, False
    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = _marker_mask_for_class(patch, cls)
    if cv2.countNonZero(mask) < MIN_PASTED_MARKER_PIXELS:
        return source_image, source_labels, False
    scale_x = new_w / max(1, sx2 - sx1)
    scale_y = new_h / max(1, sy2 - sy1)

    existing = [tuple(_yolo_to_xyxy(label, width, height)[1:]) for label in background_labels]
    for _ in range(40):
        dx1, dy1 = rng.randint(0, width - new_w), rng.randint(0, height - new_h)
        new_box = _scaled_label_box_in_patch(x1, y1, x2, y2, sx1, sy1, scale_x, scale_y, dx1, dy1)
        if any(_bbox_intersects(new_box, box) for box in existing):
            continue
        if not _placement_looks_like_ground(background_image, new_box):
            continue
        out = _paste_marker(background_image, patch, mask, dx1, dy1)
        pasted = _xyxy_to_yolo(cls, *new_box, width, height)
        if pasted:
            return out, background_labels + [pasted], True
    return source_image, source_labels, False


def _augment_one(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    rng: random.Random,
    background_item: tuple[np.ndarray, list[tuple[int, float, float, float, float]]] | None = None,
):
    ops = []
    out, out_labels = image.copy(), list(labels)
    if background_item is not None and rng.random() < 0.55:
        bg_image, bg_labels = background_item
        cross_out, cross_labels, pasted = _copy_paste_between_frames(image, labels, bg_image, bg_labels, rng)
        if pasted:
            out, out_labels = cross_out, cross_labels
            ops.append("cross_scene_copy_paste")
    if rng.random() < 0.45:
        out, out_labels = _focused_crop(out, out_labels, rng)
        ops.append("focused_crop")
    out, out_labels = _affine(out, out_labels, rng)
    ops.append("affine")
    before = len(out_labels)
    out, out_labels = _copy_paste_object(out, out_labels, rng)
    if len(out_labels) > before:
        ops.append("copy_paste")
    out = _photometric(out, rng)
    ops.append("photometric")
    out = _blur_or_sharpen(out, rng)
    ops.append("blur_sharpen")
    if rng.random() < 0.45:
        out = _safe_cutout(out, out_labels, rng)
        ops.append("safe_cutout")
    return out, out_labels, ops


def _draw_labels(image: np.ndarray, labels: list[tuple[int, float, float, float, float]]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for label in labels:
        cls, x1, y1, x2, y2 = _yolo_to_xyxy(label, width, height)
        color = (255, 0, 0) if cls == 2 else (0, 0, 255)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(out, "uap" if cls == 2 else "uai", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def _write_review_sheet(output_dir: Path, records: list[dict]) -> None:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    samples = records[:: max(1, len(records) // 48)][:48]
    thumbs = []
    for record in samples:
        image = cv2.imread(record["image"])
        if image is None:
            continue
        labels = [(item["class_id"], *item["xywhn"]) for item in record["labels"]]
        image = _draw_labels(image, labels)
        h, w = image.shape[:2]
        scale = min(270 / w, 150 / h)
        thumb = cv2.resize(image, (int(w * scale), int(h * scale)))
        canvas = np.full((182, 292, 3), 255, np.uint8)
        y, x = (150 - thumb.shape[0]) // 2, (292 - thumb.shape[1]) // 2
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(canvas, record["variant"], (5, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    rows = []
    for index in range(0, len(thumbs), 3):
        row = thumbs[index : index + 3]
        while len(row) < 3:
            row.append(np.full((182, 292, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(str(review_dir / "augmented_labels_sheet.jpg"), np.vstack(rows))


def augment_dataset(source_dir: Path, output_dir: Path, variants_per_image: int, uai_extra_variants: int, seed: int, jpeg_quality: int) -> dict:
    train_images = source_dir / "train" / "images"
    train_labels = source_dir / "train" / "labels"
    if not train_images.exists() or not train_labels.exists():
        raise FileNotFoundError(f"Expected train/images and train/labels under {source_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in train_images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    image_items = []
    for path in image_paths:
        item_labels = _read_yolo_labels(train_labels / f"{path.stem}.txt")
        if item_labels:
            image_items.append((path, item_labels))
    rng = random.Random(seed)
    image_cache: dict[Path, np.ndarray] = {}

    def read_image(path: Path):
        if path not in image_cache:
            image_cache[path] = cv2.imread(str(path))
        return image_cache[path]

    records = []
    counts = {"source_images": 0, "augmented_images": 0, "uap_labels": 0, "uai_labels": 0, "cross_scene_images": 0}
    for image_path, labels in image_items:
        image = read_image(image_path)
        if image is None:
            continue
        counts["source_images"] += 1
        extra = uai_extra_variants if any(label[0] == 3 for label in labels) else 0
        for variant_index in range(variants_per_image + extra):
            local_rng = random.Random(rng.randint(0, 2**31 - 1))
            background_item = None
            if len(image_items) > 1:
                bg_path, bg_labels = local_rng.choice(image_items)
                if bg_path != image_path:
                    bg_image = read_image(bg_path)
                    if bg_image is not None and bg_image.shape[:2] == image.shape[:2]:
                        background_item = (bg_image, bg_labels)
            augmented, new_labels, ops = _augment_one(image, labels, local_rng, background_item)
            if not new_labels:
                continue
            out_stem = f"{image_path.stem}_aug{variant_index:02d}"
            out_image = output_dir / "train" / "images" / f"{out_stem}.jpg"
            out_label = output_dir / "train" / "labels" / f"{out_stem}.txt"
            cv2.imwrite(str(out_image), augmented, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            _write_yolo_labels(out_label, new_labels)
            counts["augmented_images"] += 1
            if "cross_scene_copy_paste" in ops:
                counts["cross_scene_images"] += 1
            counts["uap_labels"] += sum(1 for label in new_labels if label[0] == 2)
            counts["uai_labels"] += sum(1 for label in new_labels if label[0] == 3)
            records.append({
                "source": str(image_path),
                "image": str(out_image),
                "variant": out_stem,
                "ops": ops,
                "labels": [{"class_id": cls, "xywhn": [x, y, w, h]} for cls, x, y, w, h in new_labels],
            })

    manifest = {
        "source_dataset": str(source_dir),
        "output_dataset": str(output_dir),
        "seed": seed,
        "variants_per_image": variants_per_image,
        "uai_extra_variants": uai_extra_variants,
        "counts": counts,
        "records": records,
        "techniques": [
            "bbox-safe focused crop",
            "small affine transform with bbox reprojection",
            "in-scene object copy-paste",
            "cross-scene object copy-paste on another natural 2026 frame",
            "brightness/contrast/HSV jitter",
            "Gaussian sensor noise",
            "motion/defocus blur and sharpening",
            "bbox-safe cutout",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_review_sheet(output_dir, records)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create bbox-safe offline augmentations for 2026 UAP/UAI frames.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants-per-image", type=int, default=10)
    parser.add_argument("--uai-extra-variants", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()
    if args.variants_per_image < 1:
        raise ValueError("--variants-per-image must be >= 1")
    manifest = augment_dataset(args.source, args.output, args.variants_per_image, args.uai_extra_variants, args.seed, args.jpeg_quality)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Output: {args.output}")
    print(f"Review sheet: {args.output / 'review' / 'augmented_labels_sheet.jpg'}")


if __name__ == "__main__":
    main()
