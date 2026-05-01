import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_UAP_WINDOWS = [(87.5, 100.5), (232.0, 248.5)]
DEFAULT_UAI_WINDOWS = [(84.2, 91.0), (232.0, 234.5)]


def _in_windows(second: float, windows: list[tuple[float, float]]) -> bool:
    return any(start <= second <= end for start, end in windows)


def _best_contour(mask: np.ndarray, min_area: int, max_area: int):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        aspect = w / h
        if aspect < 0.35 or aspect > 2.8:
            continue
        fill = area / (w * h)
        if fill < 0.12:
            continue
        score = area * fill
        if best is None or score > best[0]:
            best = (score, area, x, y, w, h, fill)
    return best


def _write_review_sheet(output_dir: Path, records: list[dict]) -> None:
    review_dir = output_dir / "review"
    review_dir.mkdir(exist_ok=True)
    samples = records[:: max(1, len(records) // 36)][:36]
    thumbs = []

    for record in samples:
        image = cv2.imread(record["image"])
        if image is None:
            continue
        for label in record["labels"]:
            class_id = label["class_id"]
            x, y, w, h = label["xywh"]
            color = (255, 0, 0) if class_id == 2 else (0, 0, 255)
            name = "uap" if class_id == 2 else "uai"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                image,
                name,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )

        height, width = image.shape[:2]
        scale = min(270 / width, 150 / height)
        thumb = cv2.resize(image, (int(width * scale), int(height * scale)))
        canvas = np.full((180, 290, 3), 255, np.uint8)
        y = (150 - thumb.shape[0]) // 2
        x = (290 - thumb.shape[1]) // 2
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(
            canvas,
            f"f{record['frame']} {record['second']:.1f}s",
            (5, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(canvas)

    rows = []
    for index in range(0, len(thumbs), 2):
        row = thumbs[index : index + 2]
        while len(row) < 2:
            row.append(np.full((180, 290, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(str(review_dir / "candidate_labels_sheet.jpg"), np.vstack(rows))


def extract_candidates(
    video_path: Path,
    output_dir: Path,
    frame_step: int = 2,
    jpeg_quality: int = 92,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "valid"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_windows = [
        (min(start for start, _ in DEFAULT_UAP_WINDOWS + DEFAULT_UAI_WINDOWS), 100.5),
        (232.0, 248.5),
    ]

    records = []
    counts = {"images": 0, "uap": 0, "uai": 0}
    index = 0

    for start, end in all_windows:
        for frame_number in range(int(start * fps), int(end * fps), frame_step):
            second = frame_number / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            labels = []

            if _in_windows(second, DEFAULT_UAP_WINDOWS):
                blue = cv2.inRange(hsv, np.array([88, 65, 65]), np.array([128, 255, 255]))
                match = _best_contour(blue, min_area=120, max_area=35000)
                if match:
                    _, area, x, y, w, h, fill = match
                    labels.append((2, x, y, w, h, area, fill))

            if _in_windows(second, DEFAULT_UAI_WINDOWS):
                red = cv2.bitwise_or(
                    cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255])),
                    cv2.inRange(hsv, np.array([170, 70, 70]), np.array([179, 255, 255])),
                )
                match = _best_contour(red, min_area=120, max_area=25000)
                if match:
                    _, area, x, y, w, h, fill = match
                    labels.append((3, x, y, w, h, area, fill))

            if not labels:
                continue

            split = "valid" if index % 5 == 0 else "train"
            image_name = f"thyz2026_visible_f{frame_number:06d}.jpg"
            image_path = output_dir / split / "images" / image_name
            label_path = output_dir / split / "labels" / f"{Path(image_name).stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            yolo_lines = []
            record_labels = []
            for class_id, x, y, w, h, area, fill in labels:
                pad = 6
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(width - x, w + 2 * pad)
                h = min(height - y, h + 2 * pad)
                yolo_lines.append(
                    f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                    f"{w / width:.6f} {h / height:.6f}\n"
                )
                record_labels.append(
                    {"class_id": class_id, "xywh": [x, y, w, h], "area": area, "fill": fill}
                )
                counts["uap" if class_id == 2 else "uai"] += 1

            label_path.write_text("".join(yolo_lines), encoding="utf-8")
            records.append(
                {
                    "split": split,
                    "frame": frame_number,
                    "second": second,
                    "image": str(image_path),
                    "labels": record_labels,
                }
            )
            index += 1

    cap.release()
    counts["images"] = len(records)
    manifest = {
        "source": "official TEKNOFEST 2026 Drive sample video",
        "source_video": str(video_path),
        "fps": fps,
        "size": [width, height],
        "uap_windows_seconds": DEFAULT_UAP_WINDOWS,
        "uai_windows_seconds": DEFAULT_UAI_WINDOWS,
        "counts": counts,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_review_sheet(output_dir, records)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract high-confidence UAP/UAI YOLO candidates from the official 2026 sample video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("datasets/thyz_2026_sample/videos/THYZ_2026_Ornek_Veri_1.MP4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/THYZ_2026_UAP_UAI_auto_labeled"),
    )
    parser.add_argument("--frame-step", type=int, default=2)
    args = parser.parse_args()

    manifest = extract_candidates(args.video, args.output, frame_step=args.frame_step)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Output: {args.output}")
    print(f"Review sheet: {args.output / 'review' / 'candidate_labels_sheet.jpg'}")


if __name__ == "__main__":
    main()
