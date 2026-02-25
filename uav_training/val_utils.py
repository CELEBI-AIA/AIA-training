"""
Validation utilities — per-class AP50, temporal leakage check.
"""
from pathlib import Path

from config import IMAGE_EXTENSIONS

TARGET_THRESHOLDS = {
    "vehicle": (0.90, 0.95),
    "human": (0.88, 0.93),
    "uap": (0.85, 0.92),
    "uai": (0.85, 0.92),
}


def run_per_class_val(model_path: str, data_path: str, split: str = "val", verbose: bool = True) -> dict:
    """
    Run validation and return per-class AP50.
    Raises FileNotFoundError if dataset.yaml not found.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed") from None

    data_dir = Path(data_path)
    yaml_path = data_dir / "dataset.yaml" if data_dir.is_dir() else Path(data_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    model = YOLO(model_path)
    metrics = model.val(data=str(yaml_path), split=split, verbose=verbose)

    class_names = ["vehicle", "human", "uap", "uai"]
    maps = getattr(metrics.box, "maps", None)
    if maps is None:
        maps = getattr(metrics.box, "ap50", [0.0] * 4)
    if not hasattr(maps, "__iter__"):
        maps = [float(maps)] * 4

    result = dict(zip(class_names, [float(m) for m in maps[:4]]))
    return result


def print_per_class_report(result: dict) -> None:
    """Print per-class mAP50 with thresholds."""
    print("\n" + "=" * 50, flush=True)
    print("  PER-CLASS mAP50", flush=True)
    print("=" * 50, flush=True)
    for name, ap50 in result.items():
        min_ok, target = TARGET_THRESHOLDS.get(name, (0.0, 0.0))
        status = "OK" if ap50 >= min_ok else "LOW"
        if ap50 < min_ok:
            status += f" (min={min_ok})"
        print(f"  {name:<10}: {ap50:.4f}  [{status}]  (target: {target})", flush=True)
    print("=" * 50, flush=True)

    low = [n for n, v in result.items() if v < TARGET_THRESHOLDS.get(n, (0, 0))[0]]
    if low:
        print(f"\nUYARI: {', '.join(low)} hedefin altında.", flush=True)
        print("Phase 2 copy_paste: 0.5 aktif.", flush=True)
    else:
        print("\nTüm sınıflar minimum eşiğin üzerinde.", flush=True)


def check_temporal_leakage(dataset_dir) -> dict:
    """
    Build sonrası train/val overlap kontrolü.
    Returns: {"exact_match": n, "video_prefix_overlap": n, "train_stems": n, "val_stems": n}
    """
    dataset_dir = Path(dataset_dir)
    train_imgs = dataset_dir / "train" / "images"
    val_imgs = dataset_dir / "val" / "images"
    if not train_imgs.exists() or not val_imgs.exists():
        return {"exact_match": 0, "video_prefix_overlap": 0, "train_stems": 0, "val_stems": 0}

    train_stems = {p.stem for p in train_imgs.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    val_stems = {p.stem for p in val_imgs.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    exact_match = len(train_stems & val_stems)

    def get_prefix(s: str) -> str:
        parts = s.rsplit("_frame_", 1)
        return parts[0] if len(parts) > 1 else s

    train_prefixes = {get_prefix(s) for s in train_stems}
    val_prefixes = {get_prefix(s) for s in val_stems}
    video_overlap = len(train_prefixes & val_prefixes)

    return {
        "exact_match": exact_match,
        "video_prefix_overlap": video_overlap,
        "train_stems": len(train_stems),
        "val_stems": len(val_stems),
    }
