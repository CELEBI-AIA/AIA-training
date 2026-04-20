"""
Validation utilities - per-class AP50, temporal leakage check.
"""
from pathlib import Path

from uav_training.config import IMAGE_EXTENSIONS
TARGET_THRESHOLDS = {
    "vehicle": (0.90, 0.95),
    "human": (0.88, 0.93),
    "uap": (0.85, 0.92),
    "uai": (0.85, 0.92),
}

def run_per_class_val(model_path: str, data_path: str, split: str = "val", verbose: bool = True) -> dict:
    """
    Run validation and return per-class AP50 and AP50-95.
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
    ap50 = getattr(metrics.box, "ap50", None)
    ap50_95 = getattr(metrics.box, "maps", None)

    if ap50 is None:
        ap50 = [0.0] * 4
    if ap50_95 is None:
        ap50_95 = [0.0] * 4

    if not hasattr(ap50, "__iter__"):
        ap50 = [float(ap50)] * 4
    if not hasattr(ap50_95, "__iter__"):
        ap50_95 = [float(ap50_95)] * 4

    result = {
        cls: {
            "ap50": float(ap50[idx]) if idx < len(ap50) else 0.0,
            "ap50_95": float(ap50_95[idx]) if idx < len(ap50_95) else 0.0,
        }
        for idx, cls in enumerate(class_names)
    }
    return result

def print_per_class_report(result: dict) -> None:
    """Print per-class mAP50 and mAP50-95 with thresholds."""
    print("\n" + "=" * 50, flush=True)
    print("  📊 PER-CLASS mAP (AP50 / AP50-95)", flush=True)
    print("=" * 50, flush=True)
    ap50_by_class = {}
    for name, value in result.items():
        if isinstance(value, dict):
            ap50 = float(value.get("ap50", 0.0))
            ap50_95 = float(value.get("ap50_95", 0.0))
        else:
            # Backward compatibility with older tests/callers.
            ap50 = float(value)
            ap50_95 = 0.0
        ap50_by_class[name] = ap50
        min_ok, target = TARGET_THRESHOLDS.get(name, (0.0, 0.0))
        status = "✅ OK" if ap50 >= min_ok else "LOW"
        if ap50 < min_ok:
            status += f" (min={min_ok})"
        print(
            f"  {name:<10}: AP50={ap50:.4f} | AP50-95={ap50_95:.4f}  "
            f"[{status}]  (target AP50: {target})",
            flush=True,
        )
    print("=" * 50, flush=True)

    low = [n for n, v in ap50_by_class.items() if v < TARGET_THRESHOLDS.get(n, (0, 0))[0]]
    if low:
        print(f"\nUYARI: {', '.join(low)} hedefin altinda.", flush=True)
        print("\nPhase 2'de dinamik augmentasyon (copy_paste) kullanilabilir.", flush=True)
    else:
        print("\nTum siniflar minimum esigin uzerinde.", flush=True)

def check_temporal_leakage(dataset_dir) -> dict:
    """
    Build sonrasi train/val overlap kontrolu.
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
