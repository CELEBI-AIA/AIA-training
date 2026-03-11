#!/usr/bin/env python3
"""
Checkpoint temizleme - best.pt, last.pt ve son 3 epoch checkpoint disindakileri siler.
Cross-platform (Windows/Linux/Colab). Otomatik olarak train pipeline tarafindan cagrilir.
"""
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_training.emoji_logs import install_emoji_print  # noqa: E402

install_emoji_print(globals())

KEEP_EPOCHS = 3  # best + last + son 3 epoch*.pt


def cleanup_checkpoints(weights_dir: Path) -> int:
    """
    Eski epoch checkpoint'lerini sil. best.pt, last.pt ve son KEEP_EPOCHS epoch*.pt kalir.
    Returns: silinen dosya sayisi
    """
    if not weights_dir.exists() or not weights_dir.is_dir():
        return 0

    epoch_files = sorted(
        weights_dir.glob("epoch*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    to_keep = set(epoch_files[:KEEP_EPOCHS])
    to_delete = [f for f in epoch_files if f not in to_keep]

    deleted = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def cleanup_run(run_dir: Path) -> int:
    """Bir run klasorundeki weights/ altindaki checkpoint'leri temizle."""
    weights_dir = run_dir / "weights"
    return cleanup_checkpoints(weights_dir)


def cleanup_all_runs(project_dir: Path) -> int:
    """Proje altindaki tum run klasorlerini temizle."""
    total = 0
    if not project_dir.exists():
        return 0
    for run_dir in project_dir.iterdir():
        if run_dir.is_dir():
            n = cleanup_run(run_dir)
            if n > 0:
                total += n
                print(f"  [CLEANUP] {run_dir.name}: {n} epoch checkpoint silindi", flush=True)
    return total


if __name__ == "__main__":
    from uav_training.config import TRAIN_CONFIG, is_colab

    # Colab: /content/runs; local: TRAIN_CONFIG["project"]
    project = Path("/content/runs") if is_colab() else Path(TRAIN_CONFIG.get("project", "."))
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            project = target
    n = cleanup_all_runs(project)
    print(f"Toplam {n} checkpoint silindi.", flush=True)
