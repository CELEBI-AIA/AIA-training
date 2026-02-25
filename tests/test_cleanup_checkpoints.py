"""Unit tests for scripts.cleanup_checkpoints."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cleanup_checkpoints import (  # noqa: E402
    cleanup_checkpoints,
    cleanup_run,
    KEEP_EPOCHS,
)


def test_cleanup_checkpoints_nonexistent_dir(tmp_path):
    """When weights_dir doesn't exist, returns 0. Uses tmp_path for cross-platform."""
    nonexistent = tmp_path / "nonexistent_weights_dir_xyz"
    assert not nonexistent.exists()
    assert cleanup_checkpoints(nonexistent) == 0


def test_cleanup_checkpoints_empty_dir(tmp_path):
    """When dir is empty, returns 0."""
    assert cleanup_checkpoints(tmp_path) == 0


def test_cleanup_checkpoints_keeps_recent(tmp_path):
    """Keeps KEEP_EPOCHS most recent epoch files, deletes older."""
    for i in range(5):
        (tmp_path / f"epoch{i}.pt").write_bytes(b"x")
    deleted = cleanup_checkpoints(tmp_path)
    assert deleted == 2  # 5 - 3 = 2
    remaining = list(tmp_path.glob("epoch*.pt"))
    assert len(remaining) == KEEP_EPOCHS


def test_cleanup_checkpoints_ignores_best_last(tmp_path):
    """best.pt and last.pt are not touched (only epoch*.pt)."""
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "last.pt").write_bytes(b"x")
    (tmp_path / "epoch0.pt").write_bytes(b"x")
    deleted = cleanup_checkpoints(tmp_path)
    assert deleted == 0
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()


def test_cleanup_run(tmp_path):
    """cleanup_run operates on weights/ subdir."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for i in range(5):
        (weights_dir / f"epoch{i}.pt").write_bytes(b"x")
    deleted = cleanup_run(tmp_path)
    assert deleted == 2
    assert len(list(weights_dir.glob("epoch*.pt"))) == KEEP_EPOCHS


def test_cleanup_run_no_weights_dir(tmp_path):
    """When weights/ doesn't exist, returns 0."""
    assert cleanup_run(tmp_path) == 0
