"""Unit tests for uav_training.val_utils."""
import pytest
from pathlib import Path

from uav_training.val_utils import (
    check_temporal_leakage,
    print_per_class_report,
    TARGET_THRESHOLDS,
)


def test_check_temporal_leakage_empty_dirs(tmp_path):
    """When train/val images dirs don't exist, returns zeros."""
    result = check_temporal_leakage(tmp_path)
    assert result["exact_match"] == 0
    assert result["video_prefix_overlap"] == 0
    assert result["train_stems"] == 0
    assert result["val_stems"] == 0


def test_check_temporal_leakage_no_overlap(tmp_path):
    """When train and val have different stems, no exact match."""
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "a1.jpg").write_bytes(b"x")
    (tmp_path / "train" / "images" / "a2.jpg").write_bytes(b"x")
    (tmp_path / "val" / "images" / "b1.jpg").write_bytes(b"x")
    result = check_temporal_leakage(tmp_path)
    assert result["exact_match"] == 0
    assert result["train_stems"] == 2
    assert result["val_stems"] == 1


def test_check_temporal_leakage_with_overlap(tmp_path):
    """When train and val share a stem, exact_match > 0."""
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "shared.jpg").write_bytes(b"x")
    (tmp_path / "val" / "images" / "shared.jpg").write_bytes(b"x")
    result = check_temporal_leakage(tmp_path)
    assert result["exact_match"] == 1
    assert result["train_stems"] == 1
    assert result["val_stems"] == 1


def test_check_temporal_leakage_video_prefix(tmp_path):
    """Video prefix overlap (frame_xxx) is detected."""
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "vid1_frame_001.jpg").write_bytes(b"x")
    (tmp_path / "val" / "images" / "vid1_frame_002.jpg").write_bytes(b"x")
    result = check_temporal_leakage(tmp_path)
    assert result["exact_match"] == 0
    assert result["video_prefix_overlap"] == 1


def test_print_per_class_report_no_raise(capsys):
    """print_per_class_report runs without error and prints."""
    result = {"vehicle": 0.92, "human": 0.90, "uap": 0.88, "uai": 0.85}
    print_per_class_report(result)
    out = capsys.readouterr().out
    assert "PER-CLASS" in out or "mAP50" in out
    assert "vehicle" in out


def test_target_thresholds_has_all_classes():
    """TARGET_THRESHOLDS includes vehicle, human, uap, uai."""
    assert "vehicle" in TARGET_THRESHOLDS
    assert "human" in TARGET_THRESHOLDS
    assert "uap" in TARGET_THRESHOLDS
    assert "uai" in TARGET_THRESHOLDS
