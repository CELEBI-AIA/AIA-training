"""Unit tests for uav_training.config — no torch/CUDA loading."""
import pytest

from uav_training.config import (
    TARGET_CLASSES,
    IMAGE_EXTENSIONS,
    TRAIN_CONFIG,
    is_colab,
    PROJECT_ROOT,
    ARTIFACTS_DIR,
    _find_nested_subdir,
    _looks_like_train_data_dir,
)


def test_is_colab_returns_false_locally():
    """When not in Colab, is_colab() returns False. Skip when actually on Colab."""
    if is_colab():
        pytest.skip("Running on Colab — is_colab() is True here")
    assert is_colab() is False


def test_is_colab_with_env(monkeypatch):
    """When COLAB_RELEASE_TAG is set, is_colab() returns True."""
    monkeypatch.setenv("COLAB_RELEASE_TAG", "1.0")
    assert is_colab() is True


def test_target_classes_canonical_mapping():
    """TARGET_CLASSES maps vehicle=0, human=1, uap=2, uai=3."""
    assert TARGET_CLASSES["vehicle"] == 0
    assert TARGET_CLASSES["car"] == 0
    assert TARGET_CLASSES["human"] == 1
    assert TARGET_CLASSES["person"] == 1
    assert TARGET_CLASSES["uap"] == 2
    assert TARGET_CLASSES["uai"] == 3


def test_image_extensions_contains_common_formats():
    """IMAGE_EXTENSIONS includes jpg, png, webp, etc."""
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS
    assert ".webp" in IMAGE_EXTENSIONS


def test_train_config_has_required_keys():
    """TRAIN_CONFIG contains essential training parameters."""
    assert "epochs" in TRAIN_CONFIG
    assert "batch" in TRAIN_CONFIG
    assert "imgsz" in TRAIN_CONFIG
    assert "model" in TRAIN_CONFIG
    assert "project" in TRAIN_CONFIG
    assert "name" in TRAIN_CONFIG


def test_train_config_name_contains_version():
    """TRAIN_CONFIG name includes version string."""
    name = TRAIN_CONFIG["name"]
    assert "uav" in name.lower()
    assert "v" in name.lower()


def test_project_root_exists():
    """PROJECT_ROOT is a valid path."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_artifacts_dir_under_project():
    """ARTIFACTS_DIR is under PROJECT_ROOT. Cross-platform path check."""
    art = ARTIFACTS_DIR.resolve()
    root = PROJECT_ROOT.resolve()
    assert art == root or root in art.parents


def test_looks_like_train_data_dir_detects_known_children(tmp_path):
    root = tmp_path / "TRAIN_DATA"
    (root / "UAI_UAP").mkdir(parents=True)
    assert _looks_like_train_data_dir(root) is True


def test_find_nested_subdir_finds_train_data_by_depth(tmp_path):
    nested = tmp_path / "dataset_extracted" / "TRAIN_DATA"
    (nested / "megaset").mkdir(parents=True)
    found = _find_nested_subdir(tmp_path, "TRAIN_DATA", max_depth=4)
    assert found == nested


def test_direct_root_layout_is_detected(tmp_path):
    (tmp_path / "UAI_UAP").mkdir(parents=True)
    (tmp_path / "megaset").mkdir(parents=True)
    assert _looks_like_train_data_dir(tmp_path) is True
