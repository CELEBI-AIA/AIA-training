import pytest
from uav_training.build_dataset import resolve_target_split, _assert_non_empty_output


def test_resolve_target_split_default_policy():
    assert resolve_target_split("train", include_test_in_val=False) == "train"
    assert resolve_target_split("valid", include_test_in_val=False) == "val"
    assert resolve_target_split("val", include_test_in_val=False) == "val"
    assert resolve_target_split("test", include_test_in_val=False) == "test"


def test_resolve_target_split_include_test_in_val():
    assert resolve_target_split("test", include_test_in_val=True) == "val"


def test_resolve_target_split_unknown_raises_error():
    with pytest.raises(ValueError, match="Unknown split name 'custom_split'. Cannot safely map to target split."):
        resolve_target_split("custom_split", include_test_in_val=False)


def test_assert_non_empty_output_raises_when_train_val_empty(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "test" / "images").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Dataset build produced empty output"):
        _assert_non_empty_output(tmp_path, tmp_path)


def test_assert_non_empty_output_passes_with_train_and_val_images(tmp_path):
    train_images = tmp_path / "train" / "images"
    val_images = tmp_path / "val" / "images"
    test_images = tmp_path / "test" / "images"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (train_images / "a.jpg").write_bytes(b"x")
    (val_images / "b.jpg").write_bytes(b"x")
    _assert_non_empty_output(tmp_path, tmp_path)
