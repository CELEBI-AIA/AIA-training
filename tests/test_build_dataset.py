import pytest
from uav_training.build_dataset import resolve_target_split


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
