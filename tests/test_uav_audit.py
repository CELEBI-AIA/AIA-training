from pathlib import Path

from uav_training.audit import (
    _compute_split_overlap,
    audit_directory,
    read_txt_classes,
    read_yaml,
)

def test_read_txt_classes_success(tmp_path):
    # Setup dummy file
    test_file = tmp_path / "classes.txt"
    test_file.write_text("car\nperson\nbicycle\n")
    
    # Execute
    classes = read_txt_classes(test_file)
    
    # Assert
    assert classes == ["car", "person", "bicycle"]
    assert len(classes) == 3

def test_read_txt_classes_missing_file(tmp_path):
    """Missing file returns []. Uses tmp_path for cross-platform path."""
    missing = tmp_path / "does_not_exist_file.txt"
    assert not missing.exists()
    classes = read_txt_classes(missing)
    assert classes == []


def test_read_yaml_missing_file(tmp_path):
    """Missing file returns None. Uses tmp_path for cross-platform path."""
    missing = tmp_path / "does_not_exist_file.yaml"
    assert not missing.exists()
    yaml_data = read_yaml(missing)
    assert yaml_data is None


def test_compute_split_overlap_no_overlap():
    split_stems = {
        "train": {"a_1", "a_2"},
        "val": {"b_1"},
        "test": {"c_1"},
    }
    overlap = _compute_split_overlap(split_stems)
    assert overlap["has_overlap"] is False
    assert overlap["train_val_overlap"] == 0
    assert overlap["train_test_overlap"] == 0
    assert overlap["val_test_overlap"] == 0


def test_audit_directory_reports_split_counts_and_overlap(tmp_path):
    dataset = tmp_path / "demo_ds"
    (dataset / "train" / "images").mkdir(parents=True)
    (dataset / "train" / "labels").mkdir(parents=True)
    (dataset / "val" / "images").mkdir(parents=True)
    (dataset / "val" / "labels").mkdir(parents=True)
    (dataset / "test" / "images").mkdir(parents=True)
    (dataset / "test" / "labels").mkdir(parents=True)

    (dataset / "data.yaml").write_text("names: ['vehicle', 'human', 'uap', 'uai']\n")

    for idx in range(10):
        stem = f"shared_{idx}" if idx == 0 else f"train_{idx}"
        (dataset / "train" / "images" / f"{stem}.jpg").write_text("img")
        (dataset / "train" / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    for idx in range(10):
        stem = "shared_0" if idx == 0 else f"val_{idx}"
        (dataset / "val" / "images" / f"{stem}.jpg").write_text("img")
        (dataset / "val" / "labels" / f"{stem}.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    for idx in range(10):
        stem = f"test_{idx}"
        (dataset / "test" / "images" / f"{stem}.jpg").write_text("img")
        (dataset / "test" / "labels" / f"{stem}.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    result = audit_directory(dataset)

    assert result["split_counts"]["train"]["images"] == 10
    assert result["split_counts"]["val"]["images"] == 10
    assert result["split_counts"]["test"]["images"] == 10
    assert result["split_overlap"]["has_overlap"] is True
    assert result["split_overlap"]["train_val_overlap"] == 1
