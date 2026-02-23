import pytest
import sys
from pathlib import Path

# Add uav_training to path for testing
sys.path.append(str(Path(__file__).parent.parent / "uav_training"))

from audit import read_txt_classes, read_yaml

def test_read_txt_classes_success(tmp_path):
    # Setup dummy file
    test_file = tmp_path / "classes.txt"
    test_file.write_text("car\nperson\nbicycle\n")
    
    # Execute
    classes = read_txt_classes(test_file)
    
    # Assert
    assert classes == ["car", "person", "bicycle"]
    assert len(classes) == 3

def test_read_txt_classes_missing_file():
    classes = read_txt_classes(Path("does_not_exist_file.txt"))
    assert classes == []

def test_read_yaml_missing_file():
    yaml_data = read_yaml(Path("does_not_exist_file.yaml"))
    assert yaml_data is None
