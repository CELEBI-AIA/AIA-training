import pytest
from pathlib import Path
from uav_training.audit import read_yaml, read_txt_classes

def test_read_yaml(tmp_path):
    d = tmp_path / "data.yaml"
    d.write_text("names: ['uai', 'vehicle']")
    data = read_yaml(str(d))
    assert 'names' in data
    assert 'uai' in data['names']

def test_read_txt_classes(tmp_path):
    d = tmp_path / "classes.txt"
    d.write_text("uai\nvehicle\nhuman")
    classes = read_txt_classes(str(d))
    assert classes == ["uai", "vehicle", "human"]
