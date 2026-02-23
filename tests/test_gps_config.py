import pytest
import sys
from pathlib import Path

# Add gps_training to path for testing
sys.path.append(str(Path(__file__).parent.parent / "gps_training"))

from config import TRAIN_CONFIG, is_colab

def test_train_config_defaults():
    # Assert critical keys exist
    assert "epochs" in TRAIN_CONFIG
    assert "batch_size" in TRAIN_CONFIG
    assert "img_size" in TRAIN_CONFIG
    
    # Assert types and expected default formats
    assert isinstance(TRAIN_CONFIG["epochs"], int)
    assert len(TRAIN_CONFIG["img_size"]) == 2

def test_is_colab_detection(monkeypatch):
    # Test positive colab detection via env var
    monkeypatch.setenv("COLAB_RELEASE_TAG", "1")
    assert is_colab() is True
    
    # Cleanup env
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
