import sys
from pathlib import Path

import pytest

# Add gps_training to path for testing
sys.path.append(str(Path(__file__).parent.parent / "gps_training"))

from train import resolve_scheduler_max_lr


def test_resolve_scheduler_max_lr_defaults_to_learning_rate():
    config = {"learning_rate": 1e-4}
    assert resolve_scheduler_max_lr(config) == pytest.approx(1e-4)


def test_resolve_scheduler_max_lr_uses_config_value():
    config = {"learning_rate": 1e-4, "max_lr": 3e-4}
    assert resolve_scheduler_max_lr(config) == pytest.approx(3e-4)


def test_resolve_scheduler_max_lr_rejects_lower_than_learning_rate():
    config = {"learning_rate": 1e-4, "max_lr": 5e-5}
    with pytest.raises(ValueError, match="max_lr"):
        resolve_scheduler_max_lr(config)
