import pytest
from pathlib import Path
# A dummy test to ensure pytest hooks look at the right directories
def test_dummy_builder():
    from uav_training.build_dataset import build_dataset
    # We do not execute it to prevent wiping real artifacts during unit test,
    # but asserting it's importable.
    assert callable(build_dataset)
