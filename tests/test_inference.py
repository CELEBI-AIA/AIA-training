"""Smoke tests for uav_training.inference — import and structure only, no model load."""
import pytest

# Skip if ultralytics not installed (e.g. minimal test env)
pytest.importorskip("ultralytics")


def test_inference_module_imports():
    """Inference module imports without error."""
    from uav_training import inference
    assert hasattr(inference, "smoke_infer")
    assert hasattr(inference, "DEFAULT_INFER_SOURCE")
    assert "val" in inference.DEFAULT_INFER_SOURCE or "images" in inference.DEFAULT_INFER_SOURCE
