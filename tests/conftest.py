"""
Pytest conftest — ensures project root is on sys.path for uav_training imports.
Works from local dev and Colab (/content/repo).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
