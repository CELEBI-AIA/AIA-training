#!/usr/bin/env python3
"""
Run all tests — single entry point for pytest.
Works from local dev and Colab (/content/repo).

Usage:
  python scripts/run_all_tests.py
  python scripts/run_all_tests.py -x   # stop on first failure
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "--tb=short", "tests/"] + sys.argv[1:]))
