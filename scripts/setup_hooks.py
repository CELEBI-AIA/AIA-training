#!/usr/bin/env python3
"""
One-time setup: install pre-commit hooks so tests run automatically before each commit.
Run once: python scripts/setup_hooks.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from uav_training.emoji_logs import install_emoji_print  # noqa: E402
    install_emoji_print(globals())
except Exception:
    pass


def main():
    print("Installing pre-commit hooks (tests run automatically on git commit)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "pre-commit", "-q"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("Failed to install pre-commit")
        sys.exit(1)
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "install"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("Failed to install hooks")
        sys.exit(1)
    print("Done. Tests will run automatically before each commit.")


if __name__ == "__main__":
    main()
