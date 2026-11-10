"""Stable project paths, independent of the process working directory."""
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent)


def setup_project_path():
    """Explicit compatibility helper; normal module imports need no mutation."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    return PROJECT_ROOT


def get_project_root():
    """Return the repository root without changing the import search path."""
    return PROJECT_ROOT
