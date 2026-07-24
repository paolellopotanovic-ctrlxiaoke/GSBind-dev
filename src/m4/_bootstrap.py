from __future__ import annotations

import sys
from pathlib import Path


def activate_validated_implementation() -> Path:
    """Expose the byte-validated implementation with its original import layout."""
    legacy_root = Path(__file__).resolve().parent / "legacy_repo"
    model_root = legacy_root / "model"
    for path in (legacy_root, model_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return legacy_root
