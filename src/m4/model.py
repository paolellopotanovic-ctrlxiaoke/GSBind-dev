"""Backward-compatible public imports for the exact M4 model."""

from .models import M4Model, build_m4_model, forward_m4_batch, load_m4_checkpoint

S1SurfaceFieldPredictor = M4Model

__all__ = [
    "M4Model",
    "S1SurfaceFieldPredictor",
    "build_m4_model",
    "forward_m4_batch",
    "load_m4_checkpoint",
]
