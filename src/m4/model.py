"""Stable public import for the exact M4 model implementation."""

from ._bootstrap import activate_validated_implementation

activate_validated_implementation()

from apex_s1_surface_field.model import S1SurfaceFieldPredictor  # noqa: E402

__all__ = ["S1SurfaceFieldPredictor"]
