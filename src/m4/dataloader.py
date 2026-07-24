"""Stable public imports for M4 datasets and batch collation."""

from ._bootstrap import activate_validated_implementation

activate_validated_implementation()

from apex_dataset import read_manifest  # noqa: E402
from apex_s1_surface_field.dataset import (  # noqa: E402
    S1SurfaceFieldDataset,
    collate_s1_surface_field_batch,
)

__all__ = [
    "S1SurfaceFieldDataset",
    "collate_s1_surface_field_batch",
    "read_manifest",
]
