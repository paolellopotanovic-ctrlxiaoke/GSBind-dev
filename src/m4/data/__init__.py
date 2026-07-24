"""Manifest datasets and optional Lightning data orchestration."""

from .dataset import S1SurfaceFieldDataset, collate_s1_surface_field_batch
from .loader import build_loader
from .manifest import read_manifest

__all__ = [
    "M4DataConfig",
    "M4DataModule",
    "S1SurfaceFieldDataset",
    "build_loader",
    "collate_s1_surface_field_batch",
    "read_manifest",
]


def __getattr__(name):
    if name in {"M4DataConfig", "M4DataModule"}:
        from .datamodule import M4DataConfig, M4DataModule

        return {"M4DataConfig": M4DataConfig, "M4DataModule": M4DataModule}[name]
    raise AttributeError(name)
