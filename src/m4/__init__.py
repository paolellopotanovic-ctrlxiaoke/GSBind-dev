"""Public API for the M4 protein interface model."""

from importlib import import_module

__all__ = [
    "M4Model",
    "S1SurfaceFieldDataset",
    "S1SurfaceFieldPredictor",
    "collate_s1_surface_field_batch",
]


def __getattr__(name: str):
    if name in {"M4Model", "S1SurfaceFieldPredictor"}:
        return getattr(import_module(".model", __name__), name)
    if name in {"S1SurfaceFieldDataset", "collate_s1_surface_field_batch"}:
        return getattr(import_module(".dataloader", __name__), name)
    raise AttributeError(name)
