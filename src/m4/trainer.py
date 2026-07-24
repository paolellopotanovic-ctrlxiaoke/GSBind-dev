"""Training entry point backed by the validated M4 implementation."""

from ._bootstrap import activate_validated_implementation

activate_validated_implementation()

from apex_s1_surface_field.train_s1_surface_field import main  # noqa: E402

__all__ = ["main"]
