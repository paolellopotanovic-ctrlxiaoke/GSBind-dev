"""Public M4 model construction and batch-level forward contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .._bootstrap import activate_validated_implementation

activate_validated_implementation()

from .components import S1SurfaceFieldPredictor  # noqa: E402
from .config import build_model_from_checkpoint_args  # noqa: E402


M4Model = S1SurfaceFieldPredictor


def build_m4_model(config: Mapping[str, Any]) -> M4Model:
    """Build the modular M4 model from training or checkpoint arguments."""

    return build_model_from_checkpoint_args(dict(config))


def forward_m4_batch(model: M4Model, batch: Mapping[str, Any]) -> dict[str, Tensor | Any]:
    """Run M4 on one collated batch.

    Training uses the original dense atom-to-residue assignment because the
    validated implementation's autograd path was trained that way. Evaluation
    and prediction use the memory-efficient index representation plus static
    geometry caches. Keeping this switch here prevents silent differences
    between the three entry points.
    """

    if model.training:
        atom_index = batch["atom_to_residue_index"].to(
            device=batch["q"].device,
            dtype=torch.long,
        )
        num_residues = int(batch["residue_counts"].sum().item())
        dense_assignment = batch["q"].new_zeros((atom_index.shape[0], num_residues))
        dense_assignment.scatter_(1, atom_index.unsqueeze(1), 1.0)
        return model(
            batch["X"],
            None,
            batch["q"],
            dense_assignment,
            batch["atom_counts"],
            batch["residue_counts"],
            residue_graph_caches=batch.get("residue_graph_caches"),
            s1_surface_caches=batch.get("s1_surface_caches"),
            true_residue_features=batch.get("true_residue_features"),
            residue_chain_ids=batch.get("residue_chain_ids"),
            teacher_hidden=batch.get("teacher_hidden"),
            atom_to_residue_index=None,
            s1_static_geometry_caches=None,
        )
    return model(
        batch["X"],
        None,
        batch["q"],
        None,
        batch["atom_counts"],
        batch["residue_counts"],
        residue_graph_caches=batch.get("residue_graph_caches"),
        s1_surface_caches=batch.get("s1_surface_caches"),
        true_residue_features=batch.get("true_residue_features"),
        residue_chain_ids=batch.get("residue_chain_ids"),
        teacher_hidden=batch.get("teacher_hidden"),
        atom_to_residue_index=batch["atom_to_residue_index"],
        s1_static_geometry_caches=batch.get("s1_static_geometry_caches"),
    )


def load_m4_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[M4Model, dict[str, Any]]:
    """Strictly load a released or training-time M4 checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("M4 checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict")
    args = checkpoint.get("args")
    if not isinstance(state, dict) or not isinstance(args, dict):
        raise ValueError("M4 checkpoint requires model_state_dict and args mappings")
    model = build_m4_model(args)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict M4 checkpoint load failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model, checkpoint
