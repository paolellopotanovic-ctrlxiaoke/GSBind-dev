"""Exact M4 multi-objective loss, separated from the training engine."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .data.soft_labels import masked_soft_bce
from .training.objectives import (
    motif_patch_aux_bce_from_batch,
    pocket_mass_loss_from_batch,
    semantic_patch_losses,
    surface_soft_bce_from_batch,
)
from .training import teacher_logit_distillation_loss
from .training.runtime import masked_bce


@dataclass(frozen=True)
class M4LossConfig:
    base_aux_loss_weight: float = 0.2
    soft_residue_loss_weight: float = 0.0
    surface_soft_loss_weight: float = 0.0
    atom_pre_aux_loss_weight: float = 0.0
    motif_patch_aux_loss_weight: float = 0.0
    semantic_patch_amplitude_loss_weight: float = 0.0
    semantic_patch_set_loss_weight: float = 0.0
    semantic_patch_existence_loss_weight: float = 0.0
    semantic_patch_connectivity_loss_weight: float = 0.0
    pocket_mass_loss_weight: float = 0.0
    teacher_logit_weight: float = 0.0
    teacher_logit_loss: str = "mse"
    teacher_temperature: float = 1.0
    stochastic_head_mode: str = "none"
    surface_soft_sigma_tangent: float = 6.0
    surface_soft_sigma_normal: float = 3.0
    surface_soft_max_value: float = 1.0
    surface_soft_cutoff: float = 24.0
    surface_soft_label_mode: str = "global_max"
    surface_soft_label_topk: int = 16
    semantic_patch_mode: str = "none"
    semantic_patch_query_count: int = 8
    semantic_patch_active_topk: int = 4
    semantic_patch_component_k: int = 16
    semantic_patch_region_existence_threshold: float = 0.05
    pocket_mass_topk: int = 16
    poisson_lambda_max: float = 20.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "M4LossConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


class M4Objective(nn.Module):
    """Compute the exact weighted objective used by the Holo20k run."""

    def __init__(self, config: M4LossConfig, pos_weight: float):
        super().__init__()
        self.config = config
        self.args = config.namespace()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, output: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Tensor]:
        pos_weight = self.pos_weight.to(device=output["logits"].device, dtype=output["logits"].dtype)
        zero = output["logits"].sum() * 0.0
        final = masked_bce(output["logits"], batch["y"], batch["mask"], pos_weight)
        base = masked_bce(output["base_logits"], batch["y"], batch["mask"], pos_weight)
        prior_diagnostic = masked_bce(output["prior_logits"], batch["y"], batch["mask"], pos_weight)
        if batch.get("soft_y") is not None and self.config.soft_residue_loss_weight > 0:
            soft_residue = masked_soft_bce(output["logits"], batch["soft_y"], batch["mask"], pos_weight)
        else:
            soft_residue = zero
        surface_soft = surface_soft_bce_from_batch(output, batch, pos_weight, self.args)
        if output.get("atom_pre_logits") is not None and self.config.atom_pre_aux_loss_weight > 0:
            atom_pre = masked_bce(output["atom_pre_logits"], batch["y"], batch["mask"], pos_weight)
        else:
            atom_pre = zero
        motif_patch = motif_patch_aux_bce_from_batch(output, batch, pos_weight, self.args)
        semantic = semantic_patch_losses(output, batch, self.args)
        pocket_mass = pocket_mass_loss_from_batch(output, batch, pos_weight, self.args)
        distillation = teacher_logit_distillation_loss(
            output["logits"],
            batch.get("teacher_logits"),
            batch["mask"],
            weight=self.config.teacher_logit_weight,
            mode=self.config.teacher_logit_loss,
            temperature=self.config.teacher_temperature,
        )
        total = (
            final
            + self.config.base_aux_loss_weight * base
            + self.config.soft_residue_loss_weight * soft_residue
            + self.config.surface_soft_loss_weight * surface_soft
            + self.config.atom_pre_aux_loss_weight * atom_pre
            + self.config.motif_patch_aux_loss_weight * motif_patch
            + self.config.semantic_patch_amplitude_loss_weight * semantic["amplitude"]
            + self.config.semantic_patch_set_loss_weight * semantic["set"]
            + self.config.semantic_patch_existence_loss_weight * semantic["existence"]
            + self.config.semantic_patch_connectivity_loss_weight * semantic["connectivity"]
            + self.config.pocket_mass_loss_weight * pocket_mass
            + distillation
        )
        return {
            "loss": total,
            "final": final,
            "base": base,
            "prior_diagnostic": prior_diagnostic,
            "soft_residue": soft_residue,
            "surface_soft": surface_soft,
            "atom_pre": atom_pre,
            "motif_patch": motif_patch,
            "semantic_amplitude": semantic["amplitude"],
            "semantic_set": semantic["set"],
            "semantic_existence": semantic["existence"],
            "semantic_connectivity": semantic["connectivity"],
            "pocket_mass": pocket_mass,
            "distillation": distillation,
        }
