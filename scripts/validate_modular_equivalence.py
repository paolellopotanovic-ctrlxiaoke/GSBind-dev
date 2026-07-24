"""Validate the modular M4 implementation against the frozen research oracle."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from m4._bootstrap import activate_validated_implementation
from m4.losses import M4LossConfig, M4Objective
from m4.models import build_m4_model, forward_m4_batch

activate_validated_implementation()

from apex_dataset import read_manifest  # noqa: E402
from apex_s1_surface_field.eval_s1_surface_field import (  # noqa: E402
    build_loader,
    build_model_from_ckpt_args as build_oracle_model,
)
from apex_s1_surface_field.soft_labels import masked_soft_bce  # noqa: E402
from apex_s1_surface_field.train_s1_surface_field import (  # noqa: E402
    motif_patch_aux_bce_from_batch,
    pocket_mass_loss_from_batch,
    semantic_patch_losses,
    surface_soft_bce_from_batch,
)
from apex_teacher_prior.modules import teacher_logit_distillation_loss  # noqa: E402
from train_apex import estimate_pos_weight  # noqa: E402
from train_apex_simple import masked_bce, move_batch_to_device  # noqa: E402


def loader_args(parsed: argparse.Namespace, checkpoint_args: Mapping[str, Any]) -> SimpleNamespace:
    def value(name: str, default: Any) -> Any:
        result = checkpoint_args.get(name, default)
        return default if result is None else result

    return SimpleNamespace(
        batch_size=parsed.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        allow_residue_mismatch=False,
        tensor_cache_dir=parsed.tensor_cache_dir,
        residue_graph_cache_dir=parsed.residue_graph_cache_dir,
        residue_graph_cache_ks=tuple(value("residue_graph_cache_ks", (16, 24, 32))),
        true_feature_cache_dir=str(value("true_feature_cache_dir", "")),
        require_true_features=bool(value("require_true_features", False)),
        true_feature_version=str(value("true_feature_version", "true_sasa_dssp_v1")),
        s1_surface_cache_dir=parsed.s1_surface_cache_dir,
        require_s1_surface_cache=True,
        disable_s1_surface_on_the_fly=True,
        s1_surface_points=int(value("s1_surface_points", 1024)),
        s1_surface_graph_k=int(value("s1_surface_graph_k", 16)),
        s1_surface_lappe_dim=int(value("s1_surface_lappe_dim", 0)),
        s1_surface_cache_version=str(value("s1_surface_cache_version", "s1_surface_shrake_v1")),
        teacher_cache_dir=str(value("teacher_cache_dir", "")),
        teacher_name=str(value("teacher_name", "")),
        teacher_cache_version=str(value("teacher_cache_version", "teacher_prior_v1")),
        require_teacher_cache=bool(value("require_teacher_cache", False)),
        s1_static_geometry_cache_dir=parsed.s1_static_geometry_cache_dir,
        s1_static_geometry_cache_version=str(
            value("s1_static_geometry_cache_version", "e89_dense_geometry_v1")
        ),
        require_s1_static_geometry_cache=True,
    )


def set_epoch_state(model: torch.nn.Module, checkpoint_args: Mapping[str, Any], epoch_index: int) -> None:
    epoch_seed = int(checkpoint_args.get("seed", 42)) + int(epoch_index)
    random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(epoch_seed)
    semantic_head = getattr(model, "semantic_patch_head", None)
    if semantic_head is not None and hasattr(semantic_head, "set_boundary_refinement_scale"):
        start = int(checkpoint_args.get("semantic_patch_boundary_start_epoch", 1))
        semantic_head.set_boundary_refinement_scale(1.0 if epoch_index + 1 >= start else 0.0)


def oracle_objective(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    pos_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    config = M4LossConfig.from_mapping(checkpoint_args)
    args = config.namespace()
    zero = output["logits"].sum() * 0.0
    final = masked_bce(output["logits"], batch["y"], batch["mask"], pos_weight)
    base = masked_bce(output["base_logits"], batch["y"], batch["mask"], pos_weight)
    prior = masked_bce(output["prior_logits"], batch["y"], batch["mask"], pos_weight)
    if batch.get("soft_y") is not None and config.soft_residue_loss_weight > 0:
        soft_residue = masked_soft_bce(output["logits"], batch["soft_y"], batch["mask"], pos_weight)
    else:
        soft_residue = zero
    surface_soft = surface_soft_bce_from_batch(output, batch, pos_weight, args)
    if output.get("atom_pre_logits") is not None and config.atom_pre_aux_loss_weight > 0:
        atom_pre = masked_bce(output["atom_pre_logits"], batch["y"], batch["mask"], pos_weight)
    else:
        atom_pre = zero
    motif_patch = motif_patch_aux_bce_from_batch(output, batch, pos_weight, args)
    semantic = semantic_patch_losses(output, batch, args)
    pocket_mass = pocket_mass_loss_from_batch(output, batch, pos_weight, args)
    distillation = teacher_logit_distillation_loss(
        output["logits"],
        batch.get("teacher_logits"),
        batch["mask"],
        weight=config.teacher_logit_weight,
        mode=config.teacher_logit_loss,
        temperature=config.teacher_temperature,
    )
    total = (
        final
        + config.base_aux_loss_weight * base
        + config.soft_residue_loss_weight * soft_residue
        + config.surface_soft_loss_weight * surface_soft
        + config.atom_pre_aux_loss_weight * atom_pre
        + config.motif_patch_aux_loss_weight * motif_patch
        + config.semantic_patch_amplitude_loss_weight * semantic["amplitude"]
        + config.semantic_patch_set_loss_weight * semantic["set"]
        + config.semantic_patch_existence_loss_weight * semantic["existence"]
        + config.semantic_patch_connectivity_loss_weight * semantic["connectivity"]
        + config.pocket_mass_loss_weight * pocket_mass
        + distillation
    )
    return {
        "loss": total,
        "final": final,
        "base": base,
        "prior_diagnostic": prior,
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


def tensor_differences(reference: Any, candidate: Any, prefix: str = "output") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if torch.is_tensor(reference):
        if not torch.is_tensor(candidate) or reference.shape != candidate.shape:
            raise AssertionError(f"Tensor contract mismatch at {prefix}")
        delta = (reference.detach().float() - candidate.detach().float()).abs()
        rows.append(
            {
                "name": prefix,
                "shape": list(reference.shape),
                "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
                "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
            }
        )
    elif isinstance(reference, Mapping):
        if set(reference) != set(candidate):
            raise AssertionError(f"Mapping keys differ at {prefix}")
        for key in reference:
            rows.extend(tensor_differences(reference[key], candidate[key], f"{prefix}.{key}"))
    elif isinstance(reference, (list, tuple)):
        if len(reference) != len(candidate):
            raise AssertionError(f"Sequence length differs at {prefix}")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            rows.extend(tensor_differences(left, right, f"{prefix}[{index}]"))
    elif reference is None or isinstance(reference, (str, int, float, bool)):
        if reference != candidate:
            raise AssertionError(f"Value differs at {prefix}: {reference!r} != {candidate!r}")
    return rows


def assert_rows_close(rows: list[dict[str, Any]], atol: float, label: str) -> dict[str, Any]:
    worst = max(rows, key=lambda row: row["max_abs"], default={"name": label, "max_abs": 0.0})
    if float(worst["max_abs"]) > atol:
        raise AssertionError(f"{label} exceeds atol={atol}: {worst}")
    return {"tensor_count": len(rows), "worst": worst, "atol": atol, "pass": True}


def compare_parameter_tensors(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    attribute: str,
) -> list[dict[str, Any]]:
    rows = []
    reference_named = dict(reference.named_parameters())
    candidate_named = dict(candidate.named_parameters())
    if list(reference_named) != list(candidate_named):
        raise AssertionError(f"Parameter names differ for {attribute}")
    for name, reference_parameter in reference_named.items():
        candidate_parameter = candidate_named[name]
        left = reference_parameter if attribute == "data" else reference_parameter.grad
        right = candidate_parameter if attribute == "data" else candidate_parameter.grad
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"Gradient presence differs for {name}")
            continue
        rows.extend(tensor_differences(left, right, f"{attribute}.{name}"))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", required=True)
    parser.add_argument("--s1-surface-cache-dir", required=True)
    parser.add_argument("--s1-static-geometry-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--epoch-index", type=int, default=3)
    parser.add_argument("--forward-atol", type=float, default=1e-7)
    parser.add_argument("--gradient-atol", type=float, default=1e-6)
    parser.add_argument("--optimizer-atol", type=float, default=3e-6)
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    device = torch.device(parsed.device)
    checkpoint = torch.load(parsed.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint["args"]
    rows = read_manifest(parsed.manifest)[: parsed.max_cases]
    pos_weight_value, label_stats = estimate_pos_weight(
        read_manifest(parsed.manifest),
        factor=float(checkpoint_args.get("pos_weight_factor", 0.5)),
    )
    batch = next(iter(build_loader(rows, loader_args(parsed, checkpoint_args))))
    if batch.get("skip_batch") or int(batch.get("skipped_count", 0) or 0):
        raise RuntimeError(f"Real-cache batch skipped cases: {batch.get('skip_reasons', [])}")
    batch = move_batch_to_device(batch, device)

    oracle = build_oracle_model(checkpoint_args).to(device)
    modular = build_m4_model(checkpoint_args).to(device)
    oracle.load_state_dict(checkpoint["model_state_dict"], strict=True)
    modular.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if list(oracle.state_dict()) != list(modular.state_dict()):
        raise AssertionError("Oracle and modular state_dict keys differ")

    oracle.eval()
    modular.eval()
    with torch.inference_mode():
        oracle_eval = forward_m4_batch(oracle, batch)
        modular_eval = forward_m4_batch(modular, batch)
    eval_check = assert_rows_close(
        tensor_differences(oracle_eval, modular_eval, "eval"), parsed.forward_atol, "eval"
    )

    oracle.train()
    modular.train()
    set_epoch_state(oracle, checkpoint_args, parsed.epoch_index)
    oracle_train = forward_m4_batch(oracle, batch)
    pos_weight = torch.tensor(pos_weight_value, device=device, dtype=oracle_train["logits"].dtype)
    oracle_losses = oracle_objective(oracle_train, batch, checkpoint_args, pos_weight)
    oracle_losses["loss"].backward()

    set_epoch_state(modular, checkpoint_args, parsed.epoch_index)
    modular_train = forward_m4_batch(modular, batch)
    modular_objective = M4Objective(
        M4LossConfig.from_mapping(checkpoint_args), pos_weight_value
    ).to(device)
    modular_losses = modular_objective(modular_train, batch)
    modular_losses["loss"].backward()

    train_check = assert_rows_close(
        tensor_differences(oracle_train, modular_train, "train"), parsed.forward_atol, "train"
    )
    loss_check = assert_rows_close(
        tensor_differences(oracle_losses, modular_losses, "loss"), parsed.forward_atol, "loss"
    )
    gradient_check = assert_rows_close(
        compare_parameter_tensors(oracle, modular, "grad"), parsed.gradient_atol, "gradient"
    )

    learning_rate = float(checkpoint_args.get("learning_rate", 5e-5))
    weight_decay = float(checkpoint_args.get("weight_decay", 1e-4))
    oracle_optimizer = torch.optim.AdamW(oracle.parameters(), lr=learning_rate, weight_decay=weight_decay)
    modular_optimizer = torch.optim.AdamW(modular.parameters(), lr=learning_rate, weight_decay=weight_decay)
    oracle_optimizer.step()
    modular_optimizer.step()
    optimizer_check = assert_rows_close(
        compare_parameter_tensors(oracle, modular, "data"), parsed.optimizer_atol, "optimizer_step"
    )

    report = {
        "pass": True,
        "checkpoint": str(Path(parsed.checkpoint).resolve()),
        "manifest": str(Path(parsed.manifest).resolve()),
        "device": str(device),
        "case_ids": [row["apex_case_id"] for row in rows],
        "state_tensors": len(modular.state_dict()),
        "parameters": sum(parameter.numel() for parameter in modular.parameters()),
        "pos_weight": pos_weight_value,
        "label_stats": label_stats,
        "eval": eval_check,
        "train": train_check,
        "loss": loss_check,
        "gradient": gradient_check,
        "optimizer_step": optimizer_check,
    }
    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
