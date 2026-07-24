from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from m4.data import M4DataConfig
from m4.lightning import M4LightningModule, M4OptimizationConfig, build_m4_scheduler
from m4.lightning.cli import apply_overrides
from m4.losses import M4LossConfig, M4Objective
from m4.models import build_m4_model, forward_m4_batch, model_kwargs_from_checkpoint_args


ROOT = Path(__file__).resolve().parents[1]


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, *args, **kwargs):
        self.received = (args, kwargs)
        return {"logits": args[2].sum(dim=-1)}


def minimal_batch() -> dict:
    return {
        "X": torch.randn(3, 3),
        "q": torch.randn(3, 2),
        "atom_to_residue_index": torch.tensor([0, 0, 1]),
        "atom_counts": torch.tensor([3]),
        "residue_counts": torch.tensor([2]),
        "residue_graph_caches": [{"graph": "cache"}],
        "s1_surface_caches": [{"surface": "cache"}],
        "s1_static_geometry_caches": [{"static": "cache"}],
    }


def test_forward_contract_uses_dense_assignment_only_during_training() -> None:
    model = CaptureModel()
    batch = minimal_batch()
    model.train()
    forward_m4_batch(model, batch)
    args, kwargs = model.received
    assert args[3].shape == (3, 2)
    assert kwargs["atom_to_residue_index"] is None
    assert kwargs["s1_static_geometry_caches"] is None

    model.eval()
    forward_m4_batch(model, batch)
    args, kwargs = model.received
    assert args[3] is None
    assert torch.equal(kwargs["atom_to_residue_index"], batch["atom_to_residue_index"])
    assert kwargs["s1_static_geometry_caches"] == batch["s1_static_geometry_caches"]


def test_holo20k_config_maps_to_modular_data_and_loss_contracts() -> None:
    payload = yaml.safe_load((ROOT / "configs" / "m4_holo20k.yaml").read_text(encoding="utf-8"))
    values = payload["args"]
    data = M4DataConfig.from_mapping(values)
    loss = M4LossConfig.from_mapping(values)
    assert data.batch_size == 2
    assert data.states == "holo"
    assert data.residue_graph_cache_ks == (16, 24, 32)
    assert loss.base_aux_loss_weight == 0.2
    assert loss.surface_soft_loss_weight == 0.05
    assert loss.semantic_patch_set_loss_weight == 0.03


def test_objective_matches_explicit_active_terms_without_optional_heads() -> None:
    config = M4LossConfig(base_aux_loss_weight=0.2)
    objective = M4Objective(config, pos_weight=2.0)
    output = {
        "logits": torch.tensor([0.2, -0.4], requires_grad=True),
        "base_logits": torch.tensor([0.1, -0.3], requires_grad=True),
        "prior_logits": torch.tensor([0.0, -0.2], requires_grad=True),
    }
    batch = {
        "y": torch.tensor([1.0, 0.0]),
        "mask": torch.ones(2),
        "residue_counts": torch.tensor([2]),
    }
    losses = objective(output, batch)
    expected_final = torch.nn.functional.binary_cross_entropy_with_logits(
        output["logits"], batch["y"], pos_weight=torch.tensor(2.0)
    )
    expected_base = torch.nn.functional.binary_cross_entropy_with_logits(
        output["base_logits"], batch["y"], pos_weight=torch.tensor(2.0)
    )
    assert torch.allclose(losses["loss"], expected_final + 0.2 * expected_base)
    losses["loss"].backward()
    assert output["logits"].grad is not None


def test_model_parameter_contract_is_deterministic() -> None:
    values = yaml.safe_load((ROOT / "configs" / "m4_holo20k.yaml").read_text(encoding="utf-8"))["args"]
    torch.manual_seed(20260724)
    reference = build_m4_model(values)
    torch.manual_seed(20260724)
    candidate = build_m4_model(values)

    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    assert type(candidate).__module__ == "m4.models.components.predictor"
    assert list(candidate_state) == list(reference_state)
    assert len(candidate_state) == 791
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 4_360_516
    for key in reference_state:
        assert candidate_state[key].shape == reference_state[key].shape
        assert torch.equal(candidate_state[key], reference_state[key]), key


def test_forward_runtime_is_split_by_scientific_responsibility() -> None:
    from m4.models.components.execution import M4ExecutionMixin
    from m4.models.components.runtime import (
        GaussianFlowMixin,
        StochasticRuntimeMixin,
        SurfaceFlowMixin,
    )

    assert M4ExecutionMixin.__bases__ == (
        StochasticRuntimeMixin,
        SurfaceFlowMixin,
        GaussianFlowMixin,
    )
    assert {name for name in M4ExecutionMixin.__dict__ if name.startswith("_")} == {
        "__module__",
        "__doc__",
    }
    assert M4ExecutionMixin.forward.__module__ == "m4.models.components.execution"
    assert StochasticRuntimeMixin.__module__.endswith("runtime.stochastic")
    assert SurfaceFlowMixin.__module__.endswith("runtime.surface_flow")
    assert GaussianFlowMixin.__module__.endswith("runtime.gaussian_flow")


def test_lightning_epoch_state_matches_semantic_boundary_schedule() -> None:
    class SemanticHead:
        def __init__(self):
            self.scale = None

        def set_boundary_refinement_scale(self, scale: float) -> None:
            self.scale = scale

    model = CaptureModel()
    model.semantic_patch_head = SemanticHead()
    module = M4LightningModule(
        model=model,
        loss_config=M4LossConfig(),
        optimization_config=M4OptimizationConfig(),
        pos_weight=1.0,
        run_config={"seed": 42, "semantic_patch_boundary_start_epoch": 3},
    )
    module.apply_epoch_state(0)
    assert model.semantic_patch_head.scale == 0.0
    first = torch.rand(3)
    module.apply_epoch_state(0)
    assert torch.equal(first, torch.rand(3))
    module.apply_epoch_state(2)
    assert model.semantic_patch_head.scale == 1.0


def test_lightning_cli_overrides_preserve_yaml_types() -> None:
    result = apply_overrides(
        {"batch_size": 2, "amp": True},
        ["batch_size=1", "amp=false", "residue_graph_cache_ks=[16, 24, 32]"],
    )
    assert result == {
        "batch_size": 1,
        "amp": False,
        "residue_graph_cache_ks": [16, 24, 32],
    }


def test_concise_holo20k_config_matches_frozen_full_snapshot() -> None:
    concise = yaml.safe_load((ROOT / "configs" / "m4_holo20k.yaml").read_text(encoding="utf-8"))["args"]
    exact = yaml.safe_load((ROOT / "configs" / "m4_holo20k_exact.yaml").read_text(encoding="utf-8"))["args"]
    assert model_kwargs_from_checkpoint_args(concise) == model_kwargs_from_checkpoint_args(exact)
    assert asdict(M4LossConfig.from_mapping(concise)) == asdict(M4LossConfig.from_mapping(exact))
    assert asdict(M4OptimizationConfig.from_mapping(concise)) == asdict(
        M4OptimizationConfig.from_mapping(exact)
    )
    concise_data = asdict(M4DataConfig.from_mapping(concise))
    exact_data = asdict(M4DataConfig.from_mapping(exact))
    path_fields = {
        "manifest",
        "fixed_train_manifest",
        "fixed_val_manifest",
        "tensor_cache_dir",
        "residue_graph_cache_dir",
        "s1_surface_cache_dir",
        "s1_static_geometry_cache_dir",
    }
    assert {key: value for key, value in concise_data.items() if key not in path_fields} == {
        key: value for key, value in exact_data.items() if key not in path_fields
    }


def test_lightning_scheduler_matches_published_recipe() -> None:
    config = M4OptimizationConfig(
        learning_rate=5e-5,
        scheduler="cosine",
        scheduler_total_epochs=20,
        warmup_ratio=0.05,
        min_learning_rate=1e-6,
    )
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler, info = build_m4_scheduler(config, optimizer, 2500, 20)
    assert info == {
        "scheduler": "cosine",
        "total_steps": 50000,
        "warmup_steps": 2500,
        "min_learning_rate": 1e-6,
    }
    assert scheduler is not None
    observed = []
    for step in range(50001):
        if step in {0, 2499, 2500, 26250, 50000}:
            observed.append((step, scheduler.get_last_lr()[0]))
        optimizer.step()
        scheduler.step()
    assert observed[0][1] == config.learning_rate / 2500
    assert observed[1][1] == config.learning_rate
    assert observed[2][1] == config.learning_rate
    assert abs(observed[3][1] - 2.55e-5) < 1e-12
    assert abs(observed[4][1] - config.min_learning_rate) < 1e-12
