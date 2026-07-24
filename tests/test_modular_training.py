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


def test_modular_model_is_parameter_identical_to_frozen_oracle() -> None:
    from apex_s1_surface_field.eval_s1_surface_field import build_model_from_ckpt_args

    values = yaml.safe_load((ROOT / "configs" / "m4_holo20k.yaml").read_text(encoding="utf-8"))["args"]
    torch.manual_seed(20260724)
    oracle = build_model_from_ckpt_args(values)
    torch.manual_seed(20260724)
    modular = build_m4_model(values)

    oracle_state = oracle.state_dict()
    modular_state = modular.state_dict()
    assert type(modular).__module__ == "m4.models.components.predictor"
    assert list(modular_state) == list(oracle_state)
    assert len(modular_state) == 791
    for key in oracle_state:
        assert modular_state[key].shape == oracle_state[key].shape
        assert torch.equal(modular_state[key], oracle_state[key]), key

    modular.load_state_dict(oracle_state, strict=True)
    oracle.load_state_dict(modular_state, strict=True)


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
    module.apply_legacy_epoch_state(0)
    assert model.semantic_patch_head.scale == 0.0
    first = torch.rand(3)
    module.apply_legacy_epoch_state(0)
    assert torch.equal(first, torch.rand(3))
    module.apply_legacy_epoch_state(2)
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


def test_lightning_scheduler_matches_frozen_trainer() -> None:
    from types import SimpleNamespace

    from apex_s1_surface_field.train_s1_surface_field import build_scheduler

    config = M4OptimizationConfig(
        learning_rate=5e-5,
        scheduler="cosine",
        scheduler_total_epochs=20,
        warmup_ratio=0.05,
        min_learning_rate=1e-6,
    )
    legacy_parameter = torch.nn.Parameter(torch.zeros(()))
    modular_parameter = torch.nn.Parameter(torch.zeros(()))
    legacy_optimizer = torch.optim.AdamW([legacy_parameter], lr=config.learning_rate)
    modular_optimizer = torch.optim.AdamW([modular_parameter], lr=config.learning_rate)
    legacy_scheduler, legacy_info = build_scheduler(
        SimpleNamespace(
            scheduler=config.scheduler,
            scheduler_total_epochs=config.scheduler_total_epochs,
            epochs=20,
            max_steps=0,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,
            min_learning_rate=config.min_learning_rate,
            learning_rate=config.learning_rate,
        ),
        legacy_optimizer,
        2500,
    )
    modular_scheduler, modular_info = build_m4_scheduler(config, modular_optimizer, 2500, 20)
    assert legacy_info == modular_info
    assert modular_scheduler is not None
    for _ in range(50001):
        assert legacy_scheduler.get_last_lr() == modular_scheduler.get_last_lr()
        legacy_optimizer.step()
        modular_optimizer.step()
        legacy_scheduler.step()
        modular_scheduler.step()
