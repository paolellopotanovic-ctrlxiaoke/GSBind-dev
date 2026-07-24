"""Train exact M4 through PyTorch Lightning using the paper YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from ..data import M4DataConfig, M4DataModule
from ..losses import M4LossConfig
from ..models import build_m4_model
from .module import M4LightningModule, M4OptimizationConfig


def load_run_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("M4 config root must be a mapping")
    command = payload.get("command")
    if command not in (None, "train", "train-lightning"):
        raise ValueError(f"Expected a training config, found command={command!r}")
    values = payload.get("args", payload)
    if not isinstance(values, dict):
        raise ValueError("M4 config args must be a mapping")
    return dict(values)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply typed KEY=VALUE command-line overrides to a copied config."""

    result = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, found {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override key cannot be empty")
        result[key] = yaml.safe_load(raw_value)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train exact M4 with PyTorch Lightning")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one YAML value; repeat for multiple values.",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    parsed = parser.parse_args(argv)
    cfg = apply_overrides(load_run_config(parsed.config), parsed.overrides)

    L.seed_everything(int(cfg.get("seed", 42)), workers=True)
    data = M4DataModule(M4DataConfig.from_mapping(cfg))
    data.prepare_data()
    data.setup("fit")
    if data.pos_weight is None:
        raise RuntimeError("M4DataModule did not estimate pos_weight")

    module = M4LightningModule(
        model=build_m4_model(cfg),
        loss_config=M4LossConfig.from_mapping(cfg),
        optimization_config=M4OptimizationConfig.from_mapping(cfg),
        pos_weight=data.pos_weight,
        run_config=cfg,
    )
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="model_epoch_{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
        save_weights_only=False,
    )
    accelerator = "gpu" if str(cfg.get("device", "cuda")).startswith("cuda") else "cpu"
    precision = "16-mixed" if bool(cfg.get("amp", False)) and accelerator == "gpu" else "32-true"
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(cfg.get("epochs", 20)),
        max_steps=int(cfg.get("max_steps", 0)) if int(cfg.get("max_steps", 0)) > 0 else -1,
        accumulate_grad_batches=int(cfg.get("grad_accum_steps", 1)),
        gradient_clip_val=float(cfg.get("grad_clip_norm", 0.0)),
        precision=precision,
        callbacks=[checkpoint],
        default_root_dir=output_dir,
        log_every_n_steps=int(cfg.get("log_every", 1)),
        check_val_every_n_epoch=1 if data.val_rows else None,
        num_sanity_val_steps=2 if data.val_rows else 0,
        deterministic=True,
        fast_dev_run=parsed.fast_dev_run,
    )
    if cfg.get("matmul_precision"):
        torch.set_float32_matmul_precision(str(cfg["matmul_precision"]))
    trainer.fit(module, datamodule=data)


if __name__ == "__main__":
    main()
