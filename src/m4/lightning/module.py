"""Lightning training loop for the exact M4 model and objective."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import lightning as L
import torch

from ..losses import M4LossConfig, M4Objective
from ..evaluation.metrics import binary_auc, binary_aupr
from ..models import M4Model, forward_m4_batch


@dataclass(frozen=True)
class M4OptimizationConfig:
    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine"
    scheduler_total_epochs: int = 20
    warmup_steps: int = 0
    warmup_ratio: float = 0.05
    min_learning_rate: float = 1.0e-6

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "M4OptimizationConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


def build_m4_scheduler(
    config: M4OptimizationConfig,
    optimizer: torch.optim.Optimizer,
    optimizer_steps_per_epoch: int,
    trainer_epochs: int,
    max_steps: int = 0,
):
    """Build the validated per-step warmup/cosine or linear schedule."""

    schedule_epochs = (
        config.scheduler_total_epochs
        if config.scheduler_total_epochs > 0
        else max(int(trainer_epochs), 1)
    )
    total_steps = max(int(optimizer_steps_per_epoch) * schedule_epochs, 1)
    if int(max_steps) > 0:
        total_steps = min(total_steps, int(max_steps))
    warmup_steps = int(config.warmup_steps)
    if warmup_steps <= 0:
        warmup_steps = int(round(total_steps * config.warmup_ratio))
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    if config.scheduler == "none":
        return None, {"scheduler": "none", "total_steps": total_steps, "warmup_steps": warmup_steps}
    if config.scheduler not in {"cosine", "linear"}:
        raise ValueError(f"Unsupported M4 scheduler: {config.scheduler}")
    min_ratio = config.min_learning_rate / max(config.learning_rate, 1e-12)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = min(
            max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
            1.0,
        )
        value = (
            1.0 - progress
            if config.scheduler == "linear"
            else 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        return min_ratio + (1.0 - min_ratio) * value

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {
        "scheduler": config.scheduler,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "min_learning_rate": config.min_learning_rate,
    }


class M4LightningModule(L.LightningModule):
    """Separate M4 science (model/loss) from training-engine mechanics."""

    def __init__(
        self,
        model: M4Model,
        loss_config: M4LossConfig,
        optimization_config: M4OptimizationConfig,
        pos_weight: float,
        run_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.model = model
        self.objective = M4Objective(loss_config, pos_weight)
        self.optimization_config = optimization_config
        self.run_config = dict(run_config or {})
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.save_hyperparameters(
            {
                "loss": asdict(loss_config),
                "optimization": asdict(optimization_config),
                "pos_weight": float(pos_weight),
                "run_config": self.run_config,
            }
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return forward_m4_batch(self.model, batch)

    def on_train_epoch_start(self) -> None:
        self.apply_epoch_state(self.current_epoch)

    def apply_epoch_state(self, epoch_index: int) -> None:
        """Match the validated trainer's epoch seed and semantic-stage switch."""

        epoch_seed = int(self.run_config.get("seed", 42)) + int(epoch_index)
        random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        semantic_head = getattr(self.model, "semantic_patch_head", None)
        if semantic_head is not None and hasattr(semantic_head, "set_boundary_refinement_scale"):
            start_epoch = int(self.run_config.get("semantic_patch_boundary_start_epoch", 1))
            boundary_active = (int(epoch_index) + 1) >= start_epoch
            semantic_head.set_boundary_refinement_scale(1.0 if boundary_active else 0.0)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int):
        if batch.get("skip_batch"):
            return None
        output = self(batch)
        losses = self.objective(output, batch)
        self._log_losses(losses, "train", on_step=True)
        return losses["loss"]

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int):
        if batch.get("skip_batch"):
            return None
        output = self(batch)
        losses = self.objective(output, batch)
        self._log_losses(losses, "val", on_step=False)
        valid = batch["mask"] > 0.5
        self.validation_outputs.append(
            {
                "scores": torch.sigmoid(output["logits"][valid]).detach().cpu(),
                "labels": batch["y"][valid].detach().float().cpu(),
            }
        )
        return losses["loss"]

    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return
        scores = torch.cat([row["scores"] for row in self.validation_outputs])
        labels = torch.cat([row["labels"] for row in self.validation_outputs])
        self.log("val/aupr", binary_aupr(scores, labels), prog_bar=True, sync_dist=True)
        self.log("val/auroc", binary_auc(scores, labels), prog_bar=False, sync_dist=True)
        self.validation_outputs.clear()

    def configure_optimizers(self):
        cfg = self.optimization_config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        trainer_epochs = max(int(getattr(self.trainer, "max_epochs", 1)), 1)
        estimated_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        steps_per_epoch = max(int(math.ceil(estimated_steps / trainer_epochs)), 1)
        trainer_max_steps = int(getattr(self.trainer, "max_steps", -1))
        scheduler, _ = build_m4_scheduler(
            cfg,
            optimizer,
            steps_per_epoch,
            trainer_epochs,
            max_steps=trainer_max_steps if trainer_max_steps > 0 else 0,
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["m4_release_metadata"] = {
            "format": "m4_lightning_v1",
            "validated_core": "S1SurfaceFieldPredictor",
            "compatibility_export_key": "model_state_dict",
        }
        checkpoint["model_state_dict"] = {
            key: value.detach().cpu() for key, value in self.model.state_dict().items()
        }
        checkpoint["args"] = dict(self.run_config)

    def _log_losses(self, losses: Mapping[str, torch.Tensor], prefix: str, *, on_step: bool) -> None:
        batch_size = 1
        for name, value in losses.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch_size,
                sync_dist=True,
            )
