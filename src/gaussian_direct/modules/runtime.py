from __future__ import annotations

import torch
import torch.nn.functional as functional


def replay_dropout_rng(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    *,
    probability: float,
    training: bool,
) -> None:
    probability = float(probability)
    if not training or probability <= 0.0:
        return
    placeholder = reference.new_zeros(shape)
    functional.dropout(placeholder, p=probability, training=True, inplace=False)
