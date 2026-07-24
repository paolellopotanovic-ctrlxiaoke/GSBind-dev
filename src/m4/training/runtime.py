"""Small runtime helpers shared by training, evaluation, and inference."""

import torch


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            moved[key] = [
                {
                    cache_key: cache_value.to(device, non_blocking=True)
                    if isinstance(cache_value, torch.Tensor)
                    else cache_value
                    for cache_key, cache_value in item.items()
                }
                for item in value
            ]
        else:
            moved[key] = value
    return moved


def masked_bce(logits, labels, mask, pos_weight=None):
    valid = mask > 0.5
    if not torch.any(valid):
        return logits.sum() * 0.0
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight.to(device=logits.device, dtype=logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], labels[valid], **kwargs)
