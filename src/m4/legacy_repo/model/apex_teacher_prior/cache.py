import hashlib
from pathlib import Path

import torch as pt

from apex_dataset import cache_key_for_row


def teacher_cache_key_for_row(row, teacher_name="pesto_v4_1", version="teacher_prior_v1"):
    raw = "|".join([cache_key_for_row(row), str(teacher_name), str(version)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def teacher_cache_path_for_row(cache_dir, row, teacher_name="pesto_v4_1", version="teacher_prior_v1"):
    return Path(cache_dir) / teacher_cache_key_for_row(row, teacher_name=teacher_name, version=version)


def save_teacher_cache(path, logits, probabilities=None, hidden=None, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "teacher_logits": logits.detach().float().cpu(),
        "teacher_probabilities": probabilities.detach().float().cpu() if isinstance(probabilities, pt.Tensor) else None,
        "teacher_hidden": hidden.detach().float().cpu() if isinstance(hidden, pt.Tensor) else None,
        "metadata": metadata or {},
    }
    pt.save(payload, path)

