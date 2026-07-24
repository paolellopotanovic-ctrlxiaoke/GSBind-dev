import csv
import math
from pathlib import Path

import torch as pt

from train_apex import binary_auc, binary_aupr


def metric_scalar(value):
    if value is None:
        return ""
    if isinstance(value, pt.Tensor):
        value = float(value.detach().float().cpu())
    if isinstance(value, float):
        return f"{value:.8f}" if math.isfinite(value) else str(value)
    return value


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def precision_at_k(scores, labels, k):
    n = int(labels.numel())
    kk = min(int(k), n)
    if kk <= 0:
        return float("nan")
    top = pt.topk(scores, k=kk, largest=True).indices
    return float(labels[top].mean())


def mean_or_nan(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(values) / len(values) if values else float("nan")


def case_metrics(scores, labels, case_id):
    n = int(labels.numel())
    pos = int(labels.sum().item()) if n else 0
    return {
        "apex_case_id": case_id,
        "valid_residues": n,
        "positive_residues": pos,
        "positive_rate": pos / max(n, 1),
        "aupr": binary_aupr(scores, labels),
        "auroc": binary_auc(scores, labels),
        "p_at_10": precision_at_k(scores, labels, 10),
        "p_at_20": precision_at_k(scores, labels, 20),
        "p_at_50": precision_at_k(scores, labels, 50),
    }


def aggregate_metrics(scores, labels, losses, case_rows, prefix):
    pos = int(labels.sum().item()) if labels.numel() else 0
    return {
        "prefix": prefix,
        "loss": sum(losses) / max(len(losses), 1),
        "aupr": binary_aupr(scores, labels),
        "auroc": binary_auc(scores, labels),
        "p_at_10": precision_at_k(scores, labels, 10),
        "p_at_20": precision_at_k(scores, labels, 20),
        "p_at_50": precision_at_k(scores, labels, 50),
        "case_wise_aupr": mean_or_nan([row["aupr"] for row in case_rows]),
        "case_wise_auroc": mean_or_nan([row["auroc"] for row in case_rows]),
        "case_wise_p_at_10": mean_or_nan([row["p_at_10"] for row in case_rows]),
        "case_wise_p_at_20": mean_or_nan([row["p_at_20"] for row in case_rows]),
        "case_wise_p_at_50": mean_or_nan([row["p_at_50"] for row in case_rows]),
        "valid_residues": int(labels.numel()),
        "positive_residues": pos,
        "positive_rate": pos / max(int(labels.numel()), 1),
        "num_cases": len(case_rows),
    }

