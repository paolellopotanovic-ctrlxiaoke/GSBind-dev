"""Residue-level and case-level metrics used throughout M4."""

import csv
import math
from pathlib import Path

import torch


def binary_auc(scores, labels):
    if scores.numel() == 0:
        return float("nan")
    labels = labels.to(dtype=torch.long)
    positive_count = int(labels.sum().item())
    count = int(labels.numel())
    negative_count = count - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, count + 1, device=scores.device, dtype=torch.float)
    positive_ranks = ranks[labels > 0].sum()
    auc = (positive_ranks - positive_count * (positive_count + 1) / 2.0) / max(
        positive_count * negative_count, 1
    )
    return float(auc.cpu())


def binary_aupr(scores, labels):
    if scores.numel() == 0:
        return float("nan")
    labels = labels.to(dtype=torch.float)
    total_positive = labels.sum().clamp_min(1.0)
    if total_positive.item() <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    ordered_labels = labels[order]
    true_positive = torch.cumsum(ordered_labels, dim=0)
    false_positive = torch.cumsum(1.0 - ordered_labels, dim=0)
    recall = true_positive / total_positive
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = torch.cat([recall.new_zeros(1), recall])
    precision = torch.cat([precision.new_ones(1), precision])
    return float(torch.sum((recall[1:] - recall[:-1]) * precision[1:]).cpu())


def metric_scalar(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
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
    count = int(labels.numel())
    selected_count = min(int(k), count)
    if selected_count <= 0:
        return float("nan")
    top = torch.topk(scores, k=selected_count, largest=True).indices
    return float(labels[top].mean())


def mean_or_nan(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def case_metrics(scores, labels, case_id):
    count = int(labels.numel())
    positive = int(labels.sum().item()) if count else 0
    return {
        "apex_case_id": case_id,
        "valid_residues": count,
        "positive_residues": positive,
        "positive_rate": positive / max(count, 1),
        "aupr": binary_aupr(scores, labels),
        "auroc": binary_auc(scores, labels),
        "p_at_10": precision_at_k(scores, labels, 10),
        "p_at_20": precision_at_k(scores, labels, 20),
        "p_at_50": precision_at_k(scores, labels, 50),
    }


def aggregate_metrics(scores, labels, losses, case_rows, prefix):
    positive = int(labels.sum().item()) if labels.numel() else 0
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
        "positive_residues": positive,
        "positive_rate": positive / max(int(labels.numel()), 1),
        "num_cases": len(case_rows),
    }
