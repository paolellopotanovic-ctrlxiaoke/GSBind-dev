"""Post-training GPSFun AF2 full375 checkpoint selection."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import CheckpointSelectionConfig, ExperimentConfig
from .data import (
    CompactGaussianDirectDataset,
    collate_single,
    load_checkpoint_model,
    model_arguments_from_batch,
    split_field,
)
from .utils import atomic_write_json, read_csv_rows, resolve_device, sha256_file


POOLED_METRICS = (
    "pooled_aupr",
    "pooled_auroc",
    "pooled_p_at_10",
    "pooled_p_at_20",
    "pooled_p_at_50",
)
CASE_MEAN_METRICS = (
    "case_mean_aupr",
    "case_mean_auroc",
    "case_mean_p_at_10",
    "case_mean_p_at_20",
    "case_mean_p_at_50",
)
CASE_MEDIAN_METRICS = (
    "case_median_aupr",
    "case_median_auroc",
    "case_median_p_at_10",
    "case_median_p_at_20",
    "case_median_p_at_50",
)
PERFORMANCE_METRICS = POOLED_METRICS + CASE_MEAN_METRICS + CASE_MEDIAN_METRICS + ("macro_recall",)
METRIC_FAMILIES = {
    "pooled": POOLED_METRICS,
    "case_mean": CASE_MEAN_METRICS,
    "case_median": CASE_MEDIAN_METRICS,
    "macro_recall": ("macro_recall",),
}


def validate_manifest_contract(
    rows: Sequence[Mapping[str, str]], settings: CheckpointSelectionConfig
) -> tuple[int, int]:
    if len(rows) != settings.expected_cases:
        raise ValueError(
            f"GPSFun validation manifest has {len(rows)} cases; "
            f"expected {settings.expected_cases}"
        )
    case_ids = [str(row.get("apex_case_id", "")) for row in rows]
    if len(set(case_ids)) != len(case_ids) or any(not case_id for case_id in case_ids):
        raise ValueError("GPSFun validation manifest case IDs must be non-empty and unique")
    valid_residues, positive_residues = protein_class_counts(rows)
    if valid_residues != settings.expected_valid_residues:
        raise ValueError(
            f"GPSFun validation manifest has {valid_residues} valid residues; "
            f"expected {settings.expected_valid_residues}"
        )
    if positive_residues != settings.expected_positive_residues:
        raise ValueError(
            f"GPSFun validation manifest has {positive_residues} positive residues; "
            f"expected {settings.expected_positive_residues}"
        )
    return valid_residues, positive_residues


def protein_class_counts(rows: Sequence[Mapping[str, str]]) -> tuple[int, int]:
    valid_residues = 0
    positive_residues = 0
    for row in rows:
        labels = np.asarray(split_field(row["labels"], int), dtype=np.int64)
        mask = np.asarray(split_field(row["label_mask"], int), dtype=np.int64) > 0
        if labels.shape != mask.shape:
            raise ValueError(f"invalid labels for {row.get('apex_case_id', '')}")
        valid_residues += int(mask.sum())
        positive_residues += int(labels[mask].sum())
    return valid_residues, positive_residues


def composite_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    epochs = [int(row["epoch"]) for row in rows]
    matrix = np.asarray(
        [[float(row[metric]) for metric in PERFORMANCE_METRICS] for row in rows],
        dtype=float,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("checkpoint selection requires finite values for all 16 performance metrics")
    standard_scores = zscore(matrix)
    ranks = competition_ranks(matrix)
    family_raw = np.asarray(
        [
            [
                matrix[index, [PERFORMANCE_METRICS.index(metric) for metric in family]].mean()
                for family in METRIC_FAMILIES.values()
            ]
            for index in range(len(rows))
        ],
        dtype=float,
    )
    family_scores = zscore(family_raw)
    family_ranks = competition_ranks(family_raw)
    output: list[dict[str, Any]] = []
    for index, epoch in enumerate(epochs):
        result: dict[str, Any] = {"epoch": epoch}
        result.update(
            {metric: float(matrix[index, column]) for column, metric in enumerate(PERFORMANCE_METRICS)}
        )
        result.update(
            {f"{metric}_z": float(standard_scores[index, column]) for column, metric in enumerate(PERFORMANCE_METRICS)}
        )
        result.update(
            {f"{metric}_rank": float(ranks[index, column]) for column, metric in enumerate(PERFORMANCE_METRICS)}
        )
        result["all16_equal_z"] = float(standard_scores[index].mean())
        result["all16_mean_rank"] = float(ranks[index].mean())
        for family_index, family in enumerate(METRIC_FAMILIES):
            result[f"{family}_family_score"] = float(family_scores[index, family_index])
            result[f"{family}_family_rank"] = float(family_ranks[index, family_index])
        result["family_balanced_z"] = float(family_scores[index].mean())
        result["family_balanced_rank"] = float(family_ranks[index].mean())
        output.append(result)
    return output


def zscore(values: np.ndarray) -> np.ndarray:
    deviation = values.std(axis=0, ddof=0)
    return (values - values.mean(axis=0)) / np.where(deviation == 0.0, 1.0, deviation)


def competition_ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-values, axis=0), axis=0).astype(float) + 1


def select_checkpoints(config: ExperimentConfig) -> dict[str, Any]:
    settings = config.training.checkpoint_selection
    if not settings.enabled:
        raise ValueError("training.checkpoint_selection.enabled must be true")
    optimizer = config.training.optimizer
    if (
        settings.epoch_min != 0
        or settings.epoch_max != optimizer.epochs - 1
    ):
        raise ValueError(
            "the historical protocol requires one checkpoint for every epoch: "
            f"0 through {optimizer.epochs - 1}"
        )
    validate_paths(settings)
    manifest_rows = read_csv_rows(settings.manifest)
    valid_residues, positive_residues = validate_manifest_contract(manifest_rows, settings)
    checkpoint_by_epoch = discover_checkpoints(config.training.output_dir, settings)
    output_root = settings.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    epoch_metrics: list[dict[str, Any]] = []
    for epoch in range(settings.epoch_min, settings.epoch_max + 1):
        print(f"Evaluating epoch {epoch} on GPSFun AF2 full375", flush=True)
        metrics = evaluate_checkpoint(
            checkpoint_by_epoch[epoch],
            manifest_rows,
            settings,
            output_root / f"epoch_{epoch:02d}",
        )
        epoch_metrics.append(metrics)
    composite_rows = composite_scores(epoch_metrics)
    atomic_write_csv(output_root / "composite_summary.csv", composite_rows)
    selected = max(composite_rows, key=lambda row: row[settings.selection_metric])
    selected_epoch = int(selected["epoch"])
    selected_path = checkpoint_by_epoch[selected_epoch]
    selected_report = {
        "schema_version": "gpsfun_af2_full375_checkpoint_selection_v1",
        "strategy": settings.strategy,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selection_metric": settings.selection_metric,
        "selection_score": selected[settings.selection_metric],
        "validation_cases": len(manifest_rows),
        "validation_valid_residues": valid_residues,
        "validation_positive_residues": positive_residues,
        "performance_metrics": list(PERFORMANCE_METRICS),
        "loss_metrics_included": False,
    }
    atomic_write_json(output_root / "selected_checkpoint.json", selected_report)
    atomic_write_json(
        output_root / "selection_contract.json",
        {
            "schema_version": "gpsfun_af2_full375_checkpoint_selection_contract_v1",
            "selection_mode": "post_training_full_epoch_sweep",
            "epochs": sorted(checkpoint_by_epoch),
            "validation_manifest": str(settings.manifest),
            "validation_manifest_sha256": sha256_file(settings.manifest),
            "validation_compact_dir": str(settings.compact_dir),
            "expected_cases": settings.expected_cases,
            "expected_valid_residues": settings.expected_valid_residues,
            "expected_positive_residues": settings.expected_positive_residues,
            "metric_families": {name: list(metrics) for name, metrics in METRIC_FAMILIES.items()},
            "formula": "mean(z(mean(pooled)), z(mean(case_mean)), z(mean(case_median)), z(macro_recall))",
            "zscore_ddof": 0,
            "primary_score": "family_balanced_z",
            "loss_or_brier_in_primary_score": False,
        },
    )
    marker = output_root / "VALIDATION_SELECTION_COMPLETE"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text("complete\n", encoding="utf-8")
    os.replace(temporary, marker)
    return selected_report


def validate_paths(settings: CheckpointSelectionConfig) -> None:
    if not settings.manifest.is_file():
        raise FileNotFoundError(f"GPSFun validation manifest does not exist: {settings.manifest}")
    if not settings.compact_dir.is_dir():
        raise FileNotFoundError(
            f"GPSFun validation compact directory does not exist: {settings.compact_dir}"
        )
    actual_hash = sha256_file(settings.manifest)
    if actual_hash != settings.validation_manifest_sha256:
        raise ValueError(
            f"GPSFun validation manifest SHA-256 mismatch: expected {settings.validation_manifest_sha256}, got {actual_hash}"
        )


def discover_checkpoints(
    output_dir: Path, settings: CheckpointSelectionConfig
) -> dict[int, Path]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"training output directory does not exist: {output_dir}")
    checkpoint_by_epoch: dict[int, Path] = {}
    pattern = re.compile(r"^model_epoch_(\d+)\.pt$")
    for path in output_dir.glob("model_epoch_*.pt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch in checkpoint_by_epoch:
            raise ValueError(f"duplicate checkpoint epoch {epoch} in {output_dir}")
        checkpoint_by_epoch[epoch] = path
    expected_epochs = set(range(settings.epoch_min, settings.epoch_max + 1))
    missing_epochs = sorted(expected_epochs - checkpoint_by_epoch.keys())
    extra_epochs = sorted(checkpoint_by_epoch.keys() - expected_epochs)
    if missing_epochs or extra_epochs:
        raise ValueError(
            f"checkpoint epoch contract mismatch; missing={missing_epochs}, extra={extra_epochs}"
        )
    return checkpoint_by_epoch


def evaluate_checkpoint(
    checkpoint: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    settings: CheckpointSelectionConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CompactGaussianDirectDataset(settings.manifest, settings.compact_dir)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        collate_fn=collate_single,
        pin_memory=False,
    )
    device = resolve_device(settings.device)
    model, payload = load_checkpoint_model(checkpoint, device)
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    case_metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(**model_arguments_from_batch(batch, device)).float().cpu()
            scores = torch.sigmoid(logits[:, 0]).numpy()
            labels = batch["y"].numpy()
            masks = batch["mask"].numpy() > 0.5
            all_labels.append(labels[masks])
            all_scores.append(scores[masks])
            offset = 0
            for row, count in zip(batch["row"], batch["residue_counts"]):
                local = slice(offset, offset + int(count))
                local_mask = masks[local]
                local_labels = labels[local][local_mask]
                local_scores = scores[local][local_mask]
                case_id = str(row.get("apex_case_id", ""))
                predicted_labels = local_scores >= settings.threshold
                recalls = predicted_labels[local_labels == 1]
                case_metrics.append(
                    {
                        "apex_case_id": case_id,
                        "valid_residues": int(local_labels.size),
                        "positive_residues": int(local_labels.sum()),
                        "recall": float(recalls.mean()) if recalls.size else None,
                        "aupr": safe_aupr(local_labels, local_scores),
                        "auroc": safe_auroc(local_labels, local_scores),
                        "p_at_10": precision_at_k(local_labels, local_scores, 10),
                        "p_at_20": precision_at_k(local_labels, local_scores, 20),
                        "p_at_50": precision_at_k(local_labels, local_scores, 50),
                    }
                )
                prediction_rows.extend(
                    {
                        "case_id": case_id,
                        "residue_index": residue_index,
                        "label": int(label),
                        "label_mask": 1,
                        "protein_probability": float(score),
                        "predicted_label": int(predicted_label),
                    }
                    for residue_index, label, score, predicted_label in zip(
                        range(int(local_labels.size)), local_labels, local_scores, predicted_labels
                    )
                )
                offset += int(count)
    if not case_metrics:
        raise ValueError("GPSFun validation evaluation produced no cases")
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    case_recall = [float(row["recall"]) for row in case_metrics]
    metrics: dict[str, Any] = {
        "epoch": int(payload["epoch"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cases": len(case_metrics),
        "valid_residues": int(labels.size),
        "positive_residues": int(labels.sum()),
        "pooled_aupr": required(safe_aupr(labels, scores), "pooled AUPR"),
        "pooled_auroc": required(safe_auroc(labels, scores), "pooled AUROC"),
        "pooled_p_at_10": precision_at_k(labels, scores, 10),
        "pooled_p_at_20": precision_at_k(labels, scores, 20),
        "pooled_p_at_50": precision_at_k(labels, scores, 50),
        "case_mean_aupr": mean_metric(case_metrics, "aupr"),
        "case_mean_auroc": mean_metric(case_metrics, "auroc"),
        "case_mean_p_at_10": mean_metric(case_metrics, "p_at_10"),
        "case_mean_p_at_20": mean_metric(case_metrics, "p_at_20"),
        "case_mean_p_at_50": mean_metric(case_metrics, "p_at_50"),
        "case_median_aupr": median_metric(case_metrics, "aupr"),
        "case_median_auroc": median_metric(case_metrics, "auroc"),
        "case_median_p_at_10": median_metric(case_metrics, "p_at_10"),
        "case_median_p_at_20": median_metric(case_metrics, "p_at_20"),
        "case_median_p_at_50": median_metric(case_metrics, "p_at_50"),
        "macro_recall": float(np.mean(case_recall)),
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    atomic_write_csv(output_dir / "case_metrics.csv", case_metrics)
    atomic_write_csv(output_dir / "residue_predictions.csv", prediction_rows)
    return metrics


def safe_aupr(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, scores))


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if labels.size == 0:
        raise ValueError("precision@k is undefined for an empty case")
    selected = np.argsort(-scores, kind="stable")[: min(k, labels.size)]
    return float(labels[selected].mean())


def required(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is undefined for the pooled validation set")
    return value


def mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if row[key] is not None]
    if len(values) != len(rows):
        raise ValueError(f"checkpoint selection requires {key} for every validation case")
    return float(np.mean(values))


def median_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if row[key] is not None]
    if len(values) != len(rows):
        raise ValueError(f"checkpoint selection requires {key} for every validation case")
    return float(np.median(values))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
