"""Strict M4 evaluation over labeled manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

from ..data.loader import build_loader
from ..data.manifest import read_manifest
from ..data.splits import estimate_pos_weight
from ..models import build_m4_model, forward_m4_batch
from ..training.runtime import masked_bce, move_batch_to_device
from .metrics import aggregate_metrics, case_metrics, write_csv


HEAD_OUTPUT_KEYS = {
    "prior": "prior_logits",
    "surface_conditioned": "surface_conditioned_logits",
    "final": "logits",
}


def load_rows(path, states, max_cases=0):
    rows = read_manifest(path)
    if states:
        allowed = {value.strip() for value in states.split(",") if value.strip()}
        rows = [row for row in rows if row.get("input_state") in allowed]
    return rows[: int(max_cases)] if max_cases else rows


def calibration_metrics(scores, labels, bins=15):
    if scores.numel() == 0:
        return {"brier": float("nan"), "nll": float("nan"), "ece": float("nan")}
    scores = scores.float().clamp(1e-7, 1.0 - 1e-7)
    labels = labels.float()
    brier = (scores - labels).square().mean()
    nll = -(labels * scores.log() + (1.0 - labels) * (1.0 - scores).log()).mean()
    edges = torch.linspace(0.0, 1.0, int(bins) + 1)
    expected_calibration_error = scores.new_zeros(())
    for index in range(int(bins)):
        selected = (
            (scores >= edges[index]) & (scores <= edges[index + 1])
            if index + 1 == int(bins)
            else (scores >= edges[index]) & (scores < edges[index + 1])
        )
        if torch.any(selected):
            expected_calibration_error += selected.float().mean() * (
                scores[selected].mean() - labels[selected].mean()
            ).abs()
    return {
        "brier": float(brier),
        "nll": float(nll),
        "ece": float(expected_calibration_error),
    }


def unique_structure_weights(structure_keys, dtype=torch.float32):
    counts = {}
    for key in structure_keys:
        counts[key] = counts.get(key, 0) + 1
    return torch.tensor([1.0 / max(counts[key], 1) for key in structure_keys], dtype=dtype)


def weighted_binary_metrics(scores, labels, weights):
    if scores.numel() == 0:
        return {"aupr": float("nan"), "auroc": float("nan")}
    scores = scores.float()
    labels = labels.float()
    weights = weights.float()
    descending = torch.argsort(scores, descending=True, stable=True)
    ordered_labels = labels[descending]
    ordered_weights = weights[descending]
    true_positive = torch.cumsum(ordered_weights * ordered_labels, dim=0)
    false_positive = torch.cumsum(ordered_weights * (1.0 - ordered_labels), dim=0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1e-12)
    total_positive = (weights * labels).sum()
    aupr = (precision * ordered_weights * ordered_labels).sum() / total_positive.clamp_min(1e-12)

    ascending = torch.argsort(scores, descending=False, stable=True)
    ascending_scores = scores[ascending]
    ascending_labels = labels[ascending]
    ascending_weights = weights[ascending]
    _, inverse = torch.unique_consecutive(ascending_scores, return_inverse=True)
    group_count = int(inverse[-1].item()) + 1
    positive_by_group = scores.new_zeros(group_count)
    negative_by_group = scores.new_zeros(group_count)
    positive_by_group.index_add_(0, inverse, ascending_weights * ascending_labels)
    negative_by_group.index_add_(0, inverse, ascending_weights * (1.0 - ascending_labels))
    negative_before = torch.cumsum(negative_by_group, dim=0) - negative_by_group
    concordant = (positive_by_group * (negative_before + 0.5 * negative_by_group)).sum()
    total_negative = (weights * (1.0 - labels)).sum()
    auroc = concordant / (total_positive * total_negative).clamp_min(1e-12)
    return {"aupr": float(aupr), "auroc": float(auroc)}


def run_evaluation(model, loader, device, pos_weight, prefix, base_aux_loss_weight=0.2):
    model.eval()
    losses = []
    head_losses = {name: [] for name in HEAD_OUTPUT_KEYS}
    head_scores = {name: [] for name in HEAD_OUTPUT_KEYS}
    head_case_rows = {name: [] for name in HEAD_OUTPUT_KEYS}
    labels_all = []
    structure_keys = []
    skipped_examples = 0
    skipped_batches = 0
    gaussian_top_records = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"evaluate_{prefix}"):
            skipped_examples += int(batch.get("skipped_count", 0) or 0)
            if batch.get("skip_batch"):
                skipped_batches += 1
                continue
            batch = move_batch_to_device(batch, device)
            output = forward_m4_batch(model, batch)
            batch_losses = {
                name: masked_bce(output[key], batch["y"], batch["mask"], pos_weight)
                for name, key in HEAD_OUTPUT_KEYS.items()
            }
            loss = batch_losses["final"] + float(base_aux_loss_weight) * batch_losses["surface_conditioned"]
            losses.append(float(loss.cpu()))
            for name in HEAD_OUTPUT_KEYS:
                head_losses[name].append(float(batch_losses[name].cpu()))
            for record in output.get("gaussian_top_records", []) or []:
                record = dict(record)
                sample_index = int(record.get("sample_index", -1))
                if 0 <= sample_index < len(batch["rows"]):
                    row = batch["rows"][sample_index]
                    record.update(
                        {
                            "case_id": row.get("apex_case_id", row.get("system_id", str(sample_index))),
                            "system_id": row.get("system_id", ""),
                            "input_state": row.get("input_state", ""),
                            "receptor_uniprot": row.get("receptor_uniprot", ""),
                        }
                    )
                record["eval_prefix"] = prefix
                gaussian_top_records.append(record)
            offset = 0
            for sample_index, raw_count in enumerate(batch["residue_counts"].detach().cpu().tolist()):
                count = int(raw_count)
                section = slice(offset, offset + count)
                valid = batch["mask"][section] > 0.5
                if torch.any(valid):
                    labels = batch["y"][section][valid].detach().float().cpu()
                    labels_all.append(labels)
                    row = batch["rows"][sample_index]
                    case_id = row.get("apex_case_id", row.get("system_id", str(sample_index)))
                    structure_key = row.get("receptor_pdb_path", "") or case_id
                    structure_keys.extend([structure_key] * int(labels.numel()))
                    for name, key in HEAD_OUTPUT_KEYS.items():
                        scores = torch.sigmoid(output[key][section][valid]).detach().float().cpu()
                        head_scores[name].append(scores)
                        case_row = case_metrics(scores, labels, case_id)
                        case_row.update(
                            {
                                "head": name,
                                "system_id": row.get("system_id", ""),
                                "input_state": row.get("input_state", ""),
                                "receptor_uniprot": row.get("receptor_uniprot", ""),
                                "receptor_pdb_path": structure_key,
                            }
                        )
                        head_case_rows[name].append(case_row)
                offset += count
    labels = torch.cat(labels_all) if labels_all else torch.empty(0)
    final_scores = torch.cat(head_scores["final"]) if head_scores["final"] else torch.empty(0)
    metrics = aggregate_metrics(final_scores, labels, losses, head_case_rows["final"], prefix)
    weights = unique_structure_weights(structure_keys) if structure_keys else torch.empty(0)
    for name in HEAD_OUTPUT_KEYS:
        scores = torch.cat(head_scores[name]) if head_scores[name] else torch.empty(0)
        aggregate = aggregate_metrics(scores, labels, head_losses[name], head_case_rows[name], f"{prefix}_{name}")
        for key, value in aggregate.items():
            if key != "prefix":
                metrics[f"{name}_{key}"] = value
        for key, value in calibration_metrics(scores, labels).items():
            metrics[f"{name}_{key}"] = value
        weighted = (
            weighted_binary_metrics(scores, labels, weights)
            if weights.numel()
            else {"aupr": float("nan"), "auroc": float("nan")}
        )
        metrics[f"{name}_unique_structure_weighted_aupr"] = weighted["aupr"]
        metrics[f"{name}_unique_structure_weighted_auroc"] = weighted["auroc"]
    metrics["skipped_examples"] = skipped_examples
    metrics["skipped_batches"] = skipped_batches
    metrics["gaussian_top_record_count"] = len(gaussian_top_records)
    case_rows = [row for name in HEAD_OUTPUT_KEYS for row in head_case_rows[name]]
    return metrics, case_rows, gaussian_top_records


def _loader_args(args, checkpoint_args):
    def checkpoint_value(name, default):
        value = checkpoint_args.get(name, default)
        return default if value is None else value

    return SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        allow_residue_mismatch=args.allow_residue_mismatch,
        tensor_cache_dir=args.tensor_cache_dir,
        require_tensor_cache=True,
        residue_graph_cache_dir=args.residue_graph_cache_dir,
        require_residue_graph_cache=True,
        residue_graph_cache_ks=args.residue_graph_cache_ks,
        true_feature_cache_dir=args.true_feature_cache_dir,
        require_true_features=args.require_true_features,
        true_feature_version=args.true_feature_version,
        s1_surface_cache_dir=args.s1_surface_cache_dir,
        require_s1_surface_cache=args.require_s1_surface_cache,
        disable_s1_surface_on_the_fly=args.disable_s1_surface_on_the_fly,
        s1_surface_points=int(checkpoint_value("s1_surface_points", args.s1_surface_points)),
        s1_surface_graph_k=int(checkpoint_value("s1_surface_graph_k", args.s1_surface_graph_k)),
        s1_surface_lappe_dim=int(checkpoint_value("s1_surface_lappe_dim", args.s1_surface_lappe_dim)),
        s1_surface_cache_version=str(checkpoint_value("s1_surface_cache_version", args.s1_surface_cache_version)),
        teacher_cache_dir=args.teacher_cache_dir,
        teacher_name=args.teacher_name,
        teacher_cache_version=args.teacher_cache_version,
        require_teacher_cache=args.require_teacher_cache,
        use_soft_residue_labels=False,
        soft_label_sigma=8.0,
        soft_label_max_value=0.7,
        soft_label_cutoff=24.0,
        s1_static_geometry_cache_dir=args.s1_static_geometry_cache_dir,
        s1_static_geometry_cache_version=args.s1_static_geometry_cache_version,
        require_s1_static_geometry_cache=args.require_s1_static_geometry_cache,
    )


def evaluate(args):
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint must contain model_state_dict")
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint args must be a mapping")
    rows = load_rows(args.manifest, args.states, max_cases=args.max_cases)
    if not rows:
        raise ValueError("No evaluation rows remain after filtering")
    if args.train_manifest:
        train_rows = load_rows(args.train_manifest, "")
        pos_weight_value, label_stats = estimate_pos_weight(
            train_rows, factor=float(checkpoint_args.get("pos_weight_factor", 0.5))
        )
    else:
        pos_weight_value, label_stats = 1.0, {}
    loader = build_loader(rows, _loader_args(args, checkpoint_args), shuffle=False)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_m4_model(checkpoint_args).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=not args.allow_non_strict_load)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if (missing or unexpected) and not args.allow_non_strict_load:
        raise RuntimeError(f"Strict checkpoint load failed: missing={missing}, unexpected={unexpected}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "load_state_report.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "missing": missing, "unexpected": unexpected}, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics, case_rows, top_records = run_evaluation(
        model,
        loader,
        device,
        torch.tensor(float(pos_weight_value), dtype=torch.float),
        args.prefix,
        base_aux_loss_weight=float(checkpoint_args.get("base_aux_loss_weight", 0.2)),
    )
    skip_fraction = int(metrics.get("skipped_examples", 0) or 0) / max(len(rows), 1)
    if skip_fraction > args.max_skip_fraction:
        raise RuntimeError(
            f"Skip fraction {skip_fraction:.4f} exceeded --max-skip-fraction {args.max_skip_fraction:.4f}"
        )
    payload = {
        "metrics": metrics,
        "label_stats_from_train_manifest": label_stats,
        "pos_weight": pos_weight_value,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "num_manifest_rows": len(rows),
    }
    (output_dir / f"{args.prefix}_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / f"{args.prefix}_metrics.csv", [metrics])
    write_csv(output_dir / f"{args.prefix}_case_metrics.csv", case_rows)
    if top_records:
        (output_dir / f"{args.prefix}_gaussian_top_records.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in top_records),
            encoding="utf-8",
        )
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate M4 on a labeled residue manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", required=True)
    parser.add_argument("--s1-surface-cache-dir", default="")
    parser.add_argument("--s1-static-geometry-cache-dir", default="")
    parser.add_argument("--s1-static-geometry-cache-version", default="e89_dense_geometry_v1")
    parser.add_argument("--require-s1-static-geometry-cache", action="store_true")
    parser.add_argument("--true-feature-cache-dir", default="")
    parser.add_argument("--teacher-cache-dir", default="")
    parser.add_argument("--teacher-name", default="")
    parser.add_argument("--teacher-cache-version", default="teacher_prior_v1")
    parser.add_argument("--states", default="apo,af2")
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--allow-residue-mismatch", action="store_true")
    parser.add_argument(
        "--residue-graph-cache-ks",
        type=lambda text: tuple(int(value) for value in str(text).split(",") if value),
        default=(16, 24, 32),
    )
    parser.add_argument("--s1-surface-points", type=int, default=1024)
    parser.add_argument("--s1-surface-graph-k", type=int, default=16)
    parser.add_argument("--s1-surface-lappe-dim", type=int, default=0)
    parser.add_argument("--s1-surface-cache-version", default="s1_surface_shrake_v1")
    parser.add_argument("--require-s1-surface-cache", action="store_true")
    parser.add_argument("--disable-s1-surface-on-the-fly", action="store_true")
    parser.add_argument("--require-true-features", action="store_true")
    parser.add_argument("--true-feature-version", default="true_sasa_dssp_v1")
    parser.add_argument("--require-teacher-cache", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-skip-fraction", type=float, default=0.01)
    parser.add_argument("--allow-non-strict-load", action="store_true")
    return parser


def main():
    payload = evaluate(build_parser().parse_args())
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
