import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch as pt

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_dataset import ApexPairedManifestDataset, collate_apex_batch, esm_cache_path_for_key, read_manifest
from apex_model import ApexNet
from config import config_model
from pinder_manifest_dataset import write_csv


ESM2_CHOICES = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]


def optimizer_steps_per_epoch(num_batches, accumulation_steps):
    return math.ceil(num_batches / max(int(accumulation_steps), 1))


def build_lr_scheduler(args, optimizer, num_batches):
    total_steps = max(optimizer_steps_per_epoch(num_batches, args.gradient_accumulation_steps) * max(args.epochs, 1), 1)
    if args.max_steps:
        total_steps = min(total_steps, optimizer_steps_per_epoch(args.max_steps, args.gradient_accumulation_steps))
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and args.warmup_ratio > 0:
        warmup_steps = int(round(total_steps * args.warmup_ratio))
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    min_lr_ratio = max(float(args.min_learning_rate) / max(float(args.learning_rate), 1e-12), 0.0)

    if args.scheduler == "none":
        return None, {
            "scheduler": "none",
            "total_optimizer_steps": total_steps,
            "warmup_steps": warmup_steps,
            "min_learning_rate": args.min_learning_rate,
        }

    if args.scheduler == "plateau":
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.plateau_mode,
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_learning_rate,
        )
        return scheduler, {
            "scheduler": "plateau",
            "metric": args.plateau_metric,
            "mode": args.plateau_mode,
            "factor": args.plateau_factor,
            "patience": args.plateau_patience,
            "total_optimizer_steps": total_steps,
            "warmup_steps": 0,
            "min_learning_rate": args.min_learning_rate,
        }

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        if args.scheduler == "linear":
            value = 1.0 - progress
        elif args.scheduler == "cosine":
            value = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported scheduler: {args.scheduler}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * value

    scheduler = pt.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {
        "scheduler": args.scheduler,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "min_learning_rate": args.min_learning_rate,
    }


def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def unwrap_compiled_model(model):
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, optimizer_step, args):
    model_to_save = unwrap_compiled_model(model)
    payload = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "args": vars(args),
    }
    pt.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = pt.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return {
            "epoch": int(checkpoint.get("epoch", 0) or 0),
            "global_step": int(checkpoint.get("global_step", 0) or 0),
            "optimizer_step": int(checkpoint.get("optimizer_step", 0) or 0),
        }
    model.load_state_dict(checkpoint)
    return {"epoch": 0, "global_step": 0, "optimizer_step": 0}


def maybe_compile_model(model, args):
    if not getattr(args, "compile_model", False):
        return model, {"enabled": False}
    if not hasattr(pt, "compile"):
        raise RuntimeError("torch.compile was requested, but this PyTorch build does not provide torch.compile.")
    if args.partner_encoder == "esm2" and args.compile_allow_esm2 != "yes":
        raise RuntimeError(
            "Refusing full torch.compile with --partner-encoder esm2 because the ESM2 adapter uses Python "
            "sequence conversion and usually causes graph breaks/recompilation. Use --compile-allow-esm2 yes "
            "only for an explicit benchmark, or benchmark compile with --partner-encoder transformer first."
        )
    options = {}
    if args.compile_dynamic != "default":
        options["dynamic"] = args.compile_dynamic == "true"
    compiled = pt.compile(
        model,
        backend=args.compile_backend,
        mode=None if args.compile_mode == "default" else args.compile_mode,
        fullgraph=args.compile_fullgraph,
        **options,
    )
    return compiled, {
        "enabled": True,
        "backend": args.compile_backend,
        "mode": args.compile_mode,
        "fullgraph": args.compile_fullgraph,
        "dynamic": args.compile_dynamic,
        "allow_esm2": args.compile_allow_esm2,
    }


def dataloader_kwargs(args, shuffle):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "collate_fn": collate_apex_batch,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, pt.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list) and value and all(isinstance(x, pt.Tensor) for x in value):
            out[key] = [x.to(device, non_blocking=True) for x in value]
        else:
            out[key] = value
    return out


def update_skip_stats(stats, batch):
    skipped = int(batch.get("skipped_count", 0) or 0)
    stats["skipped_examples"] = stats.get("skipped_examples", 0) + skipped
    if batch.get("skip_batch"):
        stats["skipped_batches"] = stats.get("skipped_batches", 0) + 1
    for reason in batch.get("skip_reasons", [])[:3]:
        if reason and len(stats.setdefault("skip_examples", [])) < 10:
            stats["skip_examples"].append(reason)


def write_log_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def masked_bce(logits, labels, mask, pos_weight=None):
    valid = mask > 0.5
    if not pt.any(valid):
        return logits.sum() * 0.0
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight.to(device=logits.device, dtype=logits.dtype)
    return pt.nn.functional.binary_cross_entropy_with_logits(logits[valid], labels[valid], **kwargs)


def pair_bce(pair_logits, target):
    if pair_logits.numel() == 0:
        return pair_logits.sum() * 0.0
    target = target.to(device=pair_logits.device, dtype=pair_logits.dtype).view_as(pair_logits)
    return pt.nn.functional.binary_cross_entropy_with_logits(pair_logits, target)


def pair_margin_loss(true_pair_logits, shuffled_pair_logits, margin=0.2):
    """Whole-pair ranking loss for pseudo-negative shuffled partners.

    Shuffled partners do not have residue-level non-binding labels. This loss
    only says the true receptor-partner pair should score above the shuffled
    pair at the protein-pair level.
    """
    if true_pair_logits.numel() == 0 or shuffled_pair_logits.numel() == 0:
        return true_pair_logits.sum() * 0.0 + shuffled_pair_logits.sum() * 0.0
    diff = true_pair_logits.view(-1) - shuffled_pair_logits.view(-1)
    return pt.nn.functional.relu(float(margin) - diff).mean()


def load_negative_partner_pool(path):
    if not path:
        return {}
    pool = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys = [x for x in row.get("negative_partner_keys", "").split(";") if x]
            if keys:
                pool[row.get("apex_case_id", "")] = keys
    return pool


def load_negative_partner_sequences(path):
    if not path:
        return {}
    seqs = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("partner_key", "")
            seq = row.get("partner_sequence", "")
            if key and seq:
                seqs[key] = seq
    return seqs


def attach_cached_negative_partners(
    batch,
    negative_partner_pool,
    negative_partner_sequences,
    esm_cache_dir,
    esm_model_name,
    esm_repr_layer,
    esm_max_length,
    step=0,
):
    if not negative_partner_pool or not esm_cache_dir or "partner_esm_embedding" not in batch:
        return None
    rows = batch.get("rows") or []
    if not rows:
        return None
    embeddings = []
    masks = []
    for i, row in enumerate(rows):
        keys = negative_partner_pool.get(row.get("apex_case_id", ""))
        if not keys:
            return None
        key = keys[(int(step) + i) % len(keys)]
        seq = negative_partner_sequences.get(key)
        if seq:
            from apex_dataset import esm_cache_path_for_sequence

            path = esm_cache_path_for_sequence(
                esm_cache_dir,
                seq,
                model_name=esm_model_name,
                repr_layer=esm_repr_layer,
                max_length=esm_max_length,
            )
        else:
            path = esm_cache_path_for_key(esm_cache_dir, key)
        if not path.is_file():
            return None
        cache = pt.load(path, map_location="cpu")
        emb = cache["embedding"].to(dtype=batch["partner_esm_embedding"].dtype)
        embeddings.append(emb)
        masks.append(pt.ones((emb.shape[0],), dtype=batch["partner_esm_mask"].dtype))
    out = dict(batch)
    device = batch["partner_esm_embedding"].device
    out["partner_esm_embedding"] = pt.nn.utils.rnn.pad_sequence(
        embeddings,
        batch_first=True,
        padding_value=0.0,
    ).to(device=device, non_blocking=True)
    out["partner_esm_mask"] = pt.nn.utils.rnn.pad_sequence(
        masks,
        batch_first=True,
        padding_value=0.0,
    ).to(device=batch["partner_esm_mask"].device, non_blocking=True)
    return out


def shuffled_partner_batch(batch):
    n = int(batch["partner_seq_indices"].shape[0])
    if n < 2:
        return None
    device = batch["partner_seq_indices"].device
    arange = pt.arange(n, device=device)
    perm = pt.randperm(n, device=device)
    for _ in range(8):
        if not bool(pt.any(perm == arange)):
            break
        perm = pt.randperm(n, device=device)
    if bool(pt.any(perm == arange)):
        perm = pt.roll(arange, shifts=1)
    out = dict(batch)
    out["partner_seq_indices"] = batch["partner_seq_indices"][perm]
    out["partner_mask"] = batch["partner_mask"][perm]
    if "partner_esm_embedding" in batch:
        out["partner_esm_embedding"] = batch["partner_esm_embedding"][perm]
    if "partner_esm_mask" in batch:
        out["partner_esm_mask"] = batch["partner_esm_mask"][perm]
    return out


def zero_partner_batch(batch):
    out = dict(batch)
    out["partner_seq_indices"] = batch["partner_seq_indices"].new_full((batch["partner_seq_indices"].shape[0], 1), 20)
    out["partner_mask"] = batch["partner_mask"].new_ones((batch["partner_mask"].shape[0], 1))
    out.pop("partner_esm_embedding", None)
    out.pop("partner_esm_mask", None)
    return out


def model_forward(model, batch, partner_seq_indices=None, partner_mask=None):
    return model(
        batch["X"],
        batch["ids_topk"],
        batch["q"],
        batch["M"],
        batch["atom_counts"],
        batch["residue_counts"],
        batch["partner_seq_indices"] if partner_seq_indices is None else partner_seq_indices,
        batch["partner_mask"] if partner_mask is None else partner_mask,
        batch.get("receptor_seq_indices"),
        batch.get("receptor_mask"),
        batch.get("partner_esm_embedding"),
        batch.get("partner_esm_mask"),
        batch.get("residue_graph_caches"),
    )


def estimate_pos_weight(rows, factor=1.0):
    pos = 0
    valid = 0
    for row in rows:
        labels = [int(x) for x in row.get("labels", "").split(";") if x != ""]
        masks = [int(x) for x in row.get("label_mask", "").split(";") if x != ""]
        for y, m in zip(labels, masks):
            if m:
                valid += 1
                pos += int(y)
    rate = min(max(pos / max(valid, 1), 1e-4), 1.0)
    return factor * (1.0 - rate) / rate, {"valid": valid, "positive": pos, "positive_rate": rate}


def split_rows(rows, val_fraction=0.1, seed=42):
    systems = sorted({row["system_id"] for row in rows})
    g = pt.Generator()
    g.manual_seed(seed)
    perm = pt.randperm(len(systems), generator=g).tolist()
    shuffled = [systems[i] for i in perm]
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_systems = set(shuffled[:n_val])
    train = [row for row in rows if row["system_id"] not in val_systems]
    val = [row for row in rows if row["system_id"] in val_systems]
    return train, val


def split_rows_grouped(rows, val_fraction=0.1, seed=42, group_key="system_id"):
    if group_key == "system_id":
        return split_rows(rows, val_fraction=val_fraction, seed=seed)
    def key_for(row):
        if group_key == "cluster_id":
            return row.get("cluster_id") or row.get("system_id")
        if group_key == "cluster_id_RL":
            return "|".join(
                [
                    row.get("cluster_id", ""),
                    row.get("cluster_id_R", ""),
                    row.get("cluster_id_L", ""),
                ]
            ) or row.get("system_id")
        if group_key == "receptor_uniprot":
            side = row.get("receptor_side", "")
            return row.get("receptor_uniprot") or row.get(f"uniprot_{side}") or row.get("system_id")
        if group_key == "partner_uniprot":
            side = row.get("partner_side", "")
            return row.get("partner_uniprot") or row.get(f"uniprot_{side}") or row.get("system_id")
        raise ValueError(f"Unsupported val split group key: {group_key}")
    groups = sorted({key_for(row) for row in rows})
    g = pt.Generator()
    g.manual_seed(seed)
    perm = pt.randperm(len(groups), generator=g).tolist()
    shuffled = [groups[i] for i in perm]
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_groups = set(shuffled[:n_val])
    train = [row for row in rows if key_for(row) not in val_groups]
    val = [row for row in rows if key_for(row) in val_groups]
    return train, val


def binary_auc(scores, labels):
    if scores.numel() == 0:
        return float("nan")
    labels = labels.to(dtype=pt.long)
    n_pos = int(labels.sum().item())
    n = int(labels.numel())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = pt.argsort(scores)
    ranks = pt.empty_like(order, dtype=pt.float)
    ranks[order] = pt.arange(1, n + 1, device=scores.device, dtype=pt.float)
    pos_ranks = ranks[labels > 0].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc.cpu())


def binary_aupr(scores, labels):
    if scores.numel() == 0:
        return float("nan")
    labels = labels.to(dtype=pt.float)
    total_pos = labels.sum().clamp_min(1.0)
    if total_pos.item() <= 0:
        return float("nan")
    order = pt.argsort(scores, descending=True)
    y = labels[order]
    tp = pt.cumsum(y, dim=0)
    fp = pt.cumsum(1.0 - y, dim=0)
    recall = tp / total_pos
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = pt.cat([recall.new_zeros(1), recall])
    precision = pt.cat([precision.new_ones(1), precision])
    aupr = pt.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(aupr.cpu())


def run_eval(model, loader, device, pos_weight, prefix="val", eval_partner_ablations=False):
    model.eval()
    skip_stats = {"skipped_examples": 0, "skipped_batches": 0, "skip_examples": []}
    losses = []
    prior_losses = []
    all_scores = []
    all_prior_scores = []
    all_shuffled_scores = []
    all_zero_scores = []
    pair_scores = []
    shuffled_pair_scores = []
    all_labels = []
    valid_count = 0
    positive_count = 0
    with pt.no_grad():
        for batch in tqdm(loader, desc=f"eval_apex_{prefix}"):
            update_skip_stats(skip_stats, batch)
            if batch.get("skip_batch"):
                continue
            batch = move_batch_to_device(batch, device)
            out = model_forward(model, batch)
            losses.append(float(masked_bce(out["logit_conditioned"], batch["y"], batch["mask"], pos_weight).cpu()))
            prior_losses.append(float(masked_bce(out["logit_prior"], batch["y"], batch["mask"], pos_weight).cpu()))
            pair_scores.append(pt.sigmoid(out["pair_logit"]).detach().float().cpu())
            shuffled = shuffled_partner_batch(batch) if eval_partner_ablations else None
            if shuffled is not None:
                out_shuffled = model_forward(model, shuffled)
                shuffled_pair_scores.append(pt.sigmoid(out_shuffled["pair_logit"]).detach().float().cpu())
            else:
                out_shuffled = None
            if eval_partner_ablations:
                zero_batch = zero_partner_batch(batch)
                out_zero = model_forward(
                    model,
                    zero_batch,
                    partner_seq_indices=zero_batch["partner_seq_indices"],
                    partner_mask=zero_batch["partner_mask"],
                )
            else:
                out_zero = None
            valid = batch["mask"] > 0.5
            if pt.any(valid):
                labels = batch["y"][valid].detach().float().cpu()
                scores = pt.sigmoid(out["logit_conditioned"][valid]).detach().float().cpu()
                prior_scores = pt.sigmoid(out["logit_prior"][valid]).detach().float().cpu()
                if out_shuffled is not None:
                    shuffled_scores = pt.sigmoid(out_shuffled["logit_conditioned"][valid]).detach().float().cpu()
                    all_shuffled_scores.append(shuffled_scores)
                if out_zero is not None:
                    zero_scores = pt.sigmoid(out_zero["logit_conditioned"][valid]).detach().float().cpu()
                    all_zero_scores.append(zero_scores)
                all_labels.append(labels)
                all_scores.append(scores)
                all_prior_scores.append(prior_scores)
                valid_count += int(labels.numel())
                positive_count += int(labels.sum().item())
    labels = pt.cat(all_labels) if all_labels else pt.empty(0)
    scores = pt.cat(all_scores) if all_scores else pt.empty(0)
    prior_scores = pt.cat(all_prior_scores) if all_prior_scores else pt.empty(0)
    shuffled_scores = pt.cat(all_shuffled_scores) if all_shuffled_scores else pt.empty(0)
    zero_scores = pt.cat(all_zero_scores) if all_zero_scores else pt.empty(0)
    pair_scores = pt.cat(pair_scores) if pair_scores else pt.empty(0)
    shuffled_pair_scores = pt.cat(shuffled_pair_scores) if shuffled_pair_scores else pt.empty(0)
    return {
        f"{prefix}_loss_conditioned": sum(losses) / max(len(losses), 1),
        f"{prefix}_loss_prior": sum(prior_losses) / max(len(prior_losses), 1),
        f"{prefix}_aupr_conditioned": binary_aupr(scores, labels),
        f"{prefix}_auroc_conditioned": binary_auc(scores, labels),
        f"{prefix}_aupr_prior": binary_aupr(prior_scores, labels),
        f"{prefix}_auroc_prior": binary_auc(prior_scores, labels),
        f"{prefix}_aupr_shuffled_partner": binary_aupr(shuffled_scores, labels) if shuffled_scores.numel() == labels.numel() else float("nan"),
        f"{prefix}_aupr_zero_partner": binary_aupr(zero_scores, labels) if zero_scores.numel() == labels.numel() else float("nan"),
        f"{prefix}_pair_score_true_mean": float(pair_scores.mean().cpu()) if pair_scores.numel() else float("nan"),
        f"{prefix}_pair_score_shuffled_mean": float(shuffled_pair_scores.mean().cpu()) if shuffled_pair_scores.numel() else float("nan"),
        f"{prefix}_partner_sensitivity": float((scores - shuffled_scores).abs().mean().cpu()) if shuffled_scores.numel() == scores.numel() and scores.numel() else float("nan"),
        f"{prefix}_valid_residues": valid_count,
        f"{prefix}_positive_residues": positive_count,
        f"{prefix}_positive_rate": positive_count / max(valid_count, 1),
        f"{prefix}_skipped_examples": skip_stats["skipped_examples"],
        f"{prefix}_skipped_batches": skip_stats["skipped_batches"],
    }


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.matmul_precision:
        pt.set_float32_matmul_precision(args.matmul_precision)
    rows = read_manifest(args.manifest)
    if args.states:
        states = {x.strip() for x in args.states.split(",") if x.strip()}
        rows = [row for row in rows if row.get("input_state") in states]
    if args.max_cases:
        rows = rows[: args.max_cases]
    if not rows:
        raise RuntimeError("No training rows after filtering.")

    train_rows, val_rows = split_rows_grouped(rows, val_fraction=args.val_fraction, seed=args.seed, group_key=args.val_split_group)
    write_csv(output_dir / "train_manifest.csv", train_rows)
    write_csv(output_dir / "val_manifest.csv", val_rows)
    pos_weight_value, label_stats = estimate_pos_weight(train_rows, factor=args.pos_weight_factor)
    pos_weight = pt.tensor(pos_weight_value, dtype=pt.float)
    negative_partner_pool = load_negative_partner_pool(args.negative_partner_pool) if args.negative_partner_pool else {}
    negative_partner_sequences = (
        load_negative_partner_sequences(args.negative_partner_sequences) if args.negative_partner_sequences else {}
    )

    train_loader = pt.utils.data.DataLoader(
        ApexPairedManifestDataset(
            train_rows,
            strict_residue_alignment=not args.allow_residue_mismatch,
            skip_bad_examples=args.skip_bad_examples,
            tensor_cache_dir=args.tensor_cache_dir,
            require_tensor_cache=args.require_tensor_cache,
            esm_cache_dir=args.esm_cache_dir,
            require_esm_cache=args.require_esm_cache,
            esm_model_name=args.esm_model_name,
            esm_repr_layer=args.esm_repr_layer or ESM2_SPECS[args.esm_model_name]["layers"],
            esm_max_length=args.esm_max_length,
            residue_graph_cache_dir=args.residue_graph_cache_dir,
            require_residue_graph_cache=args.require_residue_graph_cache,
            residue_graph_ks=args.residue_graph_cache_ks,
        ),
        **dataloader_kwargs(args, shuffle=True),
    )
    val_loader = pt.utils.data.DataLoader(
        ApexPairedManifestDataset(
            val_rows,
            strict_residue_alignment=not args.allow_residue_mismatch,
            skip_bad_examples=args.skip_bad_examples,
            tensor_cache_dir=args.tensor_cache_dir,
            require_tensor_cache=args.require_tensor_cache,
            esm_cache_dir=args.esm_cache_dir,
            require_esm_cache=args.require_esm_cache,
            esm_model_name=args.esm_model_name,
            esm_repr_layer=args.esm_repr_layer or ESM2_SPECS[args.esm_model_name]["layers"],
            esm_max_length=args.esm_max_length,
            residue_graph_cache_dir=args.residue_graph_cache_dir,
            require_residue_graph_cache=args.require_residue_graph_cache,
            residue_graph_ks=args.residue_graph_cache_ks,
        ),
        **dataloader_kwargs(args, shuffle=False),
    )
    test_rows = []
    test_loader = None
    if args.test_manifest:
        test_rows = read_manifest(args.test_manifest)
        if args.test_states:
            test_states = {x.strip() for x in args.test_states.split(",") if x.strip()}
            test_rows = [row for row in test_rows if row.get("input_state") in test_states]
        if args.max_test_cases:
            test_rows = test_rows[: args.max_test_cases]
        write_csv(output_dir / "test_manifest.csv", test_rows)
        test_loader = pt.utils.data.DataLoader(
            ApexPairedManifestDataset(
                test_rows,
                strict_residue_alignment=not args.allow_residue_mismatch,
                skip_bad_examples=args.skip_bad_examples,
                tensor_cache_dir=args.tensor_cache_dir,
                require_tensor_cache=args.require_tensor_cache,
                esm_cache_dir=args.esm_cache_dir,
                require_esm_cache=args.require_esm_cache,
                esm_model_name=args.esm_model_name,
                esm_repr_layer=args.esm_repr_layer or ESM2_SPECS[args.esm_model_name]["layers"],
                esm_max_length=args.esm_max_length,
                residue_graph_cache_dir=args.residue_graph_cache_dir,
                require_residue_graph_cache=args.require_residue_graph_cache,
                residue_graph_ks=args.residue_graph_cache_ks,
            ),
            **dataloader_kwargs(args, shuffle=False),
        )

    device = pt.device(args.device if args.device == "cpu" or pt.cuda.is_available() else "cpu")
    model = ApexNet(
        config_model,
        dim=args.dim,
        num_queries=args.num_queries,
        num_blocks=args.num_blocks,
        heads=args.heads,
        surface_fraction=args.surface_fraction,
        max_surface=args.max_surface,
        min_surface=args.min_surface,
        residue_graph_k=args.residue_graph_k,
        residue_graph_layers=args.residue_graph_layers,
        structure_encoder=args.structure_encoder,
        gvp_layers=args.gvp_layers,
        gvp_vector_dim=args.gvp_vector_dim,
        partner_encoder=args.partner_encoder,
        esm_model_name=args.esm_model_name,
        freeze_esm=not args.finetune_esm,
        esm_max_length=args.esm_max_length,
        esm_repr_layer=args.esm_repr_layer,
        receptor_sequence_encoder=args.receptor_sequence_encoder,
        partner_dropout=args.partner_dropout,
        use_cached_partner_esm=bool(args.esm_cache_dir and args.partner_encoder == "esm2"),
    ).to(device)
    model, compile_info = maybe_compile_model(model, args)
    optimizer = pt.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler, scheduler_info = build_lr_scheduler(args, optimizer, len(train_loader))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = pt.cuda.amp.GradScaler(enabled=use_amp)
    report = {
        "args": vars(args),
        "device": str(device),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "num_test_rows": len(test_rows),
        "label_stats": label_stats,
        "pos_weight": pos_weight_value,
        "model": {
            "dim": args.dim,
            "num_queries": args.num_queries,
            "num_blocks": args.num_blocks,
            "heads": args.heads,
            "surface_fraction": args.surface_fraction,
            "min_surface": args.min_surface,
            "max_surface": args.max_surface,
            "residue_graph_k": args.residue_graph_k,
            "residue_graph_layers": args.residue_graph_layers,
            "structure_encoder": args.structure_encoder,
            "gvp_layers": args.gvp_layers,
            "gvp_vector_dim": args.gvp_vector_dim,
            "partner_encoder": args.partner_encoder,
            "esm_model_name": args.esm_model_name if args.partner_encoder == "esm2" else None,
            "freeze_esm": not args.finetune_esm,
            "esm_max_length": args.esm_max_length,
            "esm_repr_layer": args.esm_repr_layer,
            "receptor_sequence_encoder": args.receptor_sequence_encoder,
            "use_cached_partner_esm": bool(args.esm_cache_dir and args.partner_encoder == "esm2"),
            "residue_graph_cache_dir": args.residue_graph_cache_dir,
            "require_residue_graph_cache": args.require_residue_graph_cache,
            "residue_graph_cache_ks": args.residue_graph_cache_ks,
            "negative_partner_pool": args.negative_partner_pool,
            "negative_partner_sequences": args.negative_partner_sequences,
            "num_negative_partner_pool_rows": len(negative_partner_pool),
            "num_negative_partner_sequences": len(negative_partner_sequences),
            "amp": args.amp,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "partner_dropout": args.partner_dropout,
            "val_split_group": args.val_split_group,
            "surface_loss_weight": args.surface_loss_weight,
            "pair_loss_weight": args.pair_loss_weight,
            "shuffled_pair_loss_weight": args.shuffled_pair_loss_weight,
            "partner_margin_loss_weight": args.partner_margin_loss_weight,
            "partner_margin": args.partner_margin,
            "interaction_delta_loss_weight": args.interaction_delta_loss_weight,
            "delta_l1_weight": args.delta_l1_weight,
            "skip_bad_examples": args.skip_bad_examples,
            "compile": compile_info,
            "eval_partner_ablations": args.eval_partner_ablations,
            "shuffled_partner_interval": args.shuffled_partner_interval,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "matmul_precision": args.matmul_precision,
            "tensor_cache_dir": args.tensor_cache_dir,
            "require_tensor_cache": args.require_tensor_cache,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            **scheduler_info,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    step_rows = []
    global_step = 0
    optimizer_step = 0
    start_epoch = 0
    train_skip_stats = {"skipped_examples": 0, "skipped_batches": 0, "skip_examples": []}
    start_time = time.time()
    if args.resume_checkpoint:
        resume_state = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, scaler, device=device)
        start_epoch = resume_state["epoch"] if args.resume_training_state else 0
        global_step = resume_state["global_step"] if args.resume_training_state else 0
        optimizer_step = resume_state["optimizer_step"] if args.resume_training_state else 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"train_apex_e{epoch + 1}")):
            step_start = time.time()
            update_skip_stats(train_skip_stats, batch)
            if batch.get("skip_batch"):
                continue
            batch = move_batch_to_device(batch, device)
            with pt.cuda.amp.autocast(enabled=use_amp):
                out = model_forward(model, batch)
                loss_cond = masked_bce(out["logit_conditioned"], batch["y"], batch["mask"], pos_weight)
                loss_prior = masked_bce(out["logit_prior"], batch["y"], batch["mask"], pos_weight)
                loss_surface = masked_bce(out["candidate_scores"], batch["y"], batch["mask"], pos_weight)
                true_pair_target = pt.ones_like(out["pair_logit"])
                loss_pair_true = pair_bce(out["pair_logit"], true_pair_target)
                loss_pair_shuffled = out["pair_logit"].sum() * 0.0
                loss_partner_margin = out["pair_logit"].sum() * 0.0
                loss_interaction_delta = masked_bce(out["logit_delta"], batch["y"], batch["mask"], pos_weight)
                use_shuffled_partner = (
                    args.shuffled_partner_interval > 0
                    and (global_step + 1) % args.shuffled_partner_interval == 0
                )
                shuffled = None
                shuffled_source = ""
                if use_shuffled_partner:
                    shuffled = attach_cached_negative_partners(
                        batch,
                        negative_partner_pool,
                        negative_partner_sequences,
                        args.esm_cache_dir,
                        args.esm_model_name,
                        args.esm_repr_layer or ESM2_SPECS[args.esm_model_name]["layers"],
                        args.esm_max_length,
                        step=global_step + 1,
                    )
                    if shuffled is not None:
                        shuffled_source = "negative_pool"
                    else:
                        shuffled = shuffled_partner_batch(batch)
                        shuffled_source = "batch_shuffle" if shuffled is not None else ""
                if shuffled is not None and (
                    args.shuffled_pair_loss_weight > 0
                    or args.partner_margin_loss_weight > 0
                ):
                    out_shuffled = model_forward(model, shuffled)
                    false_pair_target = pt.zeros_like(out_shuffled["pair_logit"])
                    loss_pair_shuffled = pair_bce(out_shuffled["pair_logit"], false_pair_target)
                    loss_partner_margin = pair_margin_loss(
                        out["pair_logit"],
                        out_shuffled["pair_logit"],
                        margin=args.partner_margin,
                    )
                loss_delta_l1 = out["logit_delta"].abs().mean()
                loss = (
                    loss_cond
                    + args.prior_loss_weight * loss_prior
                    + args.surface_loss_weight * loss_surface
                    + args.pair_loss_weight * loss_pair_true
                    + args.shuffled_pair_loss_weight * loss_pair_shuffled
                    + args.partner_margin_loss_weight * loss_partner_margin
                    + args.interaction_delta_loss_weight * loss_interaction_delta
                    + args.delta_l1_weight * loss_delta_l1
                )
                loss_to_backward = loss / max(args.gradient_accumulation_steps, 1)
            scaler.scale(loss_to_backward).backward()
            should_step = (global_step + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and args.scheduler in {"cosine", "linear"}:
                    scheduler.step()
                optimizer_step += 1
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                row = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "loss": f"{float(loss.detach().cpu()):.8f}",
                    "loss_conditioned": f"{float(loss_cond.detach().cpu()):.8f}",
                    "loss_prior": f"{float(loss_prior.detach().cpu()):.8f}",
                    "loss_surface": f"{float(loss_surface.detach().cpu()):.8f}",
                    "loss_pair_true": f"{float(loss_pair_true.detach().cpu()):.8f}",
                    "loss_pair_shuffled": f"{float(loss_pair_shuffled.detach().cpu()):.8f}",
                    "loss_partner_margin": f"{float(loss_partner_margin.detach().cpu()):.8f}",
                    "loss_interaction_delta": f"{float(loss_interaction_delta.detach().cpu()):.8f}",
                    "loss_delta_l1": f"{float(loss_delta_l1.detach().cpu()):.8f}",
                    "used_shuffled_partner": int(bool(shuffled is not None)),
                    "shuffled_partner_source": shuffled_source,
                    "step_seconds": f"{time.time() - step_start:.4f}",
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "steps_per_hour": f"{global_step / max(elapsed, 1e-6) * 3600.0:.2f}",
                    "optimizer_step": optimizer_step,
                    "learning_rate": f"{current_lr(optimizer):.10g}",
                    "skipped_examples": train_skip_stats["skipped_examples"],
                    "skipped_batches": train_skip_stats["skipped_batches"],
                }
                step_rows.append(row)
                write_log_csv(output_dir / "train_step_log.csv", step_rows)
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(
                    output_dir / f"model_step_{global_step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    global_step,
                    optimizer_step,
                    args,
                )
            if args.eval_every and global_step % args.eval_every == 0:
                save_checkpoint(
                    output_dir / f"model_step_{global_step}_pre_eval.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    global_step,
                    optimizer_step,
                    args,
                )
                metrics = run_eval(
                    model,
                    val_loader,
                    device,
                    pos_weight,
                    prefix="val",
                    eval_partner_ablations=args.eval_partner_ablations,
                )
                step_rows[-1].update({k: f"{v:.8f}" if math.isfinite(v) else "nan" for k, v in metrics.items()})
                if test_loader is not None and args.test_every and global_step % args.test_every == 0:
                    test_metrics = run_eval(
                        model,
                        test_loader,
                        device,
                        pos_weight,
                        prefix="test",
                        eval_partner_ablations=args.eval_partner_ablations,
                    )
                    step_rows[-1].update(
                        {k: f"{v:.8f}" if isinstance(v, float) and math.isfinite(v) else str(v) for k, v in test_metrics.items()}
                    )
                write_log_csv(output_dir / "train_step_log.csv", step_rows)
                save_checkpoint(
                    output_dir / f"model_step_{global_step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    global_step,
                    optimizer_step,
                    args,
                )
            if args.max_steps and global_step >= args.max_steps:
                if test_loader is not None:
                    metrics = run_eval(
                        model,
                        test_loader,
                        device,
                        pos_weight,
                        prefix="test_final",
                        eval_partner_ablations=args.eval_partner_ablations,
                    )
                    (output_dir / "test_final_metrics.json").write_text(
                        json.dumps(metrics, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                save_checkpoint(output_dir / "model_last.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, optimizer_step, args)
                return
        if args.eval_every_epoch:
            elapsed = time.time() - start_time
            save_checkpoint(
                output_dir / f"model_epoch_{epoch + 1}_pre_eval.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                global_step,
                optimizer_step,
                args,
            )
            row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "loss": "",
                "loss_conditioned": "",
                "loss_prior": "",
                "step_seconds": "",
                "elapsed_seconds": f"{elapsed:.2f}",
                "steps_per_hour": f"{global_step / max(elapsed, 1e-6) * 3600.0:.2f}",
                "optimizer_step": optimizer_step,
                "learning_rate": f"{current_lr(optimizer):.10g}",
                "skipped_examples": train_skip_stats["skipped_examples"],
                "skipped_batches": train_skip_stats["skipped_batches"],
            }
            metrics = run_eval(
                model,
                val_loader,
                device,
                pos_weight,
                prefix="val",
                eval_partner_ablations=args.eval_partner_ablations,
            )
            row.update({k: f"{v:.8f}" if math.isfinite(v) else "nan" for k, v in metrics.items()})
            if test_loader is not None and args.test_every_epoch:
                test_metrics = run_eval(
                    model,
                    test_loader,
                    device,
                    pos_weight,
                    prefix="test",
                    eval_partner_ablations=args.eval_partner_ablations,
                )
                row.update(
                    {k: f"{v:.8f}" if isinstance(v, float) and math.isfinite(v) else str(v) for k, v in test_metrics.items()}
                )
            if scheduler is not None and args.scheduler == "plateau":
                metric_value = row.get(args.plateau_metric)
                if metric_value not in (None, "", "nan"):
                    scheduler.step(float(metric_value))
            step_rows.append(row)
            write_log_csv(output_dir / "train_step_log.csv", step_rows)
        if args.save_every_epoch:
            save_checkpoint(
                output_dir / f"model_epoch_{epoch + 1}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                global_step,
                optimizer_step,
                args,
            )
    if test_loader is not None:
        metrics = run_eval(
            model,
            test_loader,
            device,
            pos_weight,
            prefix="test_final",
            eval_partner_ablations=args.eval_partner_ablations,
        )
        (output_dir / "test_final_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    save_checkpoint(output_dir / "model_last.pt", model, optimizer, scheduler, scaler, args.epochs, global_step, optimizer_step, args)


def main():
    parser = argparse.ArgumentParser(description="Train APEX-Net on paired Pinder manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-manifest", default="")
    parser.add_argument(
        "--tensor-cache-dir",
        default="",
        help="Directory containing precomputed APEX tensor .pt files keyed by manifest rows.",
    )
    parser.add_argument(
        "--require-tensor-cache",
        action="store_true",
        help="Fail or skip bad examples if a row has no precomputed tensor cache file.",
    )
    parser.add_argument(
        "--esm-cache-dir",
        default="",
        help="Directory containing precomputed partner ESM embeddings from precompute_apex_esm_cache.py.",
    )
    parser.add_argument(
        "--require-esm-cache",
        action="store_true",
        help="Fail or skip bad examples if a row has no precomputed partner ESM embedding.",
    )
    parser.add_argument(
        "--residue-graph-cache-dir",
        default="",
        help="Directory containing precomputed residue kNN graph .pt files from precompute_apex_residue_graph_cache.py.",
    )
    parser.add_argument(
        "--require-residue-graph-cache",
        action="store_true",
        help="Fail or skip bad examples if a row has no precomputed residue graph cache file.",
    )
    parser.add_argument(
        "--residue-graph-cache-ks",
        type=lambda text: tuple(int(x) for x in str(text).split(",") if x),
        default=(16, 24, 32),
        help="Comma-separated k values used when building residue graph cache. Must match the cache directory.",
    )
    parser.add_argument(
        "--negative-partner-pool",
        default="",
        help=(
            "CSV from build_apex_negative_partner_pool.py. Used only for whole-pair contrastive "
            "pseudo-negatives, never as residue-level labels."
        ),
    )
    parser.add_argument(
        "--negative-partner-sequences",
        default="",
        help="unique_partner_sequences.csv paired with --negative-partner-pool, used to resolve partner keys to ESM cache files.",
    )
    parser.add_argument("--states", default="apo,af2")
    parser.add_argument("--test-states", default="apo,af2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-test-cases", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned host memory for CUDA DataLoader transfers.")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when --num-workers > 0.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--matmul-precision",
        choices=["", "highest", "high", "medium"],
        default="high",
        help="torch.set_float32_matmul_precision value. Empty string leaves PyTorch default unchanged.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "linear", "plateau"],
        default="none",
        help="Learning-rate scheduler. cosine/linear step after optimizer updates; plateau steps after epoch eval.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--plateau-metric", default="val_loss_conditioned")
    parser.add_argument("--plateau-mode", choices=["min", "max"], default="min")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--pos-weight-factor", type=float, default=1.0)
    parser.add_argument("--prior-loss-weight", type=float, default=0.02)
    parser.add_argument("--surface-loss-weight", type=float, default=0.0)
    parser.add_argument("--pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--shuffled-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--partner-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--partner-margin", type=float, default=0.3)
    parser.add_argument(
        "--shuffled-partner-interval",
        type=int,
        default=6,
        help=(
            "Run the extra shuffled-partner whole-pair contrastive forward every N training steps. "
            "With batch_size=16 and N=6, one contrastive batch is used per 96 real samples."
        ),
    )
    parser.add_argument("--interaction-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--delta-l1-weight", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--val-split-group",
        choices=["system_id", "cluster_id", "cluster_id_RL", "receptor_uniprot", "partner_uniprot"],
        default="cluster_id",
        help="Group key used to split train/val. cluster_id is safer than random system_id for generalization checks.",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--surface-fraction", type=float, default=0.5)
    parser.add_argument("--min-surface", type=int, default=128)
    parser.add_argument("--max-surface", type=int, default=512)
    parser.add_argument("--residue-graph-k", type=int, default=24)
    parser.add_argument("--residue-graph-layers", type=int, default=2)
    parser.add_argument(
        "--structure-encoder",
        choices=["full_gvp", "pure_gvp", "pesto", "pesto_graph", "pesto_gvp", "pesto_graph_gvp"],
        default="full_gvp",
        help=(
            "Receptor structure branch. full_gvp is the main APEX receptor encoder with full scalar/vector GVP-GNN; "
            "pure_gvp is a simplified GVP-style encoder without PeSTo backbone; "
            "pesto uses only PeSTo residue features; "
            "pesto_graph adds scalar residue graph refinement; pesto_gvp adds GVP-style vector/scalar refinement; "
            "pesto_graph_gvp uses both."
        ),
    )
    parser.add_argument("--gvp-layers", type=int, default=3)
    parser.add_argument("--gvp-vector-dim", type=int, default=8)
    parser.add_argument(
        "--partner-encoder",
        choices=["transformer", "esm2"],
        default="transformer",
        help="Partner sequence encoder. transformer is lightweight; esm2 calls fair-esm pretrained ESM-2.",
    )
    parser.add_argument(
        "--esm-model-name",
        choices=ESM2_CHOICES,
        default="esm2_t33_650M_UR50D",
        help="ESM-2 checkpoint used when --partner-encoder esm2.",
    )
    parser.add_argument(
        "--finetune-esm",
        action="store_true",
        help="Fine-tune ESM-2 instead of using it as a frozen partner-sequence encoder.",
    )
    parser.add_argument(
        "--esm-max-length",
        type=int,
        default=1022,
        help="Maximum partner residues passed to ESM-2. Longer partners are truncated for memory safety.",
    )
    parser.add_argument(
        "--esm-repr-layer",
        type=int,
        default=None,
        help="Optional ESM representation layer. Default uses the final layer for the selected ESM-2 model.",
    )
    parser.add_argument(
        "--receptor-sequence-encoder",
        choices=["none", "shared_partner", "esm2", "transformer"],
        default="none",
        help="Optional receptor sequence encoder fused into receptor structural embeddings.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision autocast and GradScaler.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help=(
            "Optionally wrap the full model in torch.compile. Benchmark first: variable protein lengths and "
            "the ESM2 adapter can trigger graph breaks or recompilation."
        ),
    )
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="default",
    )
    parser.add_argument(
        "--compile-dynamic",
        choices=["default", "true", "false"],
        default="default",
        help="Passed to torch.compile(dynamic=...). default lets PyTorch decide.",
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument(
        "--compile-allow-esm2",
        choices=["no", "yes"],
        default="no",
        help="Set yes only for explicit full-model compile benchmarks with the ESM2 adapter.",
    )
    parser.add_argument("--partner-dropout", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save a training-state checkpoint every N optimizer/training steps. 0 disables step-only periodic saves.",
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--test-every", type=int, default=0)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--test-every-epoch", action="store_true")
    parser.add_argument(
        "--eval-partner-ablations",
        action="store_true",
        help="During eval, also run shuffled-partner and zero-partner ablations. This roughly triples eval time.",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help="When resume checkpoint contains optimizer/scheduler/scaler state, continue epoch/step counters instead of loading weights only.",
    )
    parser.add_argument(
        "--skip-bad-examples",
        action="store_true",
        help="Skip samples whose encoded receptor residue count does not match labels/mask instead of raising from DataLoader.",
    )
    parser.add_argument("--allow-residue-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
