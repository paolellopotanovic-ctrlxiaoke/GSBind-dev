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

from apex_dataset import ApexPairedManifestDataset, collate_apex_batch, esm_cache_path_for_sequence, read_manifest
from apex_model import ESM2_MODEL_SPECS
from apex_simple_model import SimplePartnerGVPESMNet
from pinder_manifest_dataset import write_csv
from train_apex import binary_auc, binary_aupr, estimate_pos_weight, split_rows_grouped


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
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            moved = []
            for cache in value:
                moved.append({k: (v.to(device, non_blocking=True) if isinstance(v, pt.Tensor) else v) for k, v in cache.items()})
            out[key] = moved
        else:
            out[key] = value
    return out


def masked_bce(logits, labels, mask, pos_weight=None):
    valid = mask > 0.5
    if not pt.any(valid):
        return logits.sum() * 0.0
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight.to(device=logits.device, dtype=logits.dtype)
    return pt.nn.functional.binary_cross_entropy_with_logits(logits[valid], labels[valid], **kwargs)


def pair_bce(pair_logits, target):
    target = target.to(device=pair_logits.device, dtype=pair_logits.dtype).view_as(pair_logits)
    return pt.nn.functional.binary_cross_entropy_with_logits(pair_logits, target)


def pair_margin_loss(true_pair_logits, shuffled_pair_logits, margin=0.2):
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


def attach_cached_negative_partners(batch, negative_partner_pool, negative_partner_sequences, args, step=0):
    if not negative_partner_pool or not args.esm_cache_dir or "partner_esm_embedding" not in batch:
        return None
    rows = batch.get("rows") or []
    if not rows:
        return None
    embeddings = []
    masks = []
    repr_layer = args.esm_repr_layer or ESM2_MODEL_SPECS[args.esm_model_name]["layers"]
    for i, row in enumerate(rows):
        keys = negative_partner_pool.get(row.get("apex_case_id", ""))
        if not keys:
            return None
        key = keys[(int(step) + i) % len(keys)]
        seq = negative_partner_sequences.get(key)
        if not seq:
            return None
        path = esm_cache_path_for_sequence(
            args.esm_cache_dir,
            seq,
            model_name=args.esm_model_name,
            repr_layer=repr_layer,
            max_length=args.esm_max_length,
        )
        if not path.is_file():
            return None
        cache = pt.load(path, map_location="cpu")
        emb = cache["embedding"].to(dtype=batch["partner_esm_embedding"].dtype)
        embeddings.append(emb)
        masks.append(pt.ones((emb.shape[0],), dtype=batch["partner_esm_mask"].dtype))
    out = dict(batch)
    device = batch["partner_esm_embedding"].device
    out["partner_esm_embedding"] = pt.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.0).to(device=device)
    out["partner_esm_mask"] = pt.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0.0).to(device=batch["partner_esm_mask"].device)
    return out


def build_loader(rows, args, shuffle):
    repr_layer = args.esm_repr_layer or ESM2_MODEL_SPECS[args.esm_model_name]["layers"]
    return pt.utils.data.DataLoader(
        ApexPairedManifestDataset(
            rows,
            strict_residue_alignment=not args.allow_residue_mismatch,
            skip_bad_examples=True,
            tensor_cache_dir=args.tensor_cache_dir,
            require_tensor_cache=True,
            esm_cache_dir=args.esm_cache_dir,
            require_esm_cache=True,
            esm_model_name=args.esm_model_name,
            esm_repr_layer=repr_layer,
            esm_max_length=args.esm_max_length,
            residue_graph_cache_dir=args.residue_graph_cache_dir,
            require_residue_graph_cache=True,
            residue_graph_ks=args.residue_graph_cache_ks,
        ),
        **dataloader_kwargs(args, shuffle=shuffle),
    )


def run_eval(model, loader, device, pos_weight, prefix="val"):
    model.eval()
    losses = []
    all_scores = []
    all_labels = []
    valid_count = 0
    positive_count = 0
    skipped_examples = 0
    skipped_batches = 0
    with pt.no_grad():
        for batch in tqdm(loader, desc=f"eval_simple_{prefix}"):
            skipped_examples += int(batch.get("skipped_count", 0) or 0)
            if batch.get("skip_batch"):
                skipped_batches += 1
                continue
            batch = move_batch_to_device(batch, device)
            out = model_forward(model, batch)
            loss = masked_bce(out["logits"], batch["y"], batch["mask"], pos_weight)
            losses.append(float(loss.cpu()))
            valid = batch["mask"] > 0.5
            if pt.any(valid):
                labels = batch["y"][valid].detach().float().cpu()
                scores = pt.sigmoid(out["logits"][valid]).detach().float().cpu()
                all_labels.append(labels)
                all_scores.append(scores)
                valid_count += int(labels.numel())
                positive_count += int(labels.sum().item())
    labels = pt.cat(all_labels) if all_labels else pt.empty(0)
    scores = pt.cat(all_scores) if all_scores else pt.empty(0)
    return {
        f"{prefix}_loss": sum(losses) / max(len(losses), 1),
        f"{prefix}_aupr": binary_aupr(scores, labels),
        f"{prefix}_auroc": binary_auc(scores, labels),
        f"{prefix}_valid_residues": valid_count,
        f"{prefix}_positive_residues": positive_count,
        f"{prefix}_positive_rate": positive_count / max(valid_count, 1),
        f"{prefix}_skipped_examples": skipped_examples,
        f"{prefix}_skipped_batches": skipped_batches,
    }


def model_forward(model, batch):
    return model(
        batch["X"],
        batch["ids_topk"],
        batch["q"],
        batch["M"],
        batch["atom_counts"],
        batch["residue_counts"],
        batch["partner_seq_indices"],
        batch["partner_mask"],
        batch.get("partner_esm_embedding"),
        batch.get("partner_esm_mask"),
        batch.get("residue_graph_caches"),
    )


def build_scheduler(args, optimizer, num_batches):
    total_steps = max(num_batches * max(args.epochs, 1), 1)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(round(total_steps * args.warmup_ratio)) if args.warmup_steps <= 0 else args.warmup_steps
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    min_ratio = args.min_learning_rate / max(args.learning_rate, 1e-12)
    if args.scheduler == "none":
        return None, {"scheduler": "none", "total_steps": total_steps, "warmup_steps": warmup_steps}

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        if args.scheduler == "linear":
            value = 1.0 - progress
        else:
            value = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * value

    return pt.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), {
        "scheduler": args.scheduler,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, args):
    pt.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "args": vars(args),
        },
        path,
    )


def write_log(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_log(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.matmul_precision:
        pt.set_float32_matmul_precision(args.matmul_precision)
    rows = read_manifest(args.manifest)
    states = {x.strip() for x in args.states.split(",") if x.strip()}
    rows = [row for row in rows if row.get("input_state") in states]
    if args.max_cases:
        rows = rows[: args.max_cases]
    train_rows, val_rows = split_rows_grouped(rows, val_fraction=args.val_fraction, seed=args.seed, group_key=args.val_split_group)
    write_csv(output_dir / "train_manifest.csv", train_rows)
    write_csv(output_dir / "val_manifest.csv", val_rows)
    pos_weight_value, label_stats = estimate_pos_weight(train_rows, factor=args.pos_weight_factor)
    pos_weight = pt.tensor(pos_weight_value, dtype=pt.float)
    train_loader = build_loader(train_rows, args, shuffle=True)
    val_loader = build_loader(val_rows, args, shuffle=False)
    negative_partner_pool = load_negative_partner_pool(args.negative_partner_pool)
    negative_partner_sequences = load_negative_partner_sequences(args.negative_partner_sequences)

    device = pt.device(args.device if args.device == "cpu" or pt.cuda.is_available() else "cpu")
    model = SimplePartnerGVPESMNet(
        dim=args.dim,
        heads=args.heads,
        num_blocks=args.num_blocks,
        gvp_layers=args.gvp_layers,
        gvp_vector_dim=args.gvp_vector_dim,
        residue_graph_k=args.residue_graph_k,
        esm_model_name=args.esm_model_name,
        dropout=args.dropout,
        partner_updates=not args.disable_partner_updates,
    ).to(device)
    optimizer = pt.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler, scheduler_info = build_scheduler(args, optimizer, len(train_loader))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = pt.cuda.amp.GradScaler(enabled=use_amp)
    resume_state = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        resume_state = pt.load(resume_path, map_location=device)
        model.load_state_dict(resume_state["model_state_dict"])
        if resume_state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if scheduler is not None and resume_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if scaler is not None and resume_state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
    report = {
        "args": vars(args),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "label_stats": label_stats,
        "pos_weight": pos_weight_value,
        "model": "SimplePartnerGVPESMNet",
        "scheduler": scheduler_info,
        "resume_checkpoint": args.resume_checkpoint,
    }
    (output_dir / "run_config.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows_log = read_log(output_dir / "train_step_log.csv") if args.resume_checkpoint else []
    global_step = int(resume_state.get("global_step", 0)) if resume_state is not None else 0
    start_epoch = max(int(resume_state.get("epoch", 1)) - 1, 0) if resume_state is not None else 0
    resume_skip_batches = 0
    if resume_state is not None and args.resume_skip_seen_batches and len(train_loader) > 0:
        resume_skip_batches = int(global_step % len(train_loader))
    start = time.time()
    if resume_state is not None:
        print(
            f"[resume] checkpoint={args.resume_checkpoint} epoch={start_epoch + 1} "
            f"global_step={global_step} skip_batches={resume_skip_batches}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"train_simple_e{epoch + 1}")):
            if resume_state is not None and epoch == start_epoch and batch_index < resume_skip_batches:
                continue
            if batch.get("skip_batch"):
                continue
            step_start = time.time()
            batch = move_batch_to_device(batch, device)
            use_shuffled = args.shuffled_partner_interval > 0 and (global_step + 1) % args.shuffled_partner_interval == 0
            shuffled = None
            with pt.cuda.amp.autocast(enabled=use_amp):
                out = model_forward(model, batch)
                loss_residue = masked_bce(out["logits"], batch["y"], batch["mask"], pos_weight)
                loss_pair_true = pair_bce(out["pair_logit"], pt.ones_like(out["pair_logit"]))
                loss_pair_shuffled = out["pair_logit"].sum() * 0.0
                loss_pair_margin = out["pair_logit"].sum() * 0.0
                if use_shuffled and (args.shuffled_pair_loss_weight > 0 or args.partner_margin_loss_weight > 0):
                    shuffled = attach_cached_negative_partners(
                        batch,
                        negative_partner_pool,
                        negative_partner_sequences,
                        args,
                        step=global_step + 1,
                    )
                    if shuffled is not None:
                        out_shuf = model_forward(model, shuffled)
                        loss_pair_shuffled = pair_bce(out_shuf["pair_logit"], pt.zeros_like(out_shuf["pair_logit"]))
                        loss_pair_margin = pair_margin_loss(out["pair_logit"], out_shuf["pair_logit"], margin=args.partner_margin)
                loss = (
                    loss_residue
                    + args.pair_loss_weight * loss_pair_true
                    + args.shuffled_pair_loss_weight * loss_pair_shuffled
                    + args.partner_margin_loss_weight * loss_pair_margin
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.time() - start
                row = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "loss": f"{float(loss.detach().cpu()):.8f}",
                    "loss_residue": f"{float(loss_residue.detach().cpu()):.8f}",
                    "loss_pair_true": f"{float(loss_pair_true.detach().cpu()):.8f}",
                    "loss_pair_shuffled": f"{float(loss_pair_shuffled.detach().cpu()):.8f}",
                    "loss_pair_margin": f"{float(loss_pair_margin.detach().cpu()):.8f}",
                    "used_shuffled_partner": int(shuffled is not None),
                    "step_seconds": f"{time.time() - step_start:.4f}",
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "steps_per_hour": f"{global_step / max(elapsed, 1e-6) * 3600.0:.2f}",
                    "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
                }
                rows_log.append(row)
                write_log(output_dir / "train_step_log.csv", rows_log)
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(output_dir / f"model_step_{global_step}.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args)
            if args.eval_every and global_step % args.eval_every == 0:
                metrics = run_eval(model, val_loader, device, pos_weight, prefix="val")
                rows_log[-1].update({k: f"{v:.8f}" if isinstance(v, float) and math.isfinite(v) else str(v) for k, v in metrics.items()})
                write_log(output_dir / "train_step_log.csv", rows_log)
                save_checkpoint(output_dir / f"model_step_{global_step}.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args)
            if args.max_steps and global_step >= args.max_steps:
                save_checkpoint(output_dir / "model_last.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args)
                return
        if args.save_every_epoch:
            save_checkpoint(output_dir / f"model_epoch_{epoch + 1}.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args)
    save_checkpoint(output_dir / "model_last.pt", model, optimizer, scheduler, scaler, args.epochs, global_step, args)


def main():
    parser = argparse.ArgumentParser(description="Train simple GVP+cached-ESM partner interface model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--esm-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", required=True)
    parser.add_argument("--negative-partner-pool", default="")
    parser.add_argument("--negative-partner-sequences", default="")
    parser.add_argument("--states", default="apo,af2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine", "linear"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--pos-weight-factor", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-split-group", choices=["system_id", "cluster_id", "cluster_id_RL", "receptor_uniprot", "partner_uniprot"], default="cluster_id")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gvp-layers", type=int, default=4)
    parser.add_argument("--gvp-vector-dim", type=int, default=16)
    parser.add_argument("--residue-graph-k", type=int, default=24)
    parser.add_argument("--residue-graph-cache-ks", type=lambda text: tuple(int(x) for x in str(text).split(",") if x), default=(16,24,32))
    parser.add_argument("--esm-model-name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--esm-repr-layer", type=int, default=33)
    parser.add_argument("--esm-max-length", type=int, default=1022)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-skip-seen-batches", action="store_true")
    parser.add_argument("--pair-loss-weight", type=float, default=0.01)
    parser.add_argument("--shuffled-pair-loss-weight", type=float, default=0.01)
    parser.add_argument("--partner-margin-loss-weight", type=float, default=0.01)
    parser.add_argument("--partner-margin", type=float, default=0.2)
    parser.add_argument("--shuffled-partner-interval", type=int, default=6)
    parser.add_argument("--disable-partner-updates", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--allow-residue-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pt.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
