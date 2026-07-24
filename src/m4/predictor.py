from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ._bootstrap import activate_validated_implementation
from .inputs import residue_records_from_pdb
from .models import build_m4_model, forward_m4_batch

activate_validated_implementation()

from apex_dataset import read_manifest, split_field  # noqa: E402
from apex_s1_surface_field.eval_s1_surface_field import build_loader  # noqa: E402
from train_apex_simple import move_batch_to_device  # noqa: E402


OUTPUT_FIELDS = (
    "case_id",
    "chain_id",
    "residue_number",
    "insertion_code",
    "amino_acid",
    "valid_mask",
    "interface_probability",
    "prior_probability",
    "surface_conditioned_probability",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _loader_args(args: argparse.Namespace, ckpt_args: dict[str, object]) -> SimpleNamespace:
    def ckpt(name: str, default: object) -> object:
        value = ckpt_args.get(name, default)
        return default if value is None else value

    return SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        allow_residue_mismatch=False,
        tensor_cache_dir=args.tensor_cache_dir,
        residue_graph_cache_dir=args.residue_graph_cache_dir,
        residue_graph_cache_ks=(16, 24, 32),
        true_feature_cache_dir="",
        require_true_features=False,
        true_feature_version="true_sasa_dssp_v1",
        s1_surface_cache_dir=args.s1_surface_cache_dir,
        require_s1_surface_cache=True,
        disable_s1_surface_on_the_fly=True,
        s1_surface_points=int(ckpt("s1_surface_points", 1024)),
        s1_surface_graph_k=int(ckpt("s1_surface_graph_k", 16)),
        s1_surface_lappe_dim=int(ckpt("s1_surface_lappe_dim", 0)),
        s1_surface_cache_version=str(ckpt("s1_surface_cache_version", "s1_surface_shrake_v1")),
        teacher_cache_dir="",
        teacher_name="",
        teacher_cache_version="teacher_prior_v1",
        require_teacher_cache=False,
        s1_static_geometry_cache_dir=args.s1_static_geometry_cache_dir,
        s1_static_geometry_cache_version=str(
            ckpt("s1_static_geometry_cache_version", "e89_dense_geometry_v1")
        ),
        require_s1_static_geometry_cache=True,
    )


def _validate_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    if not rows:
        raise ValueError("Prediction manifest is empty")
    case_ids = [row.get("apex_case_id", "") for row in rows]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Prediction manifest requires non-empty unique apex_case_id values")
    metadata: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        case_id = row["apex_case_id"]
        records = residue_records_from_pdb(row["receptor_pdb_path"])
        manifest_ids = split_field(row.get("input_residue_ids", ""), str)
        record_ids = [str(record["residue_id"]) for record in records]
        if manifest_ids != record_ids:
            raise ValueError(f"Residue order mismatch before prediction for {case_id}")
        if len(split_field(row.get("labels", ""), int)) != len(records):
            raise ValueError(
                f"Use 'm4 prepare-input' to create technical placeholder fields for {case_id}"
            )
        metadata[case_id] = records
    return metadata


def predict(args: argparse.Namespace) -> dict[str, object]:
    rows = read_manifest(args.manifest)
    residue_metadata = _validate_rows(rows)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and args")
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        raise ValueError("Checkpoint args must be a mapping")
    device = _resolve_device(args.device)
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    model = build_m4_model(ckpt_args).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint loading returned incompatible keys")
    loader = build_loader(rows, _loader_args(args, ckpt_args))
    model.eval()
    predictions: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    with torch.inference_mode():
        for batch in loader:
            if batch.get("skip_batch") or int(batch.get("skipped_count", 0) or 0):
                raise RuntimeError(
                    "Prediction dataset skipped an input: "
                    + "; ".join(str(reason) for reason in batch.get("skip_reasons", []))
                )
            batch = move_batch_to_device(batch, device)
            output = forward_m4_batch(model, batch)
            offset = 0
            counts = batch["residue_counts"].detach().cpu().tolist()
            for sample_index, raw_count in enumerate(counts):
                count = int(raw_count)
                case_id = batch["rows"][sample_index]["apex_case_id"]
                records = residue_metadata[case_id]
                if len(records) != count:
                    raise RuntimeError(
                        f"Model/PDB residue count mismatch for {case_id}: {count} != {len(records)}"
                    )
                sl = slice(offset, offset + count)
                final = torch.sigmoid(output["logits"][sl]).detach().float().cpu().tolist()
                prior = torch.sigmoid(output["prior_logits"][sl]).detach().float().cpu().tolist()
                surface = torch.sigmoid(output["surface_conditioned_logits"][sl]).detach().float().cpu().tolist()
                valid = batch["mask"][sl].detach().float().cpu().tolist()
                for index, record in enumerate(records):
                    predictions.append(
                        {
                            "case_id": case_id,
                            "chain_id": record["chain_id"],
                            "residue_number": record["residue_number"],
                            "insertion_code": record["insertion_code"],
                            "amino_acid": record["amino_acid"],
                            "valid_mask": int(valid[index] > 0.5),
                            "interface_probability": float(final[index]),
                            "prior_probability": float(prior[index]),
                            "surface_conditioned_probability": float(surface[index]),
                        }
                    )
                seen_cases.add(case_id)
                offset += count
    expected_cases = {row["apex_case_id"] for row in rows}
    if seen_cases != expected_cases:
        raise RuntimeError(f"Prediction coverage mismatch: missing={sorted(expected_cases - seen_cases)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "residue_predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "strict_load": True,
        "device": str(device),
        "cases": len(seen_cases),
        "residues": len(predictions),
        "output": str(output_path.resolve()),
        "output_schema": list(OUTPUT_FIELDS),
    }
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict label-free M4 inference and export per-residue probabilities."
    )
    parser.add_argument("--manifest", required=True, help="Manifest made by 'm4 prepare-input'.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", required=True)
    parser.add_argument("--s1-surface-cache-dir", required=True)
    parser.add_argument("--s1-static-geometry-cache-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--matmul-precision", default="high")
    args = parser.parse_args()
    print(json.dumps(predict(args), indent=2))


if __name__ == "__main__":
    main()
