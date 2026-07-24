from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4.models import build_m4_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate an M4 checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="")
    parser.add_argument("--registry", default="assets/checkpoints.json")
    parser.add_argument("--registry-name", default="")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model = build_m4_model(checkpoint_args)
    incompatible = model.load_state_dict(state, strict=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint_path)
    report = {
        "checkpoint": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": checkpoint_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensors": len(state),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
    }
    if args.registry_name:
        registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
        record = registry["checkpoints"][args.registry_name]
        report["registry_name"] = args.registry_name
        report["registry_match"] = (
            report["bytes"] == int(record["bytes"])
            and checkpoint_hash == record["sha256"]
            and report["parameters"] == int(record["parameter_count"])
            and report["state_tensors"] == int(record["state_tensor_count"])
        )
        if not report["registry_match"]:
            raise SystemExit(json.dumps(report, indent=2))
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
