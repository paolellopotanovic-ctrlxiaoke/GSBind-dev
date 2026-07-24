from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CHECKPOINT_NAME = "m4_holo20k_v9_clean_epoch4"
CHECKPOINT_FILENAME = "m4_holo20k_v9_clean_epoch4.pt"


def build_command_plan(
    root: Path,
    work_dir: Path,
    pdb_path: Path,
    checkpoint: Path,
    download_checkpoint: bool,
    case_id: str = "1CRN",
    input_state: str = "apo",
) -> list[list[str]]:
    python = sys.executable
    manifest = work_dir / "manifest.csv"
    tensor = work_dir / "tensor"
    graph = work_dir / "graph"
    surface = work_dir / "surface"
    static = work_dir / "static"
    predictions = work_dir / "predictions"

    commands: list[list[str]] = []
    if download_checkpoint:
        commands.append(
            [
                python,
                str(root / "scripts" / "download_checkpoint.py"),
                CHECKPOINT_NAME,
                "--output-dir",
                str(checkpoint.parent),
            ]
        )
    commands.extend(
        [
            [
                python,
                str(root / "scripts" / "validate_checkpoint.py"),
                str(checkpoint),
                "--registry-name",
                CHECKPOINT_NAME,
            ],
            [
                python,
                str(root / "main.py"),
                "prepare-input",
                "--pdb",
                str(pdb_path),
                "--case-id",
                case_id,
                "--input-state",
                input_state,
                "--output",
                str(manifest),
            ],
            [
                python,
                str(root / "main.py"),
                "build-tensor-cache",
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(tensor),
                "--workers",
                "1",
            ],
            [
                python,
                str(root / "main.py"),
                "build-graph-cache",
                "--manifest",
                str(manifest),
                "--tensor-cache-dir",
                str(tensor),
                "--graph-cache-dir",
                str(graph),
                "--ks",
                "16,24,32",
                "--workers",
                "1",
            ],
            [
                python,
                str(root / "main.py"),
                "build-surface-cache",
                "--manifest",
                str(manifest),
                "--tensor-cache-dir",
                str(tensor),
                "--residue-graph-cache-dir",
                str(graph),
                "--output-dir",
                str(surface),
                "--states",
                input_state,
                "--surface-kind",
                "shrake",
                "--surface-points",
                "1024",
                "--surface-graph-k",
                "16",
                "--surface-cache-version",
                "s1_surface_shrake_v1",
            ],
            [
                python,
                str(root / "main.py"),
                "build-static-cache",
                "--manifest",
                str(manifest),
                "--tensor-cache-dir",
                str(tensor),
                "--residue-graph-cache-dir",
                str(graph),
                "--surface-cache-dir",
                str(surface),
                "--output-dir",
                str(static),
                "--surface-cache-version",
                "s1_surface_shrake_v1",
                "--geometry-cache-version",
                "e89_dense_geometry_v1",
                "--surface-graph-k",
                "32",
            ],
            [
                python,
                str(root / "inference.py"),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(predictions),
                "--tensor-cache-dir",
                str(tensor),
                "--residue-graph-cache-dir",
                str(graph),
                "--s1-surface-cache-dir",
                str(surface),
                "--s1-static-geometry-cache-dir",
                str(static),
            ],
        ]
    )
    return commands


def require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Create the environment.yml environment first.") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for the static cache and M4 forward pass. "
            "Use --dry-run to inspect the command chain on a CPU-only machine."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download M4, build all four caches, and run one label-free 1CRN forward pass."
    )
    parser.add_argument("--work-dir", default="work/1CRN")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pdb", default="examples/inference/1CRN.pdb")
    parser.add_argument("--case-id", default="1CRN")
    parser.add_argument("--input-state", choices=("apo", "holo", "af2"), default="apo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    work_path = Path(args.work_dir).expanduser()
    work_dir = work_path.resolve() if work_path.is_absolute() else (root / work_path).resolve()
    pdb_input = Path(args.pdb).expanduser()
    pdb_path = pdb_input.resolve() if pdb_input.is_absolute() else (root / pdb_input).resolve()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file() and not args.dry_run:
            raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
        download = False
    else:
        checkpoint = root / "checkpoints" / CHECKPOINT_FILENAME
        download = not checkpoint.is_file()

    if not pdb_path.is_file() and not args.dry_run:
        raise SystemExit(f"PDB does not exist: {pdb_path}")

    commands = build_command_plan(
        root,
        work_dir,
        pdb_path,
        checkpoint,
        download,
        case_id=args.case_id,
        input_state=args.input_state,
    )
    if not args.dry_run:
        require_cuda()
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)

    if args.dry_run:
        print("Dry run complete; no files were created.")
    else:
        print(f"Predictions: {(work_dir / 'predictions' / 'residue_predictions.csv').resolve()}")


if __name__ == "__main__":
    main()
