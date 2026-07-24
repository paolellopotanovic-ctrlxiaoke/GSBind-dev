#!/usr/bin/env python3
"""Build a deterministic, receptor-only Pinder-train holo candidate table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import stable_digest, structure_key, write_csv, write_json


def build_candidates(
    index_path: str | Path,
    cluster_cap: int,
    target: int,
    pinder_root: str | Path | None = None,
    require_available: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    columns = [
        "split", "id", "cluster_id_R", "cluster_id_L", "uniprot_R",
        "uniprot_L", "holo_R_pdb", "holo_L_pdb",
    ]
    frame = pd.read_parquet(index_path, columns=columns)
    train = frame.loc[frame["split"].eq("train")]
    parts = []
    for side in ("R", "L"):
        part = train[["id", f"cluster_id_{side}", f"uniprot_{side}", f"holo_{side}_pdb"]].copy()
        part.columns = ["system_id", "cluster_id", "uniprot", "holo_pdb"]
        part["receptor_side"] = side
        parts.append(part)
    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates[candidates["holo_pdb"].notna() & candidates["holo_pdb"].ne("")]
    candidates = candidates[~candidates["cluster_id"].isin(["cluster_-1", "cluster_p", ""])]
    candidates = candidates[candidates["uniprot"].notna() & ~candidates["uniprot"].isin(["", "UNDEFINED"])]
    candidates["structure_key"] = candidates["holo_pdb"].map(structure_key)
    candidates["stable_hash"] = candidates.apply(
        lambda row: stable_digest(row["structure_key"], row["system_id"], row["receptor_side"]), axis=1
    )
    candidates = candidates.sort_values(
        ["structure_key", "stable_hash"], kind="mergesort"
    ).drop_duplicates("structure_key", keep="first")

    if require_available:
        if not pinder_root:
            raise ValueError("pinder_root is required when require_available=True")
        root = Path(pinder_root)
        available = {
            path.name.lower()
            for directory in (root / "pdbs", root / "test_set_pdbs", root)
            if directory.is_dir()
            for path in directory.iterdir()
            if path.is_file()
        }
        candidates = candidates[candidates["structure_key"].isin(available)]

    candidates = candidates.sort_values(["cluster_id", "stable_hash"], kind="mergesort")
    capped = candidates.groupby("cluster_id", sort=True, group_keys=False).head(cluster_cap)
    selected = capped.sort_values(["stable_hash", "structure_key"], kind="mergesort").head(target)
    output_rows = selected.drop(columns=["stable_hash"]).to_dict("records")
    summary = {
        "index": str(Path(index_path)),
        "source_split": "train",
        "index_rows": int(len(frame)),
        "train_system_rows": int(len(train)),
        "unique_candidate_structures": int(len(candidates)),
        "candidate_clusters": int(candidates["cluster_id"].nunique()),
        "cluster_cap": cluster_cap,
        "capped_pool_rows": int(len(capped)),
        "selected_rows": int(len(selected)),
        "selected_clusters": int(selected["cluster_id"].nunique()),
        "selected_unique_structures": int(selected["structure_key"].nunique()),
        "target": target,
        "target_reached": bool(len(selected) >= target),
    }
    return output_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--cluster-cap", type=int, default=20)
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--pinder-root")
    parser.add_argument("--require-available", action="store_true")
    args = parser.parse_args()
    rows, summary = build_candidates(
        args.index, args.cluster_cap, args.target, args.pinder_root, args.require_available
    )
    write_csv(args.output, rows)
    write_json(args.summary, summary)
    print(summary)
    if not summary["target_reached"]:
        raise SystemExit("Target was not reached; no selection constraint was relaxed")


if __name__ == "__main__":
    main()
