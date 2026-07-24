from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4.data.manifest import (
    M4ManifestDataset,
    cache_path_for_row,
    graph_cache_path_for_row,
    read_manifest,
    split_field,
)
from m4.data.dataset import s1_static_geometry_cache_path_for_row
from m4.models.components.surface_cache import s1_surface_cache_path_for_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly audit all four M4 cache families.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--graph-cache-dir", required=True)
    parser.add_argument("--surface-cache-dir", required=True)
    parser.add_argument("--static-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--surface-version", default="s1_surface_shrake_v1")
    parser.add_argument("--static-version", default="e89_dense_geometry_v1")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    tensor = Path(args.tensor_cache_dir)
    graph = Path(args.graph_cache_dir)
    surface = Path(args.surface_cache_dir)
    static = Path(args.static_cache_dir)
    report = {
        "rows": len(rows),
        "unique_case_ids": len({row["apex_case_id"] for row in rows}),
        "unique_paths": len({row["receptor_pdb_path"] for row in rows}),
        "missing_tensor": 0,
        "missing_graph": 0,
        "missing_surface": 0,
        "missing_static": 0,
        "overlay_mismatch": 0,
        "bad_static": 0,
        "zero_valid_rows": 0,
    }
    dataset = M4ManifestDataset(
        rows,
        tensor_cache_dir=tensor,
        require_tensor_cache=True,
        residue_graph_cache_dir=graph,
        require_residue_graph_cache=True,
        residue_graph_ks=(16, 24, 32),
    )
    for index, row in enumerate(rows):
        paths = {
            "missing_tensor": cache_path_for_row(tensor, row),
            "missing_graph": graph_cache_path_for_row(graph, row, (16, 24, 32)),
            "missing_surface": s1_surface_cache_path_for_row(
                surface, row, 1024, 16, 0, args.surface_version
            ),
            "missing_static": s1_static_geometry_cache_path_for_row(
                static, row, args.static_version
            ),
        }
        for key, path in paths.items():
            report[key] += int(not path.is_file())
        expected_y = torch.tensor(split_field(row.get("labels", ""), int), dtype=torch.float)
        expected_mask = torch.tensor(split_field(row.get("label_mask", ""), int), dtype=torch.float)
        report["zero_valid_rows"] += int(expected_mask.sum().item() <= 0)
        if paths["missing_tensor"].is_file() and paths["missing_graph"].is_file():
            item = dataset[index]
            report["overlay_mismatch"] += int(
                not torch.equal(item["y"], expected_y)
                or not torch.equal(item["mask"], expected_mask)
            )
        if paths["missing_static"].is_file():
            payload = torch.load(paths["missing_static"], map_location="cpu")
            report["bad_static"] += int(
                "surface_graph_idx_k32" not in payload
                or "surface_graph_dist_k32" not in payload
            )

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    failures = [key for key, value in report.items() if key.startswith("missing_") and value]
    failures += [key for key in ("overlay_mismatch", "bad_static", "zero_valid_rows") if report[key]]
    if report["rows"] != report["unique_case_ids"] or failures:
        raise SystemExit(f"cache audit failed: {failures}")


if __name__ == "__main__":
    main()
