import argparse
import json
import sys
from pathlib import Path

import torch as pt

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_dataset import ApexPairedManifestDataset, read_manifest
from apex_s1_surface_field.surface_cache import (
    build_s1_surface_proxy_cache,
    build_s1_surface_shrake_cache,
    save_s1_surface_cache,
    s1_surface_cache_path_for_row,
)


def main():
    parser = argparse.ArgumentParser(description="Build S1 surface proxy cache for APEX manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", default="")
    parser.add_argument("--states", default="apo,af2")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--surface-points", type=int, default=1024)
    parser.add_argument("--surface-graph-k", type=int, default=16)
    parser.add_argument("--surface-lappe-dim", type=int, default=0)
    parser.add_argument("--surface-cache-version", default="s1_surface_proxy_v1")
    parser.add_argument("--surface-kind", choices=["proxy", "shrake"], default="proxy")
    parser.add_argument("--shrake-probe-radius", type=float, default=1.4)
    parser.add_argument("--shrake-sphere-samples", type=int, default=64)
    parser.add_argument("--residue-graph-cache-ks", type=lambda text: tuple(int(x) for x in str(text).split(",") if x), default=(16, 24, 32))
    parser.add_argument("--allow-residue-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    states = {x.strip() for x in args.states.split(",") if x.strip()}
    rows = [row for row in rows if row.get("input_state") in states]
    if args.max_cases:
        rows = rows[: args.max_cases]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ApexPairedManifestDataset(
        rows,
        strict_residue_alignment=not args.allow_residue_mismatch,
        skip_bad_examples=True,
        tensor_cache_dir=args.tensor_cache_dir,
        require_tensor_cache=True,
        residue_graph_cache_dir=args.residue_graph_cache_dir or None,
        require_residue_graph_cache=False,
        residue_graph_ks=args.residue_graph_cache_ks,
    )
    ok = 0
    skipped = 0
    errors = []
    for i in tqdm(range(len(dataset)), desc="build_s1_surface_cache"):
        row = rows[i]
        outpath = s1_surface_cache_path_for_row(
            output_dir,
            row,
            num_surface_points=args.surface_points,
            graph_k=args.surface_graph_k,
            lappe_dim=args.surface_lappe_dim,
            version=args.surface_cache_version,
        )
        if outpath.is_file() and not args.overwrite:
            ok += 1
            continue
        try:
            item = dataset[i]
            if item.get("skip"):
                skipped += 1
                errors.append({"row": row.get("apex_case_id", str(i)), "error": item.get("skip_reason", "")})
                continue
            if args.surface_kind == "shrake":
                cache = build_s1_surface_shrake_cache(
                    item,
                    num_surface_points=args.surface_points,
                    graph_k=args.surface_graph_k,
                    lappe_dim=args.surface_lappe_dim,
                    probe_radius=args.shrake_probe_radius,
                    sphere_samples=args.shrake_sphere_samples,
                )
            else:
                cache = build_s1_surface_proxy_cache(
                    item,
                    num_surface_points=args.surface_points,
                    graph_k=args.surface_graph_k,
                    lappe_dim=args.surface_lappe_dim,
                )
            save_s1_surface_cache(outpath, cache)
            ok += 1
        except Exception as exc:
            skipped += 1
            errors.append({"row": row.get("apex_case_id", str(i)), "error": repr(exc)})
    report = {
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "ok": ok,
        "skipped": skipped,
        "surface_points": args.surface_points,
        "surface_graph_k": args.surface_graph_k,
        "surface_lappe_dim": args.surface_lappe_dim,
        "cache_version": args.surface_cache_version,
        "surface_kind": args.surface_kind,
        "shrake_probe_radius": args.shrake_probe_radius,
        "shrake_sphere_samples": args.shrake_sphere_samples,
        "errors": errors[:200],
    }
    (output_dir / "build_s1_surface_cache_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
