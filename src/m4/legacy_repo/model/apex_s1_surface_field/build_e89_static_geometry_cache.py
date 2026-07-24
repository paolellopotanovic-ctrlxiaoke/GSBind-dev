import argparse
import json
from pathlib import Path

import torch as pt

from apex_dataset import read_manifest
from apex_s1_surface_field.dataset import (
    S1SurfaceFieldDataset,
    s1_static_geometry_cache_path_for_row,
)
from apex_generic_gvp_augfeat.model import AugFeatureBuilder


def dense_geometry(item, atom_feature_dim):
    X = item["X"].cuda(non_blocking=False).float()
    q = item["q"].cuda(non_blocking=False).float()
    M = item["M"].cuda(non_blocking=False).float()
    counts = M.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    centers = M.transpose(0, 1).matmul(X) / counts
    atom_offset = int(atom_feature_dim) - 64
    vectors = []
    if atom_offset >= 0 and q.shape[1] >= atom_offset + 4:
        for role_idx in range(4):
            atom_weights = M * q[:, atom_offset + role_idx : atom_offset + role_idx + 1]
            role_counts = atom_weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            positions = atom_weights.transpose(0, 1).matmul(X) / role_counts
            present = (atom_weights.sum(dim=0) > 0).to(dtype=X.dtype).unsqueeze(-1)
            vector = positions - centers
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            vectors.append(vector * present)
    if not vectors:
        vectors = [X.new_zeros((M.shape[1], 3)) for _ in range(4)]
    return centers, pt.stack(vectors, dim=-2)


def surface_graph(surface_pos, graph_k):
    surface_pos = surface_pos.cuda(non_blocking=False).float()
    point_count = int(surface_pos.shape[0])
    if point_count <= 1:
        return surface_pos.new_empty((point_count, 0), dtype=pt.long), surface_pos.new_empty((point_count, 0))
    k = min(int(graph_k) + 1, point_count)
    distance = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
    graph_dist, graph_idx = pt.topk(distance, k=k, dim=-1, largest=False)
    return graph_idx[:, 1:].contiguous(), graph_dist[:, 1:].contiguous()


def dense_aug_features(item, centers, atom_feature_dim, residue_feature_dim=43):
    q = item["q"].cuda(non_blocking=False).float()
    M = item["M"].cuda(non_blocking=False).float()
    counts = M.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    q_res = M.transpose(0, 1).matmul(q) / counts
    builder = AugFeatureBuilder(atom_feature_dim=atom_feature_dim, residue_feature_dim=residue_feature_dim).cuda().eval()
    aa20, phys = builder.aa_features(q_res)
    graph_cache = {
        key: value.cuda(non_blocking=False) if isinstance(value, pt.Tensor) else value
        for key, value in item["residue_graph_cache"].items()
    }
    graph = builder.graph_features(centers, graph_cache=graph_cache)
    pseudo_ss = builder.pseudo_secondary_features(centers)
    atom_count = M.sum(dim=0).log1p().unsqueeze(-1) / 5.0
    features = pt.cat([aa20, phys, graph, pseudo_ss, atom_count], dim=-1)
    if features.shape[1] < residue_feature_dim:
        features = pt.cat([features, features.new_zeros((features.shape[0], residue_feature_dim - features.shape[1]))], dim=-1)
    return features[:, :residue_feature_dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", required=True)
    parser.add_argument("--surface-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--surface-cache-version", default="s1_surface_shrake_v1")
    parser.add_argument("--geometry-cache-version", default="e89_dense_geometry_v1")
    parser.add_argument("--atom-feature-dim", type=int, default=123)
    parser.add_argument("--surface-graph-k", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    pt.set_float32_matmul_precision("high")

    rows = read_manifest(args.manifest)
    if args.max_cases:
        rows = rows[: int(args.max_cases)]
    dataset = S1SurfaceFieldDataset(
        rows,
        strict_residue_alignment=True,
        skip_bad_examples=True,
        tensor_cache_dir=args.tensor_cache_dir,
        require_tensor_cache=True,
        esm_cache_dir=None,
        require_esm_cache=False,
        residue_graph_cache_dir=args.residue_graph_cache_dir,
        require_residue_graph_cache=True,
        residue_graph_ks=(16, 24, 32),
        surface_cache_dir="",
        build_surface_on_the_fly=False,
        s1_surface_cache_dir=args.surface_cache_dir,
        require_s1_surface_cache=True,
        build_s1_surface_on_the_fly=False,
        s1_surface_points=1024,
        s1_surface_graph_k=16,
        s1_surface_lappe_dim=0,
        s1_surface_cache_version=args.surface_cache_version,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"rows": len(rows), "written": 0, "existing": 0, "skipped": 0, "errors": []}
    for index in range(len(dataset)):
        row = rows[index]
        output_path = s1_static_geometry_cache_path_for_row(
            output_dir, row, version=args.geometry_cache_version
        )
        if output_path.is_file():
            report["existing"] += 1
            continue
        item = dataset[index]
        row = item.get("rows", row)
        if item.get("skip"):
            report["skipped"] += 1
            report["errors"].append({"case_id": row.get("apex_case_id", ""), "reason": item.get("skip_reason", "")})
            continue
        with pt.no_grad():
            centers, backbone_vectors = dense_geometry(item, args.atom_feature_dim)
            aug_features = dense_aug_features(item, centers, args.atom_feature_dim)
            graph_idx, graph_dist = surface_graph(
                item["s1_surface_cache"]["surface_pos"], args.surface_graph_k
            )
        payload = {
            "centers": centers.cpu(),
            "backbone_vectors": backbone_vectors.cpu(),
            "aug_features": aug_features.cpu(),
            f"surface_graph_idx_k{args.surface_graph_k}": graph_idx.cpu(),
            f"surface_graph_dist_k{args.surface_graph_k}": graph_dist.cpu(),
            "num_residues": int(centers.shape[0]),
            "num_surface_points": int(graph_idx.shape[0]),
            "geometry_cache_version": args.geometry_cache_version,
            "source_case_id": row.get("apex_case_id", ""),
            "source_receptor_pdb_path": row.get("receptor_pdb_path", ""),
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        pt.save(payload, temporary)
        temporary.replace(output_path)
        report["written"] += 1
        if report["written"] % 100 == 0:
            print(json.dumps({key: value for key, value in report.items() if key != "errors"}), flush=True)
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
