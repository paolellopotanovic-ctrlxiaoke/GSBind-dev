import hashlib
import math
import sys
from pathlib import Path

import torch as pt


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_dataset import FIELD_SEP, cache_key_for_row


def surface_cache_key_for_row(row, num_surface_points=96, residue_topk=8, graph_k=12, lappe_dim=0):
    raw = "|".join(
        [
            cache_key_for_row(row),
            f"surface_proxy_v1",
            f"p={int(num_surface_points)}",
            f"rtopk={int(residue_topk)}",
            f"gk={int(graph_k)}",
            f"lappe={int(lappe_dim)}",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def surface_cache_path_for_row(cache_dir, row, num_surface_points=96, residue_topk=8, graph_k=12, lappe_dim=0):
    return Path(cache_dir) / surface_cache_key_for_row(
        row,
        num_surface_points=num_surface_points,
        residue_topk=residue_topk,
        graph_k=graph_k,
        lappe_dim=lappe_dim,
    )


def residue_centers_from_item(item):
    X = item["X"].float()
    M = item["M"].float()
    weights = M.to(dtype=X.dtype)
    denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    return weights.transpose(0, 1).matmul(X) / denom


def _farthest_point_sample(points, count):
    n = int(points.shape[0])
    count = min(int(count), n)
    if count <= 0:
        return pt.empty((0,), dtype=pt.long)
    center = points.mean(dim=0, keepdim=True)
    dist_to_center = ((points - center) ** 2).sum(dim=-1)
    first = int(pt.argmax(dist_to_center).item())
    selected = [first]
    min_dist = ((points - points[first]) ** 2).sum(dim=-1)
    for _ in range(1, count):
        idx = int(pt.argmax(min_dist).item())
        selected.append(idx)
        min_dist = pt.minimum(min_dist, ((points - points[idx]) ** 2).sum(dim=-1))
    return pt.tensor(selected, dtype=pt.long)


def _normalized_lappe_from_edges(num_nodes, edge_idx, dim):
    dim = int(dim)
    if dim <= 0 or num_nodes <= 1:
        return pt.empty((num_nodes, 0), dtype=pt.float)
    adj = pt.zeros((num_nodes, num_nodes), dtype=pt.float)
    if edge_idx.numel():
        src = pt.arange(num_nodes).unsqueeze(-1).expand_as(edge_idx).reshape(-1)
        dst = edge_idx.reshape(-1)
        valid = (dst >= 0) & (dst < num_nodes)
        adj[src[valid], dst[valid]] = 1.0
        adj[dst[valid], src[valid]] = 1.0
    deg = adj.sum(dim=-1)
    inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    lap = pt.eye(num_nodes) - inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    try:
        eigvals, eigvecs = pt.linalg.eigh(lap)
        order = pt.argsort(eigvals)
        vecs = eigvecs[:, order[1 : dim + 1]]
    except RuntimeError:
        vecs = pt.zeros((num_nodes, min(dim, max(num_nodes - 1, 0))), dtype=pt.float)
    if vecs.shape[1] < dim:
        vecs = pt.cat([vecs, pt.zeros((num_nodes, dim - vecs.shape[1]), dtype=pt.float)], dim=-1)
    return vecs[:, :dim].float()


def build_surface_proxy_cache(
    item,
    num_surface_points=96,
    residue_topk=8,
    graph_k=12,
    lappe_dim=0,
):
    centers = residue_centers_from_item(item)
    n_res = int(centers.shape[0])
    if n_res <= 0:
        raise ValueError("Cannot build surface proxy for zero residues.")
    p = min(int(num_surface_points), n_res)
    residue_topk = min(int(residue_topk), n_res)
    graph_k = min(int(graph_k), max(p - 1, 0))
    selected = _farthest_point_sample(centers, p)
    surface_pos = centers[selected].contiguous()
    global_center = centers.mean(dim=0, keepdim=True)
    normal = surface_pos - global_center
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dist_pr = pt.cdist(surface_pos, centers).clamp_min(0.0)
    residue_idx = pt.topk(dist_pr, k=residue_topk, dim=-1, largest=False).indices
    residue_dist = pt.gather(dist_pr, 1, residue_idx)
    sigma = residue_dist[:, -1:].clamp_min(1.0)
    residue_weight = pt.softmax(-(residue_dist / sigma) ** 2, dim=-1)
    if p > 1 and graph_k > 0:
        dist_pp = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
        kk = min(graph_k + 1, p)
        graph_dist, graph_idx = pt.topk(dist_pp, k=kk, dim=-1, largest=False)
        graph_idx = graph_idx[:, 1:].contiguous()
        graph_dist = graph_dist[:, 1:].contiguous()
    else:
        graph_idx = pt.empty((p, 0), dtype=pt.long)
        graph_dist = pt.empty((p, 0), dtype=pt.float)
    radial = (surface_pos - global_center).norm(dim=-1, keepdim=True)
    radial_norm = radial / radial.mean().clamp_min(1e-6)
    density = residue_dist.mean(dim=-1, keepdim=True)
    density = density / density.mean().clamp_min(1e-6)
    geom = pt.cat([normal, radial_norm, density], dim=-1).float()
    cache = {
        "surface_pos": surface_pos.float(),
        "surface_geom": geom,
        "surface_to_residue_idx": residue_idx.long(),
        "surface_to_residue_weight": residue_weight.float(),
        "surface_graph_idx": graph_idx.long(),
        "surface_graph_dist": graph_dist.float(),
        "surface_lappe": _normalized_lappe_from_edges(p, graph_idx, lappe_dim),
        "surface_labels": pt.gather(item["y"].float(), 0, residue_idx.reshape(-1)).reshape_as(residue_idx).mul(
            residue_weight
        ).sum(dim=-1),
        "surface_mask": pt.ones((p,), dtype=pt.float),
        "num_surface_points": int(p),
        "num_residues": int(n_res),
    }
    return cache


def save_surface_proxy_cache(path, cache):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pt.save({k: (v.cpu() if isinstance(v, pt.Tensor) else v) for k, v in cache.items()}, path)

