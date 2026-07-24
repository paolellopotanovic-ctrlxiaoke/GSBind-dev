"""Surface-cache keys and deterministic proxy/Shrake builders."""

import hashlib
import math
from pathlib import Path

import torch as pt

from ...data.manifest import cache_key_for_row


def s1_surface_cache_key_for_row(row, num_surface_points=1024, graph_k=16, lappe_dim=0, version="s1_surface_proxy_v1"):
    raw = "|".join(
        [
            cache_key_for_row(row),
            str(version),
            f"p={int(num_surface_points)}",
            f"gk={int(graph_k)}",
            f"lappe={int(lappe_dim)}",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def s1_surface_cache_path_for_row(cache_dir, row, num_surface_points=1024, graph_k=16, lappe_dim=0, version="s1_surface_proxy_v1"):
    return Path(cache_dir) / s1_surface_cache_key_for_row(
        row,
        num_surface_points=num_surface_points,
        graph_k=graph_k,
        lappe_dim=lappe_dim,
        version=version,
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
    first = int(pt.argmax(((points - center) ** 2).sum(dim=-1)).item())
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


def _surface_graph(surface_pos, graph_k):
    p = int(surface_pos.shape[0])
    graph_k = min(int(graph_k), max(p - 1, 0))
    if p <= 1 or graph_k <= 0:
        return pt.empty((p, 0), dtype=pt.long), pt.empty((p, 0), dtype=pt.float)
    dist = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
    kk = min(graph_k + 1, p)
    graph_dist, graph_idx = pt.topk(dist, k=kk, dim=-1, largest=False)
    return graph_idx[:, 1:].contiguous(), graph_dist[:, 1:].contiguous()


def _atom_radii_from_q(q_atom):
    # Conservative element fallback for cached atom features whose exact layout
    # can differ across surface-cache generations. This is enough for a first
    # Shrake-Rupley-style molecular surface pilot and is clearly marked in cache.
    n = int(q_atom.shape[0])
    radii = pt.full((n,), 1.70, dtype=pt.float)
    if q_atom.numel() == 0:
        return radii
    # If the first common element one-hot slots exist, use them; otherwise all-C.
    # C/N/O/S/P/H approximate van der Waals radii in Angstrom.
    guesses = [
        (0, 1.70),  # C
        (1, 1.55),  # N
        (2, 1.52),  # O
        (3, 1.80),  # S
        (4, 1.80),  # P
        (5, 1.20),  # H
    ]
    for idx, radius in guesses:
        if q_atom.shape[1] > idx:
            mask = q_atom[:, idx] > 0.5
            radii = pt.where(mask, pt.full_like(radii, radius), radii)
    return radii


def build_s1_surface_shrake_cache(
    item,
    num_surface_points=1024,
    graph_k=16,
    lappe_dim=0,
    probe_radius=1.4,
    sphere_samples=64,
):
    """Build a minimal true-ish molecular surface point cache.

    This uses a Shrake-Rupley-style solvent accessible surface approximation:
    sample points on atom spheres with probe radius, keep points not buried
    inside neighboring atom spheres, then farthest-point sample to the requested
    cache size. It is not MSMS/EDTSurf, but unlike the proxy cache it is built
    from atom-level solvent accessibility rather than residue-center outward
    offsets.
    """
    X = item["X"].float()
    q = item.get("q")
    if q is None:
        q = pt.zeros((X.shape[0], 0), dtype=pt.float)
    else:
        q = q.float()
    centers = residue_centers_from_item(item)
    n_res = int(centers.shape[0])
    n_atom = int(X.shape[0])
    if n_res <= 0 or n_atom <= 0:
        raise ValueError("Cannot build S1 Shrake surface cache for empty structure.")

    sphere_samples = max(int(sphere_samples), 16)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []
    for i in range(sphere_samples):
        z = 1.0 - (2.0 * (i + 0.5) / sphere_samples)
        r = math.sqrt(max(1.0 - z * z, 0.0))
        theta = golden * i
        dirs.append([math.cos(theta) * r, math.sin(theta) * r, z])
    dirs = pt.tensor(dirs, dtype=pt.float)

    radii = _atom_radii_from_q(q) + float(probe_radius)
    cand_pos = X[:, None, :] + dirs[None, :, :] * radii[:, None, None]
    cand_normal = dirs[None, :, :].expand(n_atom, sphere_samples, 3).reshape(-1, 3)
    cand_pos = cand_pos.reshape(-1, 3)

    # Surface point is accessible if it is not inside any other atom's
    # solvent-expanded sphere. Chunking keeps memory predictable.
    accessible = pt.ones((cand_pos.shape[0],), dtype=pt.bool)
    atom_ids = pt.arange(n_atom).repeat_interleave(sphere_samples)
    chunk = 4096
    for start in range(0, cand_pos.shape[0], chunk):
        stop = min(start + chunk, cand_pos.shape[0])
        dist = pt.cdist(cand_pos[start:stop], X).clamp_min(0.0)
        inside = dist < (radii[None, :] - 1e-4)
        own = atom_ids[start:stop, None] == pt.arange(n_atom)[None, :]
        inside = inside & ~own
        accessible[start:stop] = ~inside.any(dim=1)

    surface_pos = cand_pos[accessible]
    surface_normal = cand_normal[accessible]
    if surface_pos.shape[0] == 0:
        return build_s1_surface_proxy_cache(
            item,
            num_surface_points=num_surface_points,
            graph_k=graph_k,
            lappe_dim=lappe_dim,
        )

    p = min(int(num_surface_points), int(surface_pos.shape[0]))
    selected = _farthest_point_sample(surface_pos, p)
    surface_pos = surface_pos[selected].contiguous()
    surface_normal = surface_normal[selected].contiguous()
    surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dist_to_res = pt.cdist(surface_pos, centers).clamp_min(0.0)
    nearest_dist, nearest_idx = pt.min(dist_to_res, dim=-1)
    centroid = centers.mean(dim=0, keepdim=True)
    radial = (surface_pos - centroid).norm(dim=-1, keepdim=True)
    radial_norm = radial / radial.mean().clamp_min(1e-6)
    density = pt.topk(dist_to_res, k=min(8, n_res), dim=-1, largest=False).values.mean(dim=-1, keepdim=True)
    density = density / density.mean().clamp_min(1e-6)
    nearest_dist_norm = nearest_dist.unsqueeze(-1) / nearest_dist.mean().clamp_min(1e-6)
    geom = pt.cat([surface_normal, radial_norm, density, nearest_dist_norm], dim=-1).float()

    graph_idx, graph_dist = _surface_graph(surface_pos, graph_k)
    cache = {
        "surface_pos": surface_pos.float(),
        "surface_normal": surface_normal.float(),
        "surface_geom": geom.float(),
        "surface_graph_idx": graph_idx.long(),
        "surface_graph_dist": graph_dist.float(),
        "surface_lappe": _normalized_lappe_from_edges(p, graph_idx, lappe_dim),
        "nearest_residue_idx": nearest_idx.long(),
        "nearest_residue_dist": nearest_dist.float(),
        "surface_mask": pt.ones((p,), dtype=pt.float),
        "num_surface_points": int(p),
        "num_residues": int(n_res),
        "cache_kind": "shrake_rypley_atom_surface_v0",
        "probe_radius": float(probe_radius),
        "sphere_samples": int(sphere_samples),
        "accessible_candidates": int(accessible.sum().item()),
    }
    if "y" in item:
        y = item["y"].float()
        cache["proxy_surface_labels"] = y[nearest_idx].float()
    return cache


def build_s1_surface_proxy_cache(item, num_surface_points=1024, graph_k=16, lappe_dim=0, outward_offset=3.0):
    """Build a runnable proxy surface cache from residue centers.

    This is not a true solvent-excluded surface. It is a fallback interface so
    S1 can run before MSMS/EDTSurf/FreeSASA-style true surface caches exist.
    True surface caches should provide the same keys: surface_pos,
    surface_normal, surface_geom, surface_graph_idx, surface_graph_dist.
    """
    centers = residue_centers_from_item(item)
    n_res = int(centers.shape[0])
    if n_res <= 0:
        raise ValueError("Cannot build S1 surface cache for zero residues.")
    p = min(int(num_surface_points), n_res)
    selected = _farthest_point_sample(centers, p)
    anchor = centers[selected].contiguous()
    centroid = centers.mean(dim=0, keepdim=True)
    normal = anchor - centroid
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    surface_pos = anchor + float(outward_offset) * normal

    dist_to_res = pt.cdist(surface_pos, centers).clamp_min(0.0)
    nearest_dist, nearest_idx = pt.min(dist_to_res, dim=-1)
    radial = (surface_pos - centroid).norm(dim=-1, keepdim=True)
    radial_norm = radial / radial.mean().clamp_min(1e-6)
    density = pt.topk(dist_to_res, k=min(8, n_res), dim=-1, largest=False).values.mean(dim=-1, keepdim=True)
    density = density / density.mean().clamp_min(1e-6)
    nearest_dist_norm = nearest_dist.unsqueeze(-1) / nearest_dist.mean().clamp_min(1e-6)
    geom = pt.cat([normal, radial_norm, density, nearest_dist_norm], dim=-1).float()

    graph_idx, graph_dist = _surface_graph(surface_pos, graph_k)
    cache = {
        "surface_pos": surface_pos.float(),
        "surface_normal": normal.float(),
        "surface_geom": geom.float(),
        "surface_graph_idx": graph_idx.long(),
        "surface_graph_dist": graph_dist.float(),
        "surface_lappe": _normalized_lappe_from_edges(p, graph_idx, lappe_dim),
        "nearest_residue_idx": nearest_idx.long(),
        "nearest_residue_dist": nearest_dist.float(),
        "surface_mask": pt.ones((p,), dtype=pt.float),
        "num_surface_points": int(p),
        "num_residues": int(n_res),
        "cache_kind": "proxy_residue_outward_surface",
    }
    if "y" in item:
        y = item["y"].float()
        cache["proxy_surface_labels"] = y[nearest_idx].float()
    return cache


def save_s1_surface_cache(path, cache):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pt.save({k: (v.cpu() if isinstance(v, pt.Tensor) else v) for k, v in cache.items()}, path)

