import math
import sys
from pathlib import Path

import torch as pt


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_generic_gvp.model import GenericGVPEncoder, GlobalContextFeedback, InterfaceHead, parse_k_schedule
from apex_surface_aux_gvp.surface_cache import build_surface_proxy_cache


class SurfaceGraphEncoder(pt.nn.Module):
    def __init__(self, dim=128, geom_dim=5, lappe_dim=0, graph_k=12, dropout=0.1, layers=1):
        super().__init__()
        self.graph_k = int(graph_k)
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim + int(lappe_dim)),
            pt.nn.Linear(dim + geom_dim + int(lappe_dim), dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.layers = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(int(layers))
            ]
        )
        self.gates = pt.nn.ModuleList([pt.nn.Sequential(pt.nn.LayerNorm(dim * 2), pt.nn.Linear(dim * 2, dim), pt.nn.Sigmoid()) for _ in self.layers])
        self.norm = pt.nn.LayerNorm(dim)

    def _neighbor_context(self, h, graph_idx):
        p, dim = h.shape
        if graph_idx is None or graph_idx.numel() == 0:
            return h.new_zeros((p, dim))
        idx = graph_idx.to(device=h.device, dtype=pt.long).clamp(0, max(p - 1, 0))
        neigh = h[idx]
        return neigh.mean(dim=1)

    def forward(self, h_surf, geom, graph_idx=None, lappe=None):
        parts = [h_surf, geom.to(device=h_surf.device, dtype=h_surf.dtype)]
        if lappe is not None and lappe.numel():
            parts.append(lappe.to(device=h_surf.device, dtype=h_surf.dtype))
        x = self.input_proj(pt.cat(parts, dim=-1))
        for gate, layer in zip(self.gates, self.layers):
            ctx = self._neighbor_context(x, graph_idx)
            z = pt.cat([x, ctx], dim=-1)
            x = self.norm(x + gate(z) * layer(z))
        return x


class SurfaceAuxGVPPredictor(pt.nn.Module):
    """Surface Auxiliary Interface Field.

    E1: residue-to-surface projection + surface MLP head.
    E2: optional surface graph encoder.
    E3: optional surface-to-residue pooling into final residue logits.
    E4: optional surface LapPE feature channel when cache contains it.
    """

    def __init__(
        self,
        surface_mode="mlp",
        atom_feature_dim=123,
        dim=128,
        vector_dim=16,
        k_schedule=(16, 16, 24, 24, 32, 32),
        rbf_bins=16,
        dropout=0.1,
        aux_layer=3,
        surface_points=96,
        surface_residue_topk=8,
        surface_graph_k=12,
        surface_lappe_dim=0,
        surface_pool_scale=0.2,
    ):
        super().__init__()
        self.surface_mode = str(surface_mode)
        self.surface_points = int(surface_points)
        self.surface_residue_topk = int(surface_residue_topk)
        self.surface_graph_k = int(surface_graph_k)
        self.surface_lappe_dim = int(surface_lappe_dim)
        self.surface_pool_scale = float(surface_pool_scale)
        self.encoder = GenericGVPEncoder(
            atom_feature_dim=atom_feature_dim,
            dim=dim,
            vector_dim=vector_dim,
            k_schedule=parse_k_schedule(k_schedule),
            rbf_bins=rbf_bins,
            dropout=dropout,
            aux_layer=aux_layer,
        )
        self.global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.primary_head = InterfaceHead(dim=dim, dropout=dropout)
        self.aux_head = InterfaceHead(dim=dim, dropout=dropout)
        use_lappe = self.surface_mode == "lappe" and self.surface_lappe_dim > 0
        self.surface_encoder = SurfaceGraphEncoder(
            dim=dim,
            geom_dim=5,
            lappe_dim=self.surface_lappe_dim if use_lappe else 0,
            graph_k=surface_graph_k,
            dropout=dropout,
            layers=1 if self.surface_mode in {"graph", "consistency", "lappe"} else 0,
        )
        self.surface_head = InterfaceHead(dim=dim, dropout=dropout)

    def _surface_from_cache_or_item(self, cache, X, q, m):
        if cache is not None:
            return cache
        item = {
            "X": X.detach().float().cpu(),
            "q": q.detach().float().cpu(),
            "M": m.detach().float().cpu(),
            "y": pt.zeros((m.shape[1],), dtype=pt.float),
        }
        return build_surface_proxy_cache(
            item,
            num_surface_points=self.surface_points,
            residue_topk=self.surface_residue_topk,
            graph_k=self.surface_graph_k,
            lappe_dim=self.surface_lappe_dim,
        )

    def _project_residue_to_surface(self, h_res, cache):
        idx = cache["surface_to_residue_idx"].to(device=h_res.device, dtype=pt.long)
        weight = cache["surface_to_residue_weight"].to(device=h_res.device, dtype=h_res.dtype)
        src = h_res.squeeze(0)[idx]
        return (src * weight.unsqueeze(-1)).sum(dim=1)

    def _pool_surface_to_residue(self, surface_logits, cache, n_res):
        idx = cache["surface_to_residue_idx"].to(device=surface_logits.device, dtype=pt.long)
        weight = cache["surface_to_residue_weight"].to(device=surface_logits.device, dtype=surface_logits.dtype)
        pooled = surface_logits.new_zeros((n_res,))
        denom = surface_logits.new_zeros((n_res,))
        contrib = surface_logits.unsqueeze(-1) * weight
        pooled.scatter_add_(0, idx.reshape(-1), contrib.reshape(-1))
        denom.scatter_add_(0, idx.reshape(-1), weight.reshape(-1))
        return pooled / denom.clamp_min(1e-6)

    def forward(self, X, ids_topk, q0, M, atom_counts, residue_counts, residue_graph_caches=None, surface_caches=None):
        logits = []
        aux_logits = []
        surface_logits = []
        surface_labels = []
        surface_masks = []
        surface_pooled_logits = []
        atom_offset = 0
        residue_offset = 0
        for sample_idx in range(len(atom_counts)):
            atom_count = int(atom_counts[sample_idx].item())
            residue_count = int(residue_counts[sample_idx].item())
            x = X[atom_offset : atom_offset + atom_count]
            q = q0[atom_offset : atom_offset + atom_count]
            m = M[atom_offset : atom_offset + atom_count, residue_offset : residue_offset + residue_count]
            graph_cache = residue_graph_caches[sample_idx] if residue_graph_caches is not None else None
            h_local, h_aux = self.encoder(x, q, m, graph_cache=graph_cache)
            mask = pt.ones((1, residue_count), dtype=h_local.dtype, device=h_local.device)
            h_final = self.global_feedback(h_local, mask)
            cache = self._surface_from_cache_or_item(
                surface_caches[sample_idx] if surface_caches is not None else None,
                x,
                q,
                m,
            )
            h_surf0 = self._project_residue_to_surface(h_final, cache)
            lappe = cache.get("surface_lappe") if self.surface_mode == "lappe" else None
            h_surf = self.surface_encoder(
                h_surf0,
                cache["surface_geom"],
                graph_idx=cache.get("surface_graph_idx"),
                lappe=lappe,
            )
            surf_logit = self.surface_head(h_surf)
            pooled = self._pool_surface_to_residue(surf_logit, cache, residue_count)
            res_logit = self.primary_head(h_final).squeeze(0)
            if self.surface_mode in {"consistency", "lappe"}:
                res_logit = res_logit + self.surface_pool_scale * pooled
            logits.append(res_logit)
            aux_logits.append(self.aux_head(h_aux).squeeze(0))
            surface_logits.append(surf_logit)
            surface_labels.append(cache["surface_labels"].to(device=surf_logit.device, dtype=surf_logit.dtype))
            surface_masks.append(cache["surface_mask"].to(device=surf_logit.device, dtype=surf_logit.dtype))
            surface_pooled_logits.append(pooled)
            atom_offset += atom_count
            residue_offset += residue_count
        return {
            "logits": pt.cat(logits, dim=0),
            "aux_logits": pt.cat(aux_logits, dim=0),
            "surface_logits": pt.cat(surface_logits, dim=0),
            "surface_labels": pt.cat(surface_labels, dim=0),
            "surface_mask": pt.cat(surface_masks, dim=0),
            "surface_pooled_logits": pt.cat(surface_pooled_logits, dim=0),
        }
