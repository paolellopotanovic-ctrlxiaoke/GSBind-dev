import math
import sys
import inspect
from pathlib import Path

import torch as pt
from torch.utils.checkpoint import checkpoint as torch_checkpoint


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_model import FullGVPMessageLayer
from apex_multigraph_plus.model import StaticMultiViewGVPEncoder, StaticResidueMultiplexGVPEncoder, GlobalContextFeedback, InterfaceHead
from apex_generic_gvp_augfeat.model import AugFeatureBuilder
from apex_s1_surface_field.surface_cache import build_s1_surface_proxy_cache
from apex_s1_surface_field.semantic_patch import SemanticConnectedPatchHead
from apex_teacher_prior.modules import TeacherHiddenAdapter


def _parse_positive_ints(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace(";", ",").split(",")
    out = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        k = int(text)
        if k > 0 and k not in out:
            out.append(k)
    return tuple(out)


class SerialK16K32MultiViewGVPEncoder(StaticMultiViewGVPEncoder):
    """Static multiview encoder with selected graph views applied as one serial view."""

    def __init__(self, *args, serial_pair=("k16", "k32"), serial_views=None, **kwargs):
        graph_views = tuple(str(x).strip() for x in kwargs.get("graph_views", ()) if str(x).strip())
        serial_views = tuple(serial_views or serial_pair)
        if len(serial_views) < 2:
            raise ValueError("serial graph flow requires at least two graph view names")
        if not all(name in graph_views for name in serial_views):
            raise ValueError(f"Serial graph flow requires all serial views in graph_views: {serial_views}")
        parent_params = inspect.signature(StaticMultiViewGVPEncoder.__init__).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in parent_params}
        super().__init__(*args, **kwargs)
        self.serial_pair = serial_views[:2]
        self.serial_views = serial_views
        self.serial_view_name = "_then_".join(serial_views)
        serial_set = set(serial_views)
        self.effective_graph_views = tuple(
            [name for name in self.graph_views if name not in serial_set] + [self.serial_view_name]
        )
        self.serial_view_order = []
        for name in self.graph_views:
            if name == serial_views[0]:
                self.serial_view_order.append(self.serial_view_name)
            elif name not in serial_set:
                self.serial_view_order.append(name)
        self.serial_view_order = tuple(self.serial_view_order)
        if self.router_mode == "global":
            self.view_logits = pt.nn.Parameter(pt.zeros(len(self.serial_view_order)))
        elif self.router_mode == "per_residue":
            dim = self.fusion_norm.normalized_shape[0]
            self.view_router = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * len(self.serial_view_order)),
                pt.nn.Linear(dim * len(self.serial_view_order), dim),
                pt.nn.GELU(),
                pt.nn.Dropout(self.views[self.graph_views[0]].dropout.p if self.graph_views else 0.0),
                pt.nn.Linear(dim, len(self.serial_view_order)),
            )

    def _run_view_state(self, name, s, v, values, centers=None, frame=None):
        view = self.views[name]
        edge_s = values["edge_s"]
        edge_v = values["edge_v"]
        nn_idx = values["nn_idx"]
        rel_type = values["rel_type"]
        if nn_idx.shape[1] > 0:
            for layer_idx, layer in enumerate(view.layers):
                edge_s, _update_norm = view._relation_edge_update(s, edge_s, nn_idx, rel_type, layer_idx)
                if view.edge_pair_mode != "none":
                    edge_s, _pair_norm, _pair_gate = view.pair_updates[layer_idx](edge_s, s, nn_idx)
                if view.oagnn_mode == "official":
                    s, v = layer(s, v, edge_s, edge_v, nn_idx, frame=frame)
                elif view.oagnn_mode == "oa_edge":
                    s, v = layer(s, v, edge_s, edge_v, nn_idx, centers=centers, rel_type=rel_type)
                else:
                    s, v = layer(s, v, edge_s, edge_v, nn_idx)
        return s, v

    def forward(self, X, q_atom, M=None, graph_cache=None, extra_residue_features=None, true_feature_cache=None, chain_ids=None, atom_to_residue_index=None, num_residues=None, static_geometry=None):
        s0, v0, centers = self.feature_builder.node_features(
            X, q_atom, M, extra_residue_features=extra_residue_features,
            atom_to_residue_index=atom_to_residue_index, num_residues=num_residues,
            static_geometry=static_geometry,
        )
        frame = None
        if self.oagnn_mode == "official":
            frame, _pos_ca, frame_valid, _centers_from_ca = self.feature_builder.backbone_frames(
                X, M, q_atom, atom_to_residue_index=atom_to_residue_index, num_residues=num_residues
            )
            frame = frame.to(device=s0.device, dtype=s0.dtype)
        self.last_residue_vectors = v0
        diag_values = {}
        for key, value in getattr(self.feature_builder, "last_diagnostics", {}).items():
            diag_values.setdefault(key, []).append(value.detach())
        if self.oagnn_mode == "official":
            diag_values.setdefault("official_oagnn_frame_valid_mean", []).append(frame_valid.to(device=s0.device, dtype=s0.dtype).mean().detach())

        view_inputs = {}
        for name in self.graph_views:
            nn_idx, nn_dist = self._view_edges(name, centers, graph_cache, true_feature_cache=true_feature_cache, chain_ids=chain_ids)
            edge_s, edge_v = self.feature_builder.edge_features(centers, nn_idx, nn_dist)
            rel_type = self._edge_relation_types(name, centers, nn_idx, nn_dist)
            view_inputs[name] = {
                "nn_idx": nn_idx,
                "edge_s": edge_s,
                "edge_v": edge_v,
                "rel_type": rel_type,
            }
        if self.cross_view_edge_mode != "none":
            raise ValueError("k16_then_k32 serial flow currently requires cross_view_edge_mode=none.")

        outputs_by_name = {}
        final_v = v0
        first, second = self.serial_pair
        for name, values in view_inputs.items():
            if name in self.serial_views:
                continue
            outputs_by_name[name] = self.views[name](
                s0,
                v0,
                values["nn_idx"],
                values["edge_s"],
                values["edge_v"],
                rel_type=values["rel_type"],
                centers=centers,
                frame=frame,
            )
            for key, value in getattr(self.views[name], "last_diagnostics", {}).items():
                diag_values.setdefault(key, []).append(value.detach())

        s_final, v_final = s0, v0
        for name in self.serial_views:
            s_final, v_final = self._run_view_state(name, s_final, v_final, view_inputs[name], centers=centers, frame=frame)
        outputs_by_name[self.serial_view_name] = self.views[self.serial_views[-1]].local_out(
            pt.cat([s_final, pt.linalg.norm(v_final, dim=-1)], dim=-1)
        )
        final_v = v_final

        view_outputs = [outputs_by_name[name] for name in self.serial_view_order]
        stacked = pt.stack(view_outputs, dim=0)
        if self.router_mode == "global":
            weights = pt.softmax(self.view_logits, dim=0).to(dtype=stacked.dtype, device=stacked.device)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            router_entropy = -(weights * weights.clamp_min(self.router_entropy_eps).log()).sum()
            router_map = weights.unsqueeze(0).expand(stacked.shape[1], -1)
        elif self.router_mode == "per_residue":
            weights = None
            per_residue = pt.cat(view_outputs, dim=-1)
            router_map = pt.softmax(self.view_router(per_residue), dim=-1)
            fused = (router_map.unsqueeze(-1) * stacked.permute(1, 0, 2)).sum(dim=1)
            router_entropy = -(router_map * router_map.clamp_min(self.router_entropy_eps).log()).sum(dim=-1).mean()
        else:
            weights = None
            view_tokens = stacked.permute(1, 0, 2).contiguous()
            query = self.residue_query(s0).unsqueeze(1)
            router_map = None
            fused = None
            for attn, update, norm in zip(self.view_attn_rounds, self.residue_updates, self.attn_round_norms):
                attn_out, attn_weights = attn(query, view_tokens, view_tokens, need_weights=True, average_attn_weights=True)
                query = norm(query + update(pt.cat([query, attn_out], dim=-1)))
                fused = query.squeeze(1)
                router_map = attn_weights.squeeze(1)
            router_entropy = -(router_map * router_map.clamp_min(self.router_entropy_eps).log()).sum(dim=-1).mean()
        self.last_residue_vectors = final_v
        self.last_diagnostics = {key: pt.stack(values).mean() for key, values in diag_values.items() if values}
        self.last_diagnostics["serial_graph_flow_enabled"] = stacked.new_tensor(1.0).detach()
        if self.serial_views == ("k16", "k32"):
            self.last_diagnostics["serial_k16_k32_enabled"] = stacked.new_tensor(1.0).detach()
        return self.fusion_norm(fused).unsqueeze(0), stacked, weights, router_map, router_entropy


class AtomPreResidueEncoder(pt.nn.Module):
    """Atom-level message passing before atom-to-residue pooling.

    This module uses the existing tensor cache tensors directly: atom
    coordinates X, atom features q_atom, and atom-to-residue assignment M. It
    produces one residue feature per residue, with separate backbone and side
    chain pooling so side-chain chemistry is not washed out by whole-residue
    averaging.
    """

    def __init__(
        self,
        atom_feature_dim,
        dim=128,
        atom_dim=64,
        layers=2,
        topk=16,
        cutoff=5.0,
        dropout=0.05,
    ):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.dim = int(dim)
        self.atom_dim = int(atom_dim)
        self.layers = max(1, int(layers))
        self.topk = max(1, int(topk))
        self.cutoff = float(cutoff)
        centers = pt.linspace(0.0, max(self.cutoff, 1.0), 12)
        self.register_buffer("rbf_centers", centers)
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(self.atom_feature_dim + 3),
            pt.nn.Linear(self.atom_feature_dim + 3, self.atom_dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(self.atom_dim, self.atom_dim),
        )
        edge_dim = 12 + 4
        msg_in = self.atom_dim * 2 + edge_dim
        self.msg_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(msg_in),
                    pt.nn.Linear(msg_in, self.atom_dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(self.atom_dim, self.atom_dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.attn_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(msg_in),
                    pt.nn.Linear(msg_in, max(16, self.atom_dim // 2)),
                    pt.nn.GELU(),
                    pt.nn.Linear(max(16, self.atom_dim // 2), 1),
                )
                for _ in range(self.layers)
            ]
        )
        self.update_norms = pt.nn.ModuleList([pt.nn.LayerNorm(self.atom_dim) for _ in range(self.layers)])
        self.bb_empty = pt.nn.Parameter(pt.zeros(self.atom_dim))
        self.sc_empty = pt.nn.Parameter(pt.zeros(self.atom_dim))
        self.res_attn = pt.nn.Sequential(
            pt.nn.LayerNorm(self.atom_dim),
            pt.nn.Linear(self.atom_dim, max(16, self.atom_dim // 2)),
            pt.nn.GELU(),
            pt.nn.Linear(max(16, self.atom_dim // 2), 1),
        )
        self.out_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(self.atom_dim * 6 + 4),
            pt.nn.Linear(self.atom_dim * 6 + 4, self.dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(self.dim, self.dim),
        )
        self.out_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(self.dim),
            pt.nn.Linear(self.dim, self.dim),
            pt.nn.Sigmoid(),
        )
        self.last_diagnostics = {}

    def _rbf(self, dist):
        width = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        return pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(dist.device, dtype=dist.dtype)) / width) ** 2)

    def _atom_edges(self, X):
        n = int(X.shape[0])
        if n <= 1:
            return pt.empty((n, 0), dtype=pt.long, device=X.device), X.new_empty((n, 0))
        dist = pt.cdist(X, X).clamp_min(0.0)
        dist = dist + pt.eye(n, device=X.device, dtype=X.dtype) * 1.0e6
        k = min(self.topk, n - 1)
        nn_dist, nn_idx = pt.topk(dist, k=k, dim=-1, largest=False)
        if self.cutoff > 0:
            fallback_dist = nn_dist[:, :1].expand_as(nn_dist)
            fallback_idx = nn_idx[:, :1].expand_as(nn_idx)
            keep = nn_dist <= self.cutoff
            nn_dist = pt.where(keep, nn_dist, fallback_dist)
            nn_idx = pt.where(keep, nn_idx, fallback_idx)
        return nn_idx.contiguous(), nn_dist.contiguous()

    def _pool_role(self, h_atom, M, role_mask, empty_token):
        """Pool atom states into residues without materializing atom x residue x dim tensors."""
        device = h_atom.device
        dtype = h_atom.dtype
        n_atom = int(h_atom.shape[0])
        n_res = int(M.shape[1])
        if n_atom <= 0 or n_res <= 0:
            empty = h_atom.new_zeros((n_res, h_atom.shape[-1]))
            present = h_atom.new_zeros((n_res,))
            return empty, empty, empty, present

        M_dev = M.to(device=device)
        role = role_mask.to(device=device, dtype=dtype) > 0
        assign_strength, res_idx = M_dev.max(dim=1)
        assigned = (assign_strength > 0) & role
        res_idx_valid = res_idx[assigned]
        h_valid = h_atom[assigned]

        present_count = h_atom.new_zeros((n_res,))
        if res_idx_valid.numel() > 0:
            present_count.index_add_(0, res_idx_valid, h_atom.new_ones((int(res_idx_valid.numel()),)))
        present = present_count > 0
        denom = present_count.clamp_min(1.0).unsqueeze(-1)

        sum_pool = h_atom.new_zeros((n_res, h_atom.shape[-1]))
        if res_idx_valid.numel() > 0:
            sum_pool.index_add_(0, res_idx_valid, h_valid)
        mean = sum_pool / denom
        empty_row = empty_token.to(device=device, dtype=dtype).view(1, -1)
        mean = pt.where(present.unsqueeze(-1), mean, empty_row)

        max_pool = h_atom.new_full((n_res, h_atom.shape[-1]), -1.0e4)
        if res_idx_valid.numel() > 0:
            if hasattr(max_pool, "scatter_reduce_"):
                max_pool.scatter_reduce_(
                    0,
                    res_idx_valid.view(-1, 1).expand(-1, h_atom.shape[-1]),
                    h_valid,
                    reduce="amax",
                    include_self=True,
                )
            else:
                for rid in res_idx_valid.unique():
                    mask = res_idx_valid == rid
                    max_pool[rid] = h_valid[mask].max(dim=0).values
        max_pool = pt.where(present.unsqueeze(-1), max_pool, empty_row)

        logits_all = self.res_attn(h_atom).squeeze(-1)
        logits_valid = logits_all[assigned].to(dtype=dtype)
        max_logit = h_atom.new_full((n_res,), -1.0e4)
        if res_idx_valid.numel() > 0:
            if hasattr(max_logit, "scatter_reduce_"):
                max_logit.scatter_reduce_(0, res_idx_valid, logits_valid, reduce="amax", include_self=True)
            else:
                for rid in res_idx_valid.unique():
                    mask = res_idx_valid == rid
                    max_logit[rid] = logits_valid[mask].max()
        exp_valid = pt.exp(logits_valid - max_logit[res_idx_valid]) if res_idx_valid.numel() > 0 else h_atom.new_empty((0,))
        exp_sum = h_atom.new_zeros((n_res,))
        if res_idx_valid.numel() > 0:
            exp_sum.index_add_(0, res_idx_valid, exp_valid)
        weighted = h_atom.new_zeros((n_res, h_atom.shape[-1]))
        if res_idx_valid.numel() > 0:
            weighted.index_add_(0, res_idx_valid, h_valid * exp_valid.unsqueeze(-1))
        attn_pool = weighted / exp_sum.clamp_min(1e-8).unsqueeze(-1)
        attn_pool = pt.where(present.unsqueeze(-1), attn_pool, empty_row)
        return mean, max_pool, attn_pool, present.to(dtype=dtype)

    def forward(self, X, q_atom, M):
        n_atom = int(X.shape[0])
        n_res = int(M.shape[1])
        if n_atom <= 0 or n_res <= 0:
            self.last_diagnostics = {}
            return X.new_zeros((n_res, self.dim))
        xyz = X.to(dtype=q_atom.dtype)
        h = self.input_proj(pt.cat([q_atom, xyz / 20.0], dim=-1))
        nn_idx, nn_dist = self._atom_edges(xyz)
        if nn_idx.shape[1] > 0:
            direction = xyz[nn_idx] - xyz.unsqueeze(1)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            edge = pt.cat([self._rbf(nn_dist.to(dtype=h.dtype)), direction.to(dtype=h.dtype), (nn_dist <= self.cutoff).to(dtype=h.dtype).unsqueeze(-1)], dim=-1)
            for msg_mlp, attn_mlp, norm in zip(self.msg_mlps, self.attn_mlps, self.update_norms):
                h_i = h.unsqueeze(1).expand(-1, nn_idx.shape[1], -1)
                h_j = h[nn_idx]
                msg_in = pt.cat([h_i, h_j, edge], dim=-1)
                msg = msg_mlp(msg_in)
                attn = pt.softmax(attn_mlp(msg_in).squeeze(-1), dim=-1)
                update = (attn.unsqueeze(-1) * msg).sum(dim=1)
                h = norm(h + update)
        atom_offset = self.atom_feature_dim - 64
        if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
            bb_mask = (q_atom[:, atom_offset : atom_offset + 4].sum(dim=-1) > 0).to(dtype=h.dtype)
        else:
            bb_mask = h.new_zeros((n_atom,))
        sc_mask = (1.0 - bb_mask).clamp_min(0.0)
        bb_mean, bb_max, bb_attn, bb_present = self._pool_role(h, M, bb_mask, self.bb_empty)
        sc_mean, sc_max, sc_attn, sc_present = self._pool_role(h, M, sc_mask, self.sc_empty)
        counts = M.to(device=h.device, dtype=h.dtype).sum(dim=0)
        bb_counts = (M.to(device=h.device, dtype=h.dtype) * bb_mask.unsqueeze(-1)).sum(dim=0)
        sc_counts = (M.to(device=h.device, dtype=h.dtype) * sc_mask.unsqueeze(-1)).sum(dim=0)
        count_feat = pt.stack(
            [
                counts.log1p(),
                bb_counts.log1p(),
                sc_counts.log1p(),
                (sc_counts / counts.clamp_min(1.0)),
            ],
            dim=-1,
        )
        pooled = pt.cat([bb_mean, bb_max, bb_attn, sc_mean, sc_max, sc_attn, count_feat], dim=-1)
        out_raw = self.out_proj(pooled)
        gate = self.out_gate(out_raw)
        out = gate * out_raw
        self.last_diagnostics = {
            "atom_pre_encoder_enabled": out.new_tensor(1.0).detach(),
            "atom_pre_edge_dist_mean": nn_dist.detach().mean() if nn_dist.numel() else out.new_tensor(0.0).detach(),
            "atom_pre_update_norm": out.detach().norm(dim=-1).mean(),
            "atom_pre_gate_mean": gate.detach().mean(),
            "atom_pre_bb_present_frac": bb_present.detach().float().mean(),
            "atom_pre_sc_present_frac": sc_present.detach().float().mean(),
        }
        return out


class GearStyleResidueGraphRefiner(pt.nn.Module):
    """Relation-aware residue message passing before the Gaussian surface readout."""

    def __init__(self, dim=128, relation_k=32, layers=2, dropout=0.1, mode="rel_edge"):
        super().__init__()
        self.dim = int(dim)
        self.relation_k = int(relation_k)
        self.layers = int(layers)
        self.mode = str(mode)
        self.use_edge_message = self.mode == "edge_msg"
        self.num_relations = 7
        centers = pt.linspace(0.0, 30.0, 12)
        self.register_buffer("rbf_centers", centers)
        self.relation_embedding = pt.nn.Embedding(self.num_relations, dim)
        edge_dim = 0
        if self.use_edge_message:
            edge_dim = 12 + 3 + 2
        msg_in = dim * 3 + edge_dim
        self.message_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(msg_in),
                    pt.nn.Linear(msg_in, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.attn_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(msg_in),
                    pt.nn.Linear(msg_in, dim // 2),
                    pt.nn.GELU(),
                    pt.nn.Linear(dim // 2, 1),
                )
                for _ in range(self.layers)
            ]
        )
        self.update_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(self.layers)])
        self.dropout = pt.nn.Dropout(dropout)
        self.router_entropy_eps = 1e-8
        self.last_diagnostics = {}

    def _cached_knn(self, centers, graph_cache):
        n = int(centers.shape[0])
        if n <= 1 or self.relation_k <= 0:
            return centers.new_empty((n, 0), dtype=pt.long), centers.new_empty((n, 0))
        if graph_cache:
            idx = graph_cache.get(f"knn_idx_k{self.relation_k}")
            dist = graph_cache.get(f"knn_dist_k{self.relation_k}")
            if idx is not None and dist is not None:
                idx = idx.to(device=centers.device, dtype=pt.long)
                dist = dist.to(device=centers.device, dtype=centers.dtype)
                return idx[:, : min(idx.shape[1], self.relation_k)], dist[:, : min(dist.shape[1], self.relation_k)]
        dist_mat = pt.cdist(centers, centers).clamp_min(0.0)
        dist_mat.fill_diagonal_(float("inf"))
        kk = min(self.relation_k, max(n - 1, 1))
        dist, idx = pt.topk(dist_mat, k=kk, dim=-1, largest=False)
        return idx.contiguous(), dist.contiguous()

    def _seq_edges(self, centers):
        n = int(centers.shape[0])
        if n <= 1:
            return centers.new_empty((n, 0), dtype=pt.long), centers.new_empty((n, 0))
        base = pt.arange(n, device=centers.device)
        prev_idx = (base - 1).clamp_min(0)
        next_idx = (base + 1).clamp_max(n - 1)
        idx = pt.stack([prev_idx, next_idx], dim=-1)
        valid = pt.stack([base > 0, base < n - 1], dim=-1)
        dist = (centers[idx] - centers.unsqueeze(1)).norm(dim=-1)
        idx = pt.where(valid, idx, base.unsqueeze(-1).expand_as(idx))
        dist = pt.where(valid, dist, dist.new_full(dist.shape, float("inf")))
        return idx, dist

    def _relation_types(self, idx, dist):
        n, e = idx.shape
        base = pt.arange(n, device=idx.device).unsqueeze(-1)
        sep = (idx - base).abs()
        rel = pt.full((n, e), 6, device=idx.device, dtype=pt.long)
        rel = pt.where(dist < 18.0, pt.full_like(rel, 5), rel)
        rel = pt.where(dist < 12.0, pt.full_like(rel, 4), rel)
        rel = pt.where(dist < 8.0, pt.full_like(rel, 3), rel)
        rel = pt.where(sep <= 4, pt.full_like(rel, 2), rel)
        rel = pt.where(sep <= 2, pt.full_like(rel, 1), rel)
        rel = pt.where(sep == 1, pt.full_like(rel, 0), rel)
        return rel.clamp(0, self.num_relations - 1)

    def _edge_features(self, centers, idx, dist):
        width = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        rbf = pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(device=dist.device, dtype=dist.dtype)) / width) ** 2)
        rel_vec = centers[idx] - centers.unsqueeze(1)
        direction = rel_vec / rel_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sep = idx - pt.arange(centers.shape[0], device=centers.device).unsqueeze(-1)
        rel_abs = pt.log1p(sep.abs().to(dtype=centers.dtype)).unsqueeze(-1) / math.log(max(int(centers.shape[0]), 2))
        rel_sign = sep.sign().to(dtype=centers.dtype).unsqueeze(-1)
        return pt.cat([rbf, direction, rel_abs, rel_sign], dim=-1)

    def _edges(self, centers, graph_cache):
        knn_idx, knn_dist = self._cached_knn(centers, graph_cache)
        seq_idx, seq_dist = self._seq_edges(centers)
        if seq_idx.shape[1] == 0:
            idx, dist = knn_idx, knn_dist
        elif knn_idx.shape[1] == 0:
            idx, dist = seq_idx, seq_dist
        else:
            idx = pt.cat([seq_idx, knn_idx], dim=1)
            dist = pt.cat([seq_dist, knn_dist], dim=1)
        if idx.shape[1] == 0:
            rel = centers.new_empty((centers.shape[0], 0), dtype=pt.long)
            valid = centers.new_empty((centers.shape[0], 0), dtype=pt.bool)
            return idx, dist, rel, valid
        valid = pt.isfinite(dist)
        dist = pt.where(valid, dist, dist.new_zeros(dist.shape))
        rel = self._relation_types(idx, dist)
        return idx, dist, rel, valid

    def forward(self, h_res, centers, graph_cache=None):
        idx, dist, rel_type, valid = self._edges(centers.to(device=h_res.device, dtype=h_res.dtype), graph_cache)
        if idx.shape[1] == 0:
            self.last_diagnostics = {
                "gear_edge_count_mean": h_res.new_tensor(0.0),
                "gear_relation_entropy": h_res.new_tensor(0.0),
                "gear_update_norm": h_res.new_tensor(0.0),
            }
            return h_res
        centers = centers.to(device=h_res.device, dtype=h_res.dtype)
        idx = idx.to(device=h_res.device, dtype=pt.long)
        rel_type = rel_type.to(device=h_res.device, dtype=pt.long)
        valid = valid.to(device=h_res.device)
        dist = dist.to(device=h_res.device, dtype=h_res.dtype)
        edge_extra = self._edge_features(centers, idx, dist) if self.use_edge_message else None
        h = h_res
        update_norms = []
        last_attn = None
        for msg_mlp, attn_mlp, update_mlp, norm in zip(self.message_mlps, self.attn_mlps, self.update_mlps, self.norms):
            center = h.unsqueeze(1).expand(-1, idx.shape[1], -1)
            neigh = h[idx]
            rel_emb = self.relation_embedding(rel_type)
            parts = [center, neigh, rel_emb]
            if edge_extra is not None:
                parts.append(edge_extra.to(dtype=h.dtype))
            msg_input = pt.cat(parts, dim=-1)
            msg = msg_mlp(msg_input)
            attn_logits = attn_mlp(msg_input).squeeze(-1)
            neg = -1.0e4 if attn_logits.dtype in {pt.float16, pt.bfloat16} else pt.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~valid, neg)
            attn = pt.softmax(attn_logits, dim=-1)
            attn = pt.where(valid, attn, attn.new_zeros(attn.shape))
            agg = (attn.unsqueeze(-1) * msg).sum(dim=1)
            update = update_mlp(pt.cat([h, agg], dim=-1))
            update_norms.append(update.detach().norm(dim=-1).mean())
            h = norm(h + self.dropout(update))
            last_attn = attn
        rel_onehot = pt.nn.functional.one_hot(rel_type.clamp(0, self.num_relations - 1), num_classes=self.num_relations).to(dtype=h.dtype)
        rel_counts = (rel_onehot * valid.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=(0, 1))
        rel_prob = rel_counts / rel_counts.sum().clamp_min(1.0)
        rel_entropy = -(rel_prob * rel_prob.clamp_min(self.router_entropy_eps).log()).sum()
        if last_attn is not None:
            attn_entropy = -(last_attn * last_attn.clamp_min(self.router_entropy_eps).log()).sum(dim=-1).mean()
        else:
            attn_entropy = h.new_tensor(0.0)
        self.last_diagnostics = {
            "gear_edge_count_mean": valid.to(dtype=h.dtype).sum(dim=-1).mean(),
            "gear_relation_entropy": rel_entropy.detach(),
            "gear_attention_entropy": attn_entropy.detach(),
            "gear_update_norm": pt.stack(update_norms).mean() if update_norms else h.new_tensor(0.0),
        }
        return h


class OAGNNResidueGraphRefiner(pt.nn.Module):
    """Orientation-aware residue message passing with local frames.

    This is an OAGNN-style residue refiner rather than a gate on top of GVP:
    messages use residue local frames, dot/cross directional features, and
    relation-specific learnable 3D vector weights before the Gaussian surface
    readout.
    """

    def __init__(self, dim=128, relation_k=32, layers=2, dropout=0.1):
        super().__init__()
        self.dim = int(dim)
        self.relation_k = int(relation_k)
        self.layers = int(layers)
        self.num_relations = 8
        centers = pt.linspace(0.0, 30.0, 12)
        self.register_buffer("rbf_centers", centers)
        self.relation_embedding = pt.nn.Embedding(self.num_relations, dim)
        # Relation-specific 3D vector weights are the key OAGNN-style part.
        self.direction_weight = pt.nn.Parameter(pt.empty(self.layers, self.num_relations, dim, 3))
        self.cross_weight = pt.nn.Parameter(pt.empty(self.layers, self.num_relations, dim, 3))
        edge_dim = 12 + 3 + 3 + 3 + 2
        msg_in = dim * 5 + edge_dim
        self.message_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(msg_in),
                    pt.nn.Linear(msg_in, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.attn_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(msg_in),
                    pt.nn.Linear(msg_in, dim // 2),
                    pt.nn.GELU(),
                    pt.nn.Linear(dim // 2, 1),
                )
                for _ in range(self.layers)
            ]
        )
        self.update_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(self.layers)])
        self.dropout = pt.nn.Dropout(dropout)
        self.last_diagnostics = {}
        self.reset_parameters()

    def reset_parameters(self):
        pt.nn.init.normal_(self.direction_weight, mean=0.0, std=0.02)
        pt.nn.init.normal_(self.cross_weight, mean=0.0, std=0.02)

    def _safe_normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _local_frames(self, backbone_vecs):
        if backbone_vecs is None or backbone_vecs.numel() == 0:
            raise ValueError("OAGNNResidueGraphRefiner requires backbone_vecs.")
        vecs = backbone_vecs
        ca = vecs[:, 0] if vecs.shape[1] > 0 else vecs.new_zeros((vecs.shape[0], 3))
        n_vec = vecs[:, 1] if vecs.shape[1] > 1 else ca
        c_vec = vecs[:, 2] if vecs.shape[1] > 2 else ca
        e1 = self._safe_normalize(c_vec - n_vec)
        fallback = self._safe_normalize(ca)
        e1 = pt.where((e1.norm(dim=-1, keepdim=True) > 1e-4), e1, fallback)
        e2 = pt.cross(e1, fallback, dim=-1)
        alt = e1.new_zeros(e1.shape)
        alt[:, 2] = 1.0
        alt2 = e1.new_zeros(e1.shape)
        alt2[:, 1] = 1.0
        alt = pt.where((pt.cross(e1, alt, dim=-1).norm(dim=-1, keepdim=True) > 1e-4), alt, alt2)
        e2 = pt.where((e2.norm(dim=-1, keepdim=True) > 1e-4), e2, pt.cross(e1, alt, dim=-1))
        e2 = self._safe_normalize(e2)
        e3 = self._safe_normalize(pt.cross(e1, e2, dim=-1))
        return pt.stack([e1, e2, e3], dim=1)

    def _cached_knn(self, centers, graph_cache):
        n = int(centers.shape[0])
        if n <= 1 or self.relation_k <= 0:
            return centers.new_empty((n, 0), dtype=pt.long), centers.new_empty((n, 0))
        if graph_cache:
            idx = graph_cache.get(f"knn_idx_k{self.relation_k}")
            dist = graph_cache.get(f"knn_dist_k{self.relation_k}")
            if idx is not None and dist is not None:
                idx = idx.to(device=centers.device, dtype=pt.long)
                dist = dist.to(device=centers.device, dtype=centers.dtype)
                return idx[:, : min(idx.shape[1], self.relation_k)], dist[:, : min(dist.shape[1], self.relation_k)]
        dist_mat = pt.cdist(centers, centers).clamp_min(0.0)
        dist_mat.fill_diagonal_(float("inf"))
        kk = min(self.relation_k, max(n - 1, 1))
        dist, idx = pt.topk(dist_mat, k=kk, dim=-1, largest=False)
        return idx.contiguous(), dist.contiguous()

    def _seq_edges(self, centers):
        n = int(centers.shape[0])
        if n <= 1:
            return centers.new_empty((n, 0), dtype=pt.long), centers.new_empty((n, 0))
        base = pt.arange(n, device=centers.device)
        prev_idx = (base - 1).clamp_min(0)
        next_idx = (base + 1).clamp_max(n - 1)
        idx = pt.stack([prev_idx, next_idx], dim=-1)
        valid = pt.stack([base > 0, base < n - 1], dim=-1)
        dist = (centers[idx] - centers.unsqueeze(1)).norm(dim=-1)
        idx = pt.where(valid, idx, base.unsqueeze(-1).expand_as(idx))
        dist = pt.where(valid, dist, dist.new_full(dist.shape, float("inf")))
        return idx, dist

    def _edges(self, centers, graph_cache):
        knn_idx, knn_dist = self._cached_knn(centers, graph_cache)
        seq_idx, seq_dist = self._seq_edges(centers)
        if seq_idx.shape[1] == 0:
            idx, dist = knn_idx, knn_dist
        elif knn_idx.shape[1] == 0:
            idx, dist = seq_idx, seq_dist
        else:
            idx = pt.cat([seq_idx, knn_idx], dim=1)
            dist = pt.cat([seq_dist, knn_dist], dim=1)
        valid = pt.isfinite(dist) if idx.shape[1] > 0 else centers.new_empty((centers.shape[0], 0), dtype=pt.bool)
        dist = pt.where(valid, dist, dist.new_zeros(dist.shape)) if idx.shape[1] > 0 else dist
        rel = self._relation_types(idx, dist) if idx.shape[1] > 0 else centers.new_empty((centers.shape[0], 0), dtype=pt.long)
        return idx, dist, rel, valid

    def _relation_types(self, idx, dist):
        n, e = idx.shape
        base = pt.arange(n, device=idx.device).unsqueeze(-1)
        sep = (idx - base).abs()
        rel = pt.full((n, e), 7, device=idx.device, dtype=pt.long)
        rel = pt.where(dist < 18.0, pt.full_like(rel, 6), rel)
        rel = pt.where(dist < 12.0, pt.full_like(rel, 5), rel)
        rel = pt.where(dist < 8.0, pt.full_like(rel, 4), rel)
        rel = pt.where(sep <= 8, pt.minimum(rel, pt.full_like(rel, 3)), rel)
        rel = pt.where(sep <= 4, pt.minimum(rel, pt.full_like(rel, 2)), rel)
        rel = pt.where(sep <= 2, pt.minimum(rel, pt.full_like(rel, 1)), rel)
        rel = pt.where(sep == 1, pt.full_like(rel, 0), rel)
        return rel.clamp(0, self.num_relations - 1)

    def _edge_geometry(self, centers, idx, dist, frames):
        width = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        rbf = pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(device=dist.device, dtype=dist.dtype)) / width) ** 2)
        rel_vec = centers[idx] - centers.unsqueeze(1)
        direction = self._safe_normalize(rel_vec)
        frame_i = frames.unsqueeze(1).expand(-1, idx.shape[1], -1, -1)
        frame_j = frames[idx]
        local_dir = (direction.unsqueeze(-2) * frame_i).sum(dim=-1)
        cross_vec = pt.cross(direction, frame_i[:, :, 0, :], dim=-1)
        cross_local = (cross_vec.unsqueeze(-2) * frame_i).sum(dim=-1)
        orient = (frame_i * frame_j).sum(dim=-1).clamp(-1.0, 1.0)
        sep = idx - pt.arange(centers.shape[0], device=centers.device).unsqueeze(-1)
        rel_abs = pt.log1p(sep.abs().to(dtype=centers.dtype)).unsqueeze(-1) / math.log(max(int(centers.shape[0]), 2))
        rel_sign = sep.sign().to(dtype=centers.dtype).unsqueeze(-1)
        edge_feat = pt.cat([rbf, local_dir, cross_local, orient, rel_abs, rel_sign], dim=-1)
        return edge_feat, local_dir, cross_local

    def forward(self, h_res, centers, backbone_vecs, graph_cache=None):
        centers = centers.to(device=h_res.device, dtype=h_res.dtype)
        backbone_vecs = backbone_vecs.to(device=h_res.device, dtype=h_res.dtype)
        idx, dist, rel_type, valid = self._edges(centers, graph_cache)
        if idx.shape[1] == 0:
            self.last_diagnostics = {
                "oagnn_edge_count_mean": h_res.new_tensor(0.0),
                "oagnn_relation_entropy": h_res.new_tensor(0.0),
                "oagnn_update_norm": h_res.new_tensor(0.0),
            }
            return h_res
        idx = idx.to(device=h_res.device, dtype=pt.long)
        dist = dist.to(device=h_res.device, dtype=h_res.dtype)
        rel_type = rel_type.to(device=h_res.device, dtype=pt.long)
        valid = valid.to(device=h_res.device)
        frames = self._local_frames(backbone_vecs)
        edge_feat, local_dir, cross_local = self._edge_geometry(centers, idx, dist, frames)
        h = h_res
        update_norms = []
        last_attn = None
        for layer_idx, (msg_mlp, attn_mlp, update_mlp, norm) in enumerate(
            zip(self.message_mlps, self.attn_mlps, self.update_mlps, self.norms)
        ):
            center = h.unsqueeze(1).expand(-1, idx.shape[1], -1)
            neigh = h[idx]
            rel_emb = self.relation_embedding(rel_type)
            dir_w = self.direction_weight[layer_idx][rel_type]
            cross_w = self.cross_weight[layer_idx][rel_type]
            dir_bias = (dir_w * local_dir.unsqueeze(-2)).sum(dim=-1)
            cross_bias = (cross_w * cross_local.unsqueeze(-2)).sum(dim=-1)
            msg_input = pt.cat([center, neigh, rel_emb, dir_bias, cross_bias, edge_feat.to(dtype=h.dtype)], dim=-1)
            msg = msg_mlp(msg_input)
            attn_logits = attn_mlp(msg_input).squeeze(-1)
            neg = -1.0e4 if attn_logits.dtype in {pt.float16, pt.bfloat16} else pt.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~valid, neg)
            attn = pt.softmax(attn_logits, dim=-1)
            attn = pt.where(valid, attn, attn.new_zeros(attn.shape))
            agg = (attn.unsqueeze(-1) * msg).sum(dim=1)
            update = update_mlp(pt.cat([h, agg], dim=-1))
            update_norms.append(update.detach().norm(dim=-1).mean())
            h = norm(h + self.dropout(update))
            last_attn = attn
        rel_onehot = pt.nn.functional.one_hot(rel_type.clamp(0, self.num_relations - 1), num_classes=self.num_relations).to(dtype=h.dtype)
        rel_counts = (rel_onehot * valid.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=(0, 1))
        rel_prob = rel_counts / rel_counts.sum().clamp_min(1.0)
        rel_entropy = -(rel_prob * rel_prob.clamp_min(1e-8).log()).sum()
        attn_entropy = (
            -(last_attn * last_attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
            if last_attn is not None
            else h.new_tensor(0.0)
        )
        self.last_diagnostics = {
            "oagnn_edge_count_mean": valid.to(dtype=h.dtype).sum(dim=-1).mean(),
            "oagnn_relation_entropy": rel_entropy.detach(),
            "oagnn_attention_entropy": attn_entropy.detach(),
            "oagnn_update_norm": pt.stack(update_norms).mean() if update_norms else h.new_tensor(0.0),
            "oagnn_direction_weight_norm": self.direction_weight.detach().norm(),
            "oagnn_cross_weight_norm": self.cross_weight.detach().norm(),
            "oagnn_local_dir_std": local_dir.detach().std(unbiased=False),
        }
        return h


class SurfacePointGraphEncoder(pt.nn.Module):
    def __init__(self, dim=128, geom_dim=6, dropout=0.1, layers=2):
        super().__init__()
        edge_dim = 8
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.edge_mlp = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2 + edge_dim),
                    pt.nn.Linear(dim * 2 + edge_dim, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(int(layers))
            ]
        )
        self.gates = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.Sigmoid(),
                )
                for _ in range(int(layers))
            ]
        )
        self.norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(int(layers))])

    def _edge_features(self, surface_pos, surface_normal, graph_idx, graph_dist):
        p = int(surface_pos.shape[0])
        if graph_idx is None or graph_idx.numel() == 0:
            return surface_pos.new_empty((p, 0, 8))
        idx = graph_idx.to(device=surface_pos.device, dtype=pt.long).clamp(0, max(p - 1, 0))
        neigh_pos = surface_pos[idx]
        rel = neigh_pos - surface_pos.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist
        neigh_normal = surface_normal[idx]
        center_normal = surface_normal.unsqueeze(1).expand_as(neigh_normal)
        normal_dot = (center_normal * neigh_normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        tangent = (direction * center_normal).sum(dim=-1, keepdim=True)
        if graph_dist is None or graph_dist.numel() == 0:
            graph_dist = dist.squeeze(-1)
        graph_dist = graph_dist.to(device=surface_pos.device, dtype=surface_pos.dtype).unsqueeze(-1)
        return pt.cat([dist, graph_dist, direction, normal_dot, tangent, tangent.abs()], dim=-1)

    def forward(self, h_surf0, surface_geom, surface_pos, surface_normal, graph_idx, graph_dist=None):
        geom = surface_geom.to(device=h_surf0.device, dtype=h_surf0.dtype)
        x = self.input_proj(pt.cat([h_surf0, geom], dim=-1))
        surface_pos = surface_pos.to(device=x.device, dtype=x.dtype)
        surface_normal = surface_normal.to(device=x.device, dtype=x.dtype)
        edge_feat = self._edge_features(surface_pos, surface_normal, graph_idx, graph_dist)
        if graph_idx is None or graph_idx.numel() == 0:
            return x
        idx = graph_idx.to(device=x.device, dtype=pt.long).clamp(0, max(x.shape[0] - 1, 0))
        for layer, gate, norm in zip(self.edge_mlp, self.gates, self.norms):
            neigh = x[idx]
            center = x.unsqueeze(1).expand_as(neigh)
            msg = layer(pt.cat([center, neigh, edge_feat.to(device=x.device, dtype=x.dtype)], dim=-1))
            ctx = msg.mean(dim=1)
            x = norm(x + gate(pt.cat([x, ctx], dim=-1)) * ctx)
        return x


class SurfacePointGVPGraphEncoder(pt.nn.Module):
    """Surface-point graph update with scalar/vector GVP message passing."""

    def __init__(self, dim=128, vector_dim=16, geom_dim=6, dropout=0.1, layers=2):
        super().__init__()
        edge_dim = 8
        self.dim = int(dim)
        self.vector_dim = int(vector_dim)
        self.input_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.node_vector_proj = pt.nn.Linear(1, self.vector_dim, bias=False)
        self.edge_scalar_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(edge_dim),
            pt.nn.Linear(edge_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.edge_vector_proj = pt.nn.Linear(1, self.vector_dim, bias=False)
        self.layers = pt.nn.ModuleList(
            [FullGVPMessageLayer((dim, self.vector_dim), (dim, self.vector_dim), dropout=dropout) for _ in range(int(layers))]
        )
        self.last_surface_vectors = None

    def _edge_features(self, surface_pos, surface_normal, graph_idx, graph_dist):
        p = int(surface_pos.shape[0])
        if graph_idx is None or graph_idx.numel() == 0:
            return surface_pos.new_empty((p, 0, 8)), surface_pos.new_empty((p, 0, 1, 3))
        idx = graph_idx.to(device=surface_pos.device, dtype=pt.long).clamp(0, max(p - 1, 0))
        neigh_pos = surface_pos[idx]
        rel = neigh_pos - surface_pos.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist
        neigh_normal = surface_normal[idx]
        center_normal = surface_normal.unsqueeze(1).expand_as(neigh_normal)
        normal_dot = (center_normal * neigh_normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        tangent = (direction * center_normal).sum(dim=-1, keepdim=True)
        if graph_dist is None or graph_dist.numel() == 0:
            graph_dist = dist.squeeze(-1)
        graph_dist = graph_dist.to(device=surface_pos.device, dtype=surface_pos.dtype).unsqueeze(-1)
        edge_s = pt.cat([dist, graph_dist, direction, normal_dot, tangent, tangent.abs()], dim=-1)
        return edge_s, direction.unsqueeze(-2)

    def _project_vectors(self, vectors, proj):
        return proj(vectors.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, h_surf0, surface_geom, surface_pos, surface_normal, graph_idx, graph_dist=None, initial_vectors=None):
        geom = surface_geom.to(device=h_surf0.device, dtype=h_surf0.dtype)
        s = self.input_proj(pt.cat([h_surf0, geom], dim=-1))
        surface_pos = surface_pos.to(device=s.device, dtype=s.dtype)
        surface_normal = surface_normal.to(device=s.device, dtype=s.dtype)
        if initial_vectors is not None and initial_vectors.shape == (s.shape[0], self.vector_dim, 3):
            v = initial_vectors.to(device=s.device, dtype=s.dtype)
        else:
            v = self._project_vectors(surface_normal.unsqueeze(-2), self.node_vector_proj)
        if graph_idx is None or graph_idx.numel() == 0:
            self.last_surface_vectors = v
            return s
        idx = graph_idx.to(device=s.device, dtype=pt.long).clamp(0, max(s.shape[0] - 1, 0))
        edge_feat, edge_vec = self._edge_features(surface_pos, surface_normal, idx, graph_dist)
        edge_s = self.edge_scalar_proj(edge_feat.to(device=s.device, dtype=s.dtype))
        edge_v = self._project_vectors(edge_vec.to(device=s.device, dtype=s.dtype), self.edge_vector_proj)
        for layer in self.layers:
            s, v = layer(s, v, edge_s, edge_v, idx)
        self.last_surface_vectors = v
        return s


class ResidueVectorToSurfaceCrossAttention(pt.nn.Module):
    """Equivariant residue-vector value readout for surface GVP initialization."""

    def __init__(self, dim=128, vector_dim=16, geom_dim=6, residue_topk=16, dropout=0.1):
        super().__init__()
        self.residue_topk = max(1, int(residue_topk))
        self.vector_dim = int(vector_dim)
        score_dim = dim * 2 + geom_dim + 4
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(score_dim),
            pt.nn.Linear(score_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.value_weight = pt.nn.Parameter(pt.eye(self.vector_dim))
        self.last_diagnostics = {}

    def forward(self, h_surf, h_res, residue_vectors, surface_pos, surface_normal, surface_geom, residue_centers):
        self.last_diagnostics = {}
        if residue_vectors is None or residue_vectors.ndim != 3 or residue_vectors.shape[0] != h_res.shape[0]:
            return None
        if residue_vectors.shape[1] != self.vector_dim or residue_vectors.shape[2] != 3:
            return None
        if h_surf.numel() == 0 or h_res.numel() == 0:
            return None

        device = h_surf.device
        dtype = h_surf.dtype
        centers = residue_centers.to(device=device, dtype=dtype)
        surf_pos = surface_pos.to(device=device, dtype=dtype)
        surf_normal = surface_normal.to(device=device, dtype=dtype)
        surf_geom = surface_geom.to(device=device, dtype=dtype)
        res_v_all = residue_vectors.to(device=device, dtype=dtype)

        dist = pt.cdist(surf_pos, centers).clamp_min(0.0)
        k = min(self.residue_topk, centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        local_dist = dist.gather(1, idx).clamp_min(1e-6)
        rel = centers[idx] - surf_pos.unsqueeze(1)
        direction = rel / local_dist.unsqueeze(-1)
        normal = surf_normal / surf_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal_proj = (direction * normal.unsqueeze(1)).sum(dim=-1, keepdim=True)
        tangent_dist = (local_dist.square() - (local_dist * normal_proj.squeeze(-1)).square()).clamp_min(0.0).sqrt()

        surf_expand = h_surf.unsqueeze(1).expand(-1, k, -1)
        res = h_res[idx]
        geom_pair = pt.cat(
            [
                local_dist.unsqueeze(-1),
                tangent_dist.unsqueeze(-1),
                (local_dist * normal_proj.squeeze(-1)).abs().unsqueeze(-1),
                normal_proj,
                surf_geom.unsqueeze(1).expand(-1, k, -1),
            ],
            dim=-1,
        )
        logits = self.score(pt.cat([surf_expand, res, geom_pair], dim=-1)).squeeze(-1)
        attn = pt.softmax(logits, dim=-1)
        res_v = res_v_all[idx]
        mixed_v = pt.einsum("oi,pkic->pkoc", self.value_weight.to(device=device, dtype=dtype), res_v)
        out = (mixed_v * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        entropy = -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
        self.last_diagnostics = {
            "residue_vector_surface_attn_entropy": entropy.detach(),
            "residue_vector_surface_attn_max": attn.max(dim=-1).values.mean().detach(),
            "residue_vector_surface_value_norm": out.norm(dim=-1).mean().detach(),
            "residue_vector_surface_topk": h_surf.new_tensor(float(k)).detach(),
        }
        return out


class DynamicSurfacePatchCluster(pt.nn.Module):
    """Compress dense surface points into learnable patch tokens."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        num_patches=32,
        mode="none",
        beta=0.1,
        dropout=0.1,
        attention_heads=4,
        lappe_dim=0,
    ):
        super().__init__()
        self.num_patches = int(num_patches)
        self.mode = str(mode)
        self.beta = float(beta)
        self.lappe_dim = max(0, int(lappe_dim))
        self.assignment = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2 + geom_dim + 3 + self.lappe_dim),
            pt.nn.Linear(dim * 2 + geom_dim + 3 + self.lappe_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.num_patches),
        )
        self.patch_token = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim + 7 + self.lappe_dim),
            pt.nn.Linear(dim + geom_dim + 7 + self.lappe_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.patch_importance = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
            pt.nn.Sigmoid(),
        )
        heads = max(1, int(attention_heads))
        if dim % heads != 0:
            heads = 1
        self.patch_self_attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.residue_patch_attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.patch_norm = pt.nn.LayerNorm(dim)
        self.residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        if self.connected_mode:
            for parameter in self.assignment.parameters():
                parameter.requires_grad_(False)

    @property
    def connected_mode(self):
        return self.mode in {"connected_pool", "connected_pool_adj"}

    def enabled(self):
        return self.mode in {
            "gate", "attn", "self_attn", "replace_attn", "independent_attn",
            "connected_pool", "connected_pool_adj",
        } and self.num_patches > 0 and self.beta != 0.0

    @staticmethod
    def _undirected_edges(graph_idx, num_points):
        graph_idx = graph_idx.to(dtype=pt.long)
        if graph_idx.ndim != 2 or graph_idx.shape[0] != num_points:
            raise ValueError(
                f"surface graph must have shape [P, K], got {tuple(graph_idx.shape)} for P={num_points}"
            )
        rows = pt.arange(num_points, device=graph_idx.device, dtype=pt.long).unsqueeze(1).expand_as(graph_idx)
        valid = (graph_idx >= 0) & (graph_idx < num_points) & (graph_idx != rows)
        source = pt.cat([rows[valid], graph_idx[valid]], dim=0)
        target = pt.cat([graph_idx[valid], rows[valid]], dim=0)
        if source.numel() == 0:
            raise ValueError("surface graph has no usable edges")
        edge_code = source * num_points + target
        edge_code = pt.unique(edge_code, sorted=True)
        return edge_code.div(num_points, rounding_mode="floor"), edge_code.remainder(num_points)

    def _euclidean_fps_seeds(self, surface_pos):
        num_points = int(surface_pos.shape[0])
        patch_count = min(self.num_patches, num_points)
        seeds = []
        position = surface_pos.detach().float()
        centered_pos = position - position.mean(dim=0, keepdim=True)
        next_seed = int(pt.argmax(centered_pos.square().sum(dim=-1)).item())
        min_distance_sq = position.new_full((num_points,), float("inf"))
        for _ in range(patch_count):
            seeds.append(next_seed)
            distance_sq = (position - position[next_seed]).square().sum(dim=-1)
            min_distance_sq = pt.minimum(min_distance_sq, distance_sq)
            min_distance_sq[position.new_tensor(seeds, dtype=pt.long)] = -1.0
            if len(seeds) < patch_count:
                next_seed = int(pt.argmax(min_distance_sq).item())
        return surface_pos.new_tensor(seeds, dtype=pt.long)

    def connected_partition(self, surface_pos, surface_graph_idx):
        num_points = int(surface_pos.shape[0])
        if num_points <= 0:
            raise ValueError("connected patch mode requires at least one surface point")
        patch_count = min(self.num_patches, num_points)
        graph_idx = surface_graph_idx.to(device=surface_pos.device, dtype=pt.long)
        with pt.no_grad():
            if num_points == 1:
                source = graph_idx.new_zeros((1,))
                target = graph_idx.new_zeros((1,))
            else:
                source, target = self._undirected_edges(graph_idx, num_points)
            seeds = self._euclidean_fps_seeds(surface_pos)
            labels = source.new_full((num_points,), -1)
            labels[seeds] = pt.arange(patch_count, device=source.device, dtype=pt.long)
            hop_distance = source.new_full((num_points,), -1)
            hop_distance[seeds] = 0
            for hop in range(1, num_points + 1):
                candidate = labels[target]
                candidate = pt.where(candidate >= 0, candidate, candidate.new_full(candidate.shape, patch_count))
                best = labels.new_full((num_points,), patch_count)
                best.scatter_reduce_(0, source, candidate, reduce="amin", include_self=True)
                frontier = (labels < 0) & (best < patch_count)
                if not pt.any(frontier):
                    break
                labels[frontier] = best[frontier]
                hop_distance[frontier] = hop
            if pt.any(labels < 0):
                missing = int((labels < 0).sum().item())
                raise ValueError(f"connected patch propagation left {missing} surface points unassigned")
            assign = pt.nn.functional.one_hot(labels, num_classes=patch_count).to(dtype=surface_pos.dtype)
            patch_adjacency = pt.zeros(
                (patch_count, patch_count), device=surface_pos.device, dtype=pt.bool
            )
            patch_adjacency[labels[source], labels[target]] = True
            patch_adjacency.fill_diagonal_(True)
            patch_sizes = pt.bincount(labels, minlength=patch_count).to(dtype=pt.float32)
            patch_mean = patch_sizes.mean().clamp_min(1.0)
            cross_edge = labels[source] != labels[target]
            patch_degree = patch_adjacency.sum(dim=-1).to(dtype=pt.float32) - 1.0
            diagnostics = {
                "connected_patch_components_mean": surface_pos.new_tensor(1.0),
                "connected_patch_components_max": surface_pos.new_tensor(1.0),
                "connected_patch_size_min": patch_sizes.min().to(device=surface_pos.device),
                "connected_patch_size_max": patch_sizes.max().to(device=surface_pos.device),
                "connected_patch_size_cv": (patch_sizes.std(unbiased=False) / patch_mean).to(device=surface_pos.device),
                "connected_patch_boundary_fraction": cross_edge.float().mean().to(device=surface_pos.device),
                "connected_patch_degree_mean": patch_degree.mean().to(device=surface_pos.device),
                "connected_patch_hop_radius_mean": hop_distance.float().mean().to(device=surface_pos.device),
                "connected_patch_hop_radius_max": hop_distance.max().float().to(device=surface_pos.device),
            }
        return {
            "assign": assign,
            "labels": labels,
            "adjacency": patch_adjacency,
            "hop_distance": hop_distance,
            "surface_graph_idx": graph_idx.detach(),
            "seeds": seeds.detach(),
            "diagnostics": diagnostics,
        }

    def _summary(self, assign, usage, patch_pos, patch_normal, patch_tokens, centered_pos, surface_normal, patch_attn=None, patch_update=None, residue_gate=None, extra_diagnostics=None):
        effective_patch_count = int(assign.shape[1])
        entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(effective_patch_count, 2)), 1e-8)
        usage_norm = usage / usage.sum().clamp_min(1e-8)
        active = (usage_norm > (0.25 / max(effective_patch_count, 1))).float().sum()
        usage_entropy = -(usage_norm * usage_norm.clamp_min(1e-8).log()).sum()
        usage_entropy = usage_entropy / max(math.log(max(effective_patch_count, 2)), 1e-8)
        displacement = centered_pos.unsqueeze(1) - patch_pos.unsqueeze(0)
        compactness = (assign * displacement.square().sum(dim=-1)).sum() / usage.sum().clamp_min(1e-8)
        normal_alignment = (surface_normal.unsqueeze(1) * patch_normal.unsqueeze(0)).sum(dim=-1).abs()
        normal_consistency = (assign * normal_alignment).sum() / usage.sum().clamp_min(1e-8)
        token_unit = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        token_cos = token_unit.matmul(token_unit.transpose(0, 1))
        offdiag = ~pt.eye(token_cos.shape[0], device=token_cos.device, dtype=pt.bool)
        token_diversity = 1.0 - token_cos[offdiag].mean() if pt.any(offdiag) else token_cos.new_zeros(())
        out = {
            "patch_assignment_entropy": entropy.detach(),
            "patch_count_effective": assign.new_tensor(float(effective_patch_count)).detach(),
            "patch_usage": usage_norm.detach(),
            "patch_usage_active": active.detach(),
            "patch_usage_entropy": usage_entropy.detach(),
            "patch_spatial_compactness": compactness.detach(),
            "patch_normal_consistency": normal_consistency.detach(),
            "patch_token_norm_mean": patch_tokens.norm(dim=-1).mean().detach(),
            "patch_token_diversity": token_diversity.detach(),
        }
        if patch_attn is not None:
            attn = patch_attn.squeeze(0).clamp_min(1e-8)
            attn_entropy = -(attn * attn.log()).sum(dim=-1)
            out.update({
                "patch_residue_attention_entropy": attn_entropy.mean().detach(),
                "patch_residue_attention_max": attn.max(dim=-1).values.mean().detach(),
                "patch_residue_effective_count": attn_entropy.exp().mean().detach(),
            })
        if patch_update is not None:
            out["patch_context_update_norm"] = patch_update.norm(dim=-1).mean().detach()
        if residue_gate is not None:
            out["patch_residue_gate_mean"] = residue_gate.mean().detach()
            out["patch_residue_gate_std"] = residue_gate.std(unbiased=False).detach()
        if extra_diagnostics:
            out.update({key: value.detach() for key, value in extra_diagnostics.items()})
        return out

    def forward(
        self,
        h_res,
        h_surf,
        surface_pos,
        surface_normal,
        surface_geom,
        rs_idx,
        rs_weight,
        surf_context=None,
        patch_state=None,
        surface_lappe=None,
        connected_partition=None,
    ):
        if not self.enabled():
            return h_surf, surf_context, {}
        device = h_surf.device
        dtype = h_surf.dtype
        rs_idx = rs_idx.to(device=device, dtype=pt.long)
        rs_weight = rs_weight.to(device=device, dtype=dtype)
        nearest_h = (h_res[rs_idx] * rs_weight.unsqueeze(-1)).sum(dim=1)
        surface_normal = surface_normal.to(device=device, dtype=dtype)
        surface_geom = surface_geom.to(device=device, dtype=dtype)
        lappe = None
        if self.lappe_dim > 0:
            if surface_lappe is None:
                lappe = h_surf.new_zeros((h_surf.shape[0], self.lappe_dim))
            else:
                lappe = surface_lappe.to(device=device, dtype=dtype)
                if lappe.shape[-1] < self.lappe_dim:
                    lappe = pt.cat([lappe, lappe.new_zeros((lappe.shape[0], self.lappe_dim - lappe.shape[-1]))], dim=-1)
                elif lappe.shape[-1] > self.lappe_dim:
                    lappe = lappe[:, : self.lappe_dim]
        if self.connected_mode:
            if connected_partition is None:
                raise ValueError("connected patch mode requires a precomputed connected partition")
            assign = connected_partition["assign"].to(device=device, dtype=dtype)
        else:
            assign_parts = [h_surf, nearest_h, surface_geom, surface_normal]
            if lappe is not None:
                assign_parts.append(lappe)
            assign_logits = self.assignment(pt.cat(assign_parts, dim=-1))
            assign = pt.softmax(assign_logits, dim=-1)
        usage = assign.sum(dim=0).clamp_min(1e-6)
        patch_h = assign.transpose(0, 1).matmul(h_surf) / usage.unsqueeze(-1)
        centered_pos = surface_pos.to(device=device, dtype=dtype) - surface_pos.to(device=device, dtype=dtype).mean(dim=0, keepdim=True)
        patch_pos = assign.transpose(0, 1).matmul(centered_pos) / usage.unsqueeze(-1)
        patch_normal = assign.transpose(0, 1).matmul(surface_normal) / usage.unsqueeze(-1)
        patch_normal = patch_normal / patch_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        patch_geom = assign.transpose(0, 1).matmul(surface_geom) / usage.unsqueeze(-1)
        patch_lappe = None
        if lappe is not None:
            patch_lappe = assign.transpose(0, 1).matmul(lappe) / usage.unsqueeze(-1)
        density = (usage / max(float(assign.shape[0]), 1.0)).unsqueeze(-1)
        token_parts = [patch_h, patch_pos, patch_normal, patch_geom, density]
        if patch_lappe is not None:
            token_parts.append(patch_lappe)
        patch_tokens = self.patch_token(pt.cat(token_parts, dim=-1))
        if patch_state is not None and patch_state.shape == patch_tokens.shape:
            patch_tokens = self.patch_norm(patch_tokens + patch_state.to(device=device, dtype=dtype))
        if self.mode in {"self_attn", "connected_pool_adj"}:
            attention_mask = None
            if self.mode == "connected_pool_adj":
                attention_mask = ~connected_partition["adjacency"].to(device=device, dtype=pt.bool)
            attn_out, _ = self.patch_self_attn(
                patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0),
                attn_mask=attention_mask, need_weights=False,
            )
            patch_tokens = self.patch_norm(patch_tokens + attn_out.squeeze(0))
        replay_patch_gate = self.patch_importance(patch_tokens)
        replay_surface_gate = assign.matmul(replay_patch_gate).squeeze(-1)
        patch_gate = replay_patch_gate if self.mode == "gate" else None
        surface_gate = replay_surface_gate if self.mode == "gate" else None
        patch_attn = None
        patch_context = None
        gate = None
        h_surf_out = h_surf
        surf_context_out = surf_context
        if self.mode == "gate":
            h_surf_out = h_surf * (1.0 + self.beta * surface_gate.unsqueeze(-1))
        elif self.mode in {"attn", "self_attn", "replace_attn", "independent_attn", "connected_pool", "connected_pool_adj"} and surf_context is not None:
            ctx, _ = self.residue_patch_attn(
                h_res.unsqueeze(0), patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0),
                need_weights=False,
            )
            if not self.training:
                with pt.no_grad():
                    _unused_ctx, patch_attn = self.residue_patch_attn(
                        h_res.unsqueeze(0), patch_tokens.unsqueeze(0), patch_tokens.unsqueeze(0),
                        need_weights=True, average_attn_weights=True,
                    )
            ctx = ctx.squeeze(0)
            gate = self.residue_gate(pt.cat([h_res, ctx], dim=-1))
            patch_context = self.beta * gate * ctx
            if self.mode == "replace_attn":
                surf_context_out = patch_context
            elif self.mode == "independent_attn":
                # Patch tokens form an independent residue update path; they do
                # not rewrite the dense surface tokens, only the residue context.
                surf_context_out = surf_context + patch_context
            else:
                surf_context_out = surf_context + patch_context
        diag = self._summary(
            assign, usage, patch_pos, patch_normal, patch_tokens, centered_pos, surface_normal,
            patch_attn=patch_attn, patch_update=patch_context, residue_gate=gate,
            extra_diagnostics=(connected_partition.get("diagnostics") if connected_partition is not None else None),
        )
        if patch_gate is not None:
            diag["patch_surface_gate"] = surface_gate.detach()
            diag["patch_gate"] = patch_gate.squeeze(-1).detach()
        elif self.connected_mode:
            diag["patch_surface_gate"] = replay_surface_gate.detach()
            diag["patch_gate"] = replay_patch_gate.squeeze(-1).detach()
        diag["patch_tokens"] = patch_tokens
        diag["patch_positions"] = patch_pos
        diag["patch_assignment"] = assign
        if connected_partition is not None:
            diag["patch_adjacency"] = connected_partition["adjacency"].to(device=device)
            diag["patch_surface_graph_idx"] = connected_partition["surface_graph_idx"].to(device=device)
            diag["patch_seed_surface_idx"] = connected_partition["seeds"].to(device=device)
        primary_surface_idx = assign.argmax(dim=0).to(device=device, dtype=pt.long)
        diag["patch_token_primary_residue_idx"] = rs_idx[primary_surface_idx, 0].detach().to(device=device, dtype=pt.long)
        return h_surf_out, surf_context_out, diag


class SurfaceLaplacianSpectralFilter(pt.nn.Module):
    """Low-rank surface operator using cached Laplacian eigenvectors."""

    def __init__(self, dim=128, lappe_dim=16, dropout=0.1):
        super().__init__()
        self.lappe_dim = max(0, int(lappe_dim))
        self.freq_logits = pt.nn.Parameter(pt.zeros(self.lappe_dim))
        self.update_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.update_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h_surf, surface_lappe):
        if self.lappe_dim <= 0 or surface_lappe is None:
            return h_surf
        u = surface_lappe.to(device=h_surf.device, dtype=h_surf.dtype)
        if u.shape[-1] < self.lappe_dim:
            u = pt.cat([u, u.new_zeros((u.shape[0], self.lappe_dim - u.shape[-1]))], dim=-1)
        elif u.shape[-1] > self.lappe_dim:
            u = u[:, : self.lappe_dim]
        # Normalize columns so U^T H is a stable graph-spectral projection even
        # when eigenvectors come from cached approximations.
        u = u / u.norm(dim=0, keepdim=True).clamp_min(1e-6)
        coeff = u.transpose(0, 1).matmul(h_surf)
        gate = pt.sigmoid(self.freq_logits.to(device=h_surf.device, dtype=h_surf.dtype)).view(-1, 1)
        filtered = u.matmul(coeff * gate)
        update = self.update_proj(filtered)
        mix = self.update_gate(pt.cat([h_surf, update], dim=-1))
        return self.norm(h_surf + mix * update)


class ResidueInternalLaplacianOperator(pt.nn.Module):
    """Learned protein-internal diffusion over the residue contact graph."""

    def __init__(self, dim=128, relation_k=16, layers=2, beta=0.1, dropout=0.1):
        super().__init__()
        self.relation_k = max(2, int(relation_k))
        self.layers = max(1, int(layers))
        self.beta = float(beta)
        self.update_mlps = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.gates = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2),
                    pt.nn.Linear(dim * 2, dim),
                    pt.nn.Sigmoid(),
                )
                for _ in range(self.layers)
            ]
        )
        self.norms = pt.nn.ModuleList([pt.nn.LayerNorm(dim) for _ in range(self.layers)])
        self.last_diagnostics = {}

    def _knn(self, centers, graph_cache):
        n = int(centers.shape[0])
        if n <= 1:
            empty_idx = centers.new_empty((n, 0), dtype=pt.long)
            empty_dist = centers.new_empty((n, 0))
            return empty_idx, empty_dist
        if graph_cache:
            idx = graph_cache.get(f"knn_idx_k{self.relation_k}")
            dist = graph_cache.get(f"knn_dist_k{self.relation_k}")
            if isinstance(idx, pt.Tensor) and isinstance(dist, pt.Tensor) and idx.numel() > 0:
                idx = idx.to(device=centers.device, dtype=pt.long)
                dist = dist.to(device=centers.device, dtype=centers.dtype)
                return idx[:, : min(idx.shape[1], n - 1)], dist[:, : min(dist.shape[1], n - 1)]
        dist_mat = pt.cdist(centers, centers).clamp_min(0.0)
        dist_mat = dist_mat + pt.eye(n, device=centers.device, dtype=centers.dtype) * 1e6
        k = min(self.relation_k, n - 1)
        dist, idx = pt.topk(dist_mat, k=k, dim=-1, largest=False)
        return idx, dist

    def forward(self, h_res, centers, graph_cache=None):
        self.last_diagnostics = {}
        if self.beta == 0.0 or h_res.shape[0] <= 1:
            return h_res
        centers = centers.to(device=h_res.device, dtype=h_res.dtype)
        idx, dist = self._knn(centers, graph_cache)
        if idx.shape[1] == 0:
            return h_res
        valid = pt.isfinite(dist)
        weight = pt.softmax(-dist.masked_fill(~valid, 1e6) / dist.detach().median().clamp_min(1.0), dim=-1)
        weight = weight * valid.to(dtype=h_res.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        h = h_res
        diffusion_norms = []
        for update_mlp, gate_mlp, norm in zip(self.update_mlps, self.gates, self.norms):
            neigh = (h[idx] * weight.unsqueeze(-1)).sum(dim=1)
            lap_msg = neigh - h
            update = update_mlp(pt.cat([h, lap_msg], dim=-1))
            gate = gate_mlp(pt.cat([h, lap_msg], dim=-1))
            h = norm(h + self.beta * gate * update)
            diffusion_norms.append(lap_msg.norm(dim=-1).mean().detach())
        if diffusion_norms:
            self.last_diagnostics["internal_operator_diffusion_norm"] = pt.stack(diffusion_norms).mean()
        return h


class PairGaussianResidueToSurfaceCorrection(pt.nn.Module):
    """Cooperative residue-pair Gaussian correction to a single-residue surface field."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        pair_topk=8,
        sigma_t=8.0,
        sigma_n=4.0,
        sigma_mode="fixed",
        sigma_min=0.5,
        beta=0.05,
        mode="k_nearest",
        joint_mode="independent",
        fusion_mode="residual",
        content_mode="none",
        content_weight=1.0,
        rho_condition_mode="none",
        vector_condition_mode="none",
        single_compete_mode="none",
        single_rescue_beta=0.25,
        vector_dim=16,
        dropout=0.1,
    ):
        super().__init__()
        self.pair_topk = max(2, int(pair_topk))
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.sigma_mode = str(sigma_mode)
        if self.sigma_mode not in {"fixed", "learned", "reuse_single"}:
            raise ValueError("pair Gaussian sigma_mode must be one of: fixed, learned, reuse_single")
        self.sigma_min = float(sigma_min)
        self.beta = float(beta)
        self.mode = str(mode)
        if self.mode not in {"k_nearest", "random"}:
            raise ValueError("pair Gaussian mode must be one of: k_nearest, random")
        self.joint_mode = str(joint_mode)
        if self.joint_mode not in {"independent", "joint_residual", "joint_only"}:
            raise ValueError("pair Gaussian joint_mode must be one of: independent, joint_residual, joint_only")
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"residual", "coattn"}:
            raise ValueError("pair Gaussian fusion_mode must be one of: residual, coattn")
        self.content_mode = str(content_mode)
        if self.content_mode not in {"none", "additive", "gaussian_distance"}:
            raise ValueError("pair Gaussian content_mode must be one of: none, additive, gaussian_distance")
        self.content_weight = float(content_weight)
        self.rho_condition_mode = str(rho_condition_mode)
        if self.rho_condition_mode not in {"none", "surface_dot"}:
            raise ValueError("pair Gaussian rho_condition_mode must be one of: none, surface_dot")
        self.vector_condition_mode = str(vector_condition_mode)
        if self.vector_condition_mode not in {"none", "surface_frame"}:
            raise ValueError("pair Gaussian vector_condition_mode must be one of: none, surface_frame")
        self.single_compete_mode = str(single_compete_mode)
        if self.single_compete_mode not in {"none", "conservative_rescue", "strong_gate"}:
            raise ValueError("single_compete_mode must be one of: none, conservative_rescue, strong_gate")
        self.single_rescue_beta = float(single_rescue_beta)
        self.vector_dim = int(vector_dim)
        self._last_vector_condition_diagnostics = {}
        self.pair_geom_dim = 10 if self.sigma_mode in {"learned", "reuse_single"} else 6
        pair_input_dim = dim * 4 + self.pair_geom_dim
        self.vector_condition_proj = None
        if self.vector_condition_mode == "surface_frame":
            raw_vector_dim = 10 * self.vector_dim
            self.vector_condition_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(raw_vector_dim),
                pt.nn.Linear(raw_vector_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            pair_input_dim += dim
        self.pair_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(pair_input_dim),
            pt.nn.Linear(pair_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.sigma_head = None
        if self.sigma_mode == "learned":
            self.sigma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
            last = self.sigma_head[-1]
            pt.nn.init.zeros_(last.weight)
            base_sigma = pt.as_tensor([self.sigma_t, self.sigma_n], dtype=pt.float32)
            with pt.no_grad():
                last.bias.copy_(inverse_softplus(base_sigma - self.sigma_min))
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.rho_proj = None
        self.rho_surface_proj = None
        rho_input_dim = pair_input_dim
        if self.rho_condition_mode == "surface_dot":
            self.rho_surface_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + geom_dim),
                pt.nn.Linear(dim + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            rho_input_dim += dim + 3
        if self.joint_mode != "independent":
            self.rho_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(rho_input_dim),
                pt.nn.Linear(rho_input_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 3),
                pt.nn.Tanh(),
            )
        self.content_query = None
        self.content_key = None
        self.content_distance_log_weight = None
        if self.content_mode in {"additive", "gaussian_distance"}:
            self.content_query = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
            )
            self.content_key = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + geom_dim),
                pt.nn.Linear(dim + geom_dim, dim),
            )
            if self.content_mode == "gaussian_distance":
                self.content_distance_log_weight = pt.nn.Parameter(pt.zeros(dim))
        self.single_rescue_gate = None
        self.single_compete_gate = None
        if self.single_compete_mode == "conservative_rescue":
            self.single_rescue_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + geom_dim),
                pt.nn.Linear(dim * 2 + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
                pt.nn.Sigmoid(),
            )
            last = self.single_rescue_gate[-2]
            pt.nn.init.zeros_(last.weight)
            pt.nn.init.constant_(last.bias, -2.5)
        elif self.single_compete_mode == "strong_gate":
            self.single_compete_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + geom_dim),
                pt.nn.Linear(dim * 2 + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
        self.coattn_pair_proj = None
        self.coattn_channel = None
        if self.fusion_mode == "coattn":
            self.coattn_pair_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + geom_dim),
                pt.nn.Linear(dim + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.coattn_channel = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + geom_dim),
                pt.nn.Linear(dim * 2 + geom_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
        self.norm = pt.nn.LayerNorm(dim)

    def _pair_indices(self, dist, k):
        nearest = pt.topk(dist, k=k, dim=-1, largest=False).indices
        if self.mode == "random":
            n_res = dist.shape[1]
            offsets = pt.arange(k, device=dist.device).view(1, k)
            base = pt.argmin(dist, dim=-1, keepdim=True)
            nearest = (base + 13 * (offsets + 1) + 17 * pt.arange(dist.shape[0], device=dist.device).view(-1, 1)) % max(n_res, 1)
        pair_i, pair_j = pt.triu_indices(k, k, offset=1, device=dist.device)
        return nearest[:, pair_i], nearest[:, pair_j]

    def _pair_sigmas(self, h_res, idx_i, idx_j, sigma_res_override=None):
        if self.sigma_mode == "reuse_single" and sigma_res_override is not None:
            sigma_res = sigma_res_override.to(device=h_res.device, dtype=h_res.dtype).clamp_min(self.sigma_min)
            return sigma_res, sigma_res[idx_i].clamp_min(self.sigma_min), sigma_res[idx_j].clamp_min(self.sigma_min)
        if self.sigma_mode != "learned" or self.sigma_head is None:
            return None, None, None
        sigma_res = self.sigma_min + pt.nn.functional.softplus(self.sigma_head(h_res))
        return sigma_res, sigma_res[idx_i].clamp_min(self.sigma_min), sigma_res[idx_j].clamp_min(self.sigma_min)

    def _pair_vector_condition(self, residue_vectors, idx_i, idx_j, rel_i, rel_j, surface_normal, pair_vec, pair_dist):
        self._last_vector_condition_diagnostics = {}
        if self.vector_condition_proj is None or residue_vectors is None:
            return None
        if residue_vectors.ndim != 3 or residue_vectors.shape[1] != self.vector_dim or residue_vectors.shape[2] != 3:
            return None
        if idx_i.numel() and int(pt.maximum(idx_i.max(), idx_j.max()).detach().cpu()) >= residue_vectors.shape[0]:
            return None
        res_v = residue_vectors.to(device=rel_i.device, dtype=rel_i.dtype)
        vi = res_v[idx_i]
        vj = res_v[idx_j]
        normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = normal.unsqueeze(1)
        dir_i = rel_i / rel_i.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_j = rel_j / rel_j.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pair_axis = pair_vec / pair_dist.unsqueeze(-1).clamp_min(1e-6)

        def features(v, direction, axis):
            v_norm = v.norm(dim=-1)
            dot_n = (v * normal.unsqueeze(-2)).sum(dim=-1)
            dot_r = (v * direction.unsqueeze(-2)).sum(dim=-1)
            tangent_sq = (v_norm.square() - dot_n.square()).clamp_min(0.0)
            tangent = tangent_sq.sqrt()
            dot_axis = (v * axis.unsqueeze(-2)).sum(dim=-1)
            return pt.cat([v_norm, dot_n, dot_r, tangent, dot_axis], dim=-1), {
                "v_norm": v_norm,
                "dot_n": dot_n,
                "dot_r": dot_r,
                "tangent": tangent,
                "tangent_sq": tangent_sq,
                "dot_axis": dot_axis,
            }

        feat_i, diag_i = features(vi, dir_i, pair_axis)
        feat_j, diag_j = features(vj, dir_j, -pair_axis)
        raw = pt.cat([feat_i, feat_j], dim=-1)
        projected = self.vector_condition_proj(raw)

        tangent_sq = pt.cat([diag_i["tangent_sq"], diag_j["tangent_sq"]], dim=-1)
        tangent = pt.cat([diag_i["tangent"], diag_j["tangent"]], dim=-1)
        v_norm = pt.cat([diag_i["v_norm"], diag_j["v_norm"]], dim=-1)
        dot_n = pt.cat([diag_i["dot_n"], diag_j["dot_n"]], dim=-1)
        dot_r = pt.cat([diag_i["dot_r"], diag_j["dot_r"]], dim=-1)
        dot_axis = pt.cat([diag_i["dot_axis"], diag_j["dot_axis"]], dim=-1)
        tangent_sq_f = tangent_sq.detach().float()
        self._last_vector_condition_diagnostics = {
            "pair_gaussian_vector_raw_abs": raw.detach().float().abs().mean(),
            "pair_gaussian_vector_raw_std": raw.detach().float().std(unbiased=False),
            "pair_gaussian_vector_raw_max_abs": raw.detach().float().abs().max(),
            "pair_gaussian_vector_raw_finite_frac": pt.isfinite(raw.detach()).float().mean(),
            "pair_gaussian_vector_proj_abs": projected.detach().float().abs().mean(),
            "pair_gaussian_vector_proj_std": projected.detach().float().std(unbiased=False),
            "pair_gaussian_vector_proj_max_abs": projected.detach().float().abs().max(),
            "pair_gaussian_vector_proj_finite_frac": pt.isfinite(projected.detach()).float().mean(),
            "pair_gaussian_vector_v_norm_mean": v_norm.detach().float().mean(),
            "pair_gaussian_vector_v_norm_min": v_norm.detach().float().min(),
            "pair_gaussian_vector_v_norm_max": v_norm.detach().float().max(),
            "pair_gaussian_vector_dot_n_abs": dot_n.detach().float().abs().mean(),
            "pair_gaussian_vector_dot_n_max_abs": dot_n.detach().float().abs().max(),
            "pair_gaussian_vector_dot_r_abs": dot_r.detach().float().abs().mean(),
            "pair_gaussian_vector_dot_axis_abs": dot_axis.detach().float().abs().mean(),
            "pair_gaussian_vector_tangent_mean": tangent.detach().float().mean(),
            "pair_gaussian_vector_tangent_min": tangent.detach().float().min(),
            "pair_gaussian_vector_tangent_sq_min": tangent_sq_f.min(),
            "pair_gaussian_vector_tangent_sq_mean": tangent_sq_f.mean(),
            "pair_gaussian_vector_tangent_sq_zero_frac": (tangent_sq_f <= 0.0).float().mean(),
            "pair_gaussian_vector_tangent_sq_lt1em12_frac": (tangent_sq_f <= 1e-12).float().mean(),
            "pair_gaussian_vector_tangent_sq_lt1em8_frac": (tangent_sq_f <= 1e-8).float().mean(),
            "pair_gaussian_vector_tangent_grad_risk_max": (0.5 / tangent_sq_f.clamp_min(1e-30).sqrt()).max(),
        }
        return projected

    @staticmethod
    def _surface_frame(surface_normal):
        normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ref_x = normal.new_tensor([1.0, 0.0, 0.0]).view(1, 3).expand_as(normal)
        ref_y = normal.new_tensor([0.0, 1.0, 0.0]).view(1, 3).expand_as(normal)
        ref = pt.where(normal[:, :1].abs() < 0.9, ref_x, ref_y)
        tangent1 = pt.cross(normal, ref, dim=-1)
        tangent1 = tangent1 / tangent1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tangent2 = pt.cross(normal, tangent1, dim=-1)
        tangent2 = tangent2 / tangent2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return tangent1, tangent2, normal

    def _joint_scores(self, rel_i, rel_j, surface_normal, rho_input, legacy_score, sigma_i=None, sigma_j=None):
        if self.joint_mode == "independent" or self.rho_proj is None:
            return legacy_score, None, None, None
        t1, t2, normal = self._surface_frame(surface_normal)
        basis = [t1.unsqueeze(1), t2.unsqueeze(1), normal.unsqueeze(1)]
        xi = pt.stack([(rel_i * b).sum(dim=-1) for b in basis], dim=-1)
        xj = pt.stack([(rel_j * b).sum(dim=-1) for b in basis], dim=-1)
        if sigma_i is None or sigma_j is None:
            sigma = xi.new_tensor([self.sigma_t, self.sigma_t, self.sigma_n]).clamp_min(1e-6)
            xi = xi / sigma.view(1, 1, 3)
            xj = xj / sigma.view(1, 1, 3)
        else:
            sigma_i3 = pt.stack([sigma_i[..., 0], sigma_i[..., 0], sigma_i[..., 1]], dim=-1).clamp_min(1e-6)
            sigma_j3 = pt.stack([sigma_j[..., 0], sigma_j[..., 0], sigma_j[..., 1]], dim=-1).clamp_min(1e-6)
            xi = xi / sigma_i3
            xj = xj / sigma_j3
        rho = 0.85 * self.rho_proj(rho_input)
        rho2 = rho.square().clamp(max=0.95)
        denom = (1.0 - rho2).clamp_min(1e-4)
        independent_energy = 0.5 * (xi.square() + xj.square()).sum(dim=-1)
        joint_energy = 0.5 * ((xi.square() - 2.0 * rho * xi * xj + xj.square()) / denom).sum(dim=-1)
        log_det_penalty = 0.5 * denom.log().sum(dim=-1)
        independent_score = -independent_energy
        joint_score = -joint_energy - log_det_penalty
        coupling_gain = joint_score - independent_score
        return joint_score, independent_score, coupling_gain, rho

    def forward(
        self,
        h_res,
        residue_centers,
        surface_pos,
        surface_normal,
        surface_geom,
        h_surf_single,
        residue_vectors=None,
        pair_only=False,
        beta_override=None,
        pair_dist_scale=None,
        sigma_res_override=None,
        precomputed_dist=None,
        precomputed_pair_indices=None,
    ):
        if self.beta == 0.0 or h_res.shape[0] < 2:
            self.last_top_record = None
            return h_surf_single, {}
        beta_eff = self.beta if beta_override is None else beta_override
        k = min(self.pair_topk, residue_centers.shape[0])
        if precomputed_pair_indices is None:
            dist = precomputed_dist
            if dist is None:
                dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
            idx_i, idx_j = self._pair_indices(dist, k)
        else:
            idx_i, idx_j = precomputed_pair_indices
        sigma_res, sigma_i, sigma_j = self._pair_sigmas(h_res, idx_i, idx_j, sigma_res_override=sigma_res_override)
        rel_i = residue_centers[idx_i] - surface_pos.unsqueeze(1)
        rel_j = residue_centers[idx_j] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        dn_i = (rel_i * normal).sum(dim=-1).abs()
        dn_j = (rel_j * normal).sum(dim=-1).abs()
        dt_i = (rel_i.square().sum(dim=-1).clamp_min(0.0) - dn_i.square()).clamp_min(0.0).sqrt()
        dt_j = (rel_j.square().sum(dim=-1).clamp_min(0.0) - dn_j.square()).clamp_min(0.0).sqrt()
        pair_vec = residue_centers[idx_i] - residue_centers[idx_j]
        pair_dist = pair_vec.norm(dim=-1).clamp_min(1e-6)
        pair_mid = 0.5 * (residue_centers[idx_i] + residue_centers[idx_j])
        mid_rel = pair_mid - surface_pos.unsqueeze(1)
        mid_dn = (mid_rel * normal).sum(dim=-1).abs()
        mid_dt = (mid_rel.square().sum(dim=-1).clamp_min(0.0) - mid_dn.square()).clamp_min(0.0).sqrt()
        if sigma_i is None or sigma_j is None:
            sigma_i_t = dt_i.new_full(dt_i.shape, max(self.sigma_t, 1e-6))
            sigma_i_n = dn_i.new_full(dn_i.shape, max(self.sigma_n, 1e-6))
            sigma_j_t = dt_j.new_full(dt_j.shape, max(self.sigma_t, 1e-6))
            sigma_j_n = dn_j.new_full(dn_j.shape, max(self.sigma_n, 1e-6))
        else:
            sigma_i_t = sigma_i[..., 0].clamp_min(1e-6)
            sigma_i_n = sigma_i[..., 1].clamp_min(1e-6)
            sigma_j_t = sigma_j[..., 0].clamp_min(1e-6)
            sigma_j_n = sigma_j[..., 1].clamp_min(1e-6)
        sigma_pair_t = (sigma_i_t * sigma_j_t).clamp_min(1e-12).sqrt()
        sigma_pair_n = (sigma_i_n * sigma_j_n).clamp_min(1e-12).sqrt()
        score = -0.5 * (
            (dt_i / sigma_i_t).square()
            + (dn_i / sigma_i_n).square()
            + (dt_j / sigma_j_t).square()
            + (dn_j / sigma_j_n).square()
            + 0.25 * (mid_dt / sigma_pair_t).square()
            + 0.25 * (mid_dn / sigma_pair_n).square()
        )
        weight = pt.softmax(score, dim=-1)
        hi = h_res[idx_i]
        hj = h_res[idx_j]
        if pair_dist_scale is None:
            pair_dist_scale = pair_dist.mean().clamp_min(1e-6)
        else:
            pair_dist_scale = pair_dist_scale.to(device=pair_dist.device, dtype=pair_dist.dtype).clamp_min(1e-6)
        pair_geom_parts = [
            dt_i / sigma_i_t,
            dn_i / sigma_i_n,
            dt_j / sigma_j_t,
            dn_j / sigma_j_n,
            mid_dt / sigma_pair_t,
            pair_dist / pair_dist_scale,
        ]
        if sigma_i is not None and sigma_j is not None:
            base_t = max(self.sigma_t, 1e-6)
            base_n = max(self.sigma_n, 1e-6)
            pair_geom_parts.extend(
                [
                    (sigma_i_t / base_t).clamp_min(1e-6).log(),
                    (sigma_i_n / base_n).clamp_min(1e-6).log(),
                    (sigma_j_t / base_t).clamp_min(1e-6).log(),
                    (sigma_j_n / base_n).clamp_min(1e-6).log(),
                ]
            )
        pair_geom = pt.stack(
            pair_geom_parts,
            dim=-1,
        )
        pair_parts = [hi, hj, hi * hj, (hi - hj).abs(), pair_geom]
        vector_condition = self._pair_vector_condition(
            residue_vectors,
            idx_i,
            idx_j,
            rel_i,
            rel_j,
            surface_normal,
            pair_vec,
            pair_dist,
        )
        if vector_condition is not None:
            pair_parts.append(vector_condition)
        pair_input = pt.cat(pair_parts, dim=-1)
        pair_feat = self.pair_proj(pair_input)
        rho_input = pair_input
        rho_dot_i = None
        rho_dot_j = None
        rho_dot_pair = None
        if self.rho_condition_mode == "surface_dot" and self.rho_surface_proj is not None:
            surface_rho = self.rho_surface_proj(pt.cat([h_surf_single, surface_geom], dim=-1))
            surface_rho_pair = surface_rho.unsqueeze(1).expand_as(hi)
            dot_scale = math.sqrt(max(hi.shape[-1], 1))
            rho_dot_i = (hi * surface_rho_pair).sum(dim=-1, keepdim=True) / dot_scale
            rho_dot_j = (hj * surface_rho_pair).sum(dim=-1, keepdim=True) / dot_scale
            rho_dot_pair = ((hi * hj) * surface_rho_pair).sum(dim=-1, keepdim=True) / dot_scale
            rho_input = pt.cat(
                [
                    pair_input,
                    surface_rho_pair,
                    rho_dot_i,
                    rho_dot_j,
                    rho_dot_pair,
                ],
                dim=-1,
            )
        content_score = None
        if self.content_query is not None and self.content_key is not None and self.content_weight != 0.0:
            pair_q = self.content_query(pair_feat)
            surf_k = self.content_key(pt.cat([h_surf_single, surface_geom], dim=-1))
            if self.content_mode == "gaussian_distance":
                diff = pair_q - surf_k.unsqueeze(1)
                weight_content = pt.nn.functional.softplus(self.content_distance_log_weight).to(device=diff.device, dtype=diff.dtype)
                content_score = -0.5 * (weight_content.view(1, 1, -1) * diff.square()).sum(dim=-1)
                content_score = content_score / math.sqrt(max(pair_q.shape[-1], 1))
            else:
                content_score = (pair_q * surf_k.unsqueeze(1)).sum(dim=-1) / math.sqrt(max(pair_q.shape[-1], 1))
            content_score = self.content_weight * content_score
        joint_score, independent_score, coupling_gain, rho = self._joint_scores(
            rel_i,
            rel_j,
            surface_normal,
            rho_input,
            score,
            sigma_i=sigma_i,
            sigma_j=sigma_j,
        )
        if self.joint_mode == "independent":
            route_score = score if content_score is None else score + content_score
            weight = pt.softmax(route_score, dim=-1)
            pair_ctx = (pair_feat * weight.unsqueeze(-1)).sum(dim=1)
        elif self.joint_mode == "joint_residual":
            route_score = joint_score if content_score is None else joint_score + content_score
            independent_route_score = independent_score if content_score is None else independent_score + content_score
            weight = pt.softmax(route_score, dim=-1)
            independent_weight = pt.softmax(independent_route_score, dim=-1)
            signed_weight = weight - independent_weight
            pair_ctx = (pair_feat * signed_weight.unsqueeze(-1)).sum(dim=1)
        else:
            route_score = joint_score if content_score is None else joint_score + content_score
            weight = pt.softmax(route_score, dim=-1)
            pair_ctx = (pair_feat * weight.unsqueeze(-1)).sum(dim=1)
        gate = self.gate(pt.cat([pair_ctx, surface_geom], dim=-1))
        base = h_surf_single.new_zeros(h_surf_single.shape) if pair_only else h_surf_single
        if self.fusion_mode == "coattn":
            pair_msg = self.coattn_pair_proj(pt.cat([pair_ctx, surface_geom], dim=-1))
            channel = pt.softmax(self.coattn_channel(pt.cat([base, pair_msg, surface_geom], dim=-1)), dim=-1)
            out = self.norm(channel[:, :1] * base + channel[:, 1:] * pair_msg)
        else:
            channel = None
            pair_msg = beta_eff * gate * pair_ctx
            if pair_only and self.single_compete_mode == "conservative_rescue" and self.single_rescue_gate is not None:
                single_gate = self.single_rescue_gate(pt.cat([h_surf_single, pair_msg, surface_geom], dim=-1))
                out = self.norm(pair_msg + self.single_rescue_beta * single_gate * (h_surf_single - pair_msg))
            elif pair_only and self.single_compete_mode == "strong_gate" and self.single_compete_gate is not None:
                compete = pt.softmax(self.single_compete_gate(pt.cat([h_surf_single, pair_msg, surface_geom], dim=-1)), dim=-1)
                out = self.norm(compete[:, :1] * h_surf_single + compete[:, 1:] * pair_msg)
            else:
                out = self.norm(base + pair_msg)
        entropy = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_top_record = None
        if weight.numel() > 0:
            with pt.no_grad():
                w_det = weight.detach().float()
                flat_idx = int(w_det.reshape(-1).argmax().cpu().item())
                pair_count = int(w_det.shape[1]) if w_det.ndim == 2 else 1
                surf_idx = flat_idx // max(pair_count, 1)
                pair_rank = flat_idx % max(pair_count, 1)
                record = {
                    "module": "pair_gaussian",
                    "surface_index": int(surf_idx),
                    "pair_rank": int(pair_rank),
                    "weight": float(w_det[surf_idx, pair_rank].cpu().item()),
                    "residue_i": int(idx_i.detach()[surf_idx, pair_rank].cpu().item()),
                    "residue_j": int(idx_j.detach()[surf_idx, pair_rank].cpu().item()),
                    "route_score": float(route_score.detach().float()[surf_idx, pair_rank].cpu().item()),
                    "base_score": float(score.detach().float()[surf_idx, pair_rank].cpu().item()),
                    "dt_i": float(dt_i.detach().float()[surf_idx, pair_rank].cpu().item()),
                    "dn_i": float(dn_i.detach().float()[surf_idx, pair_rank].cpu().item()),
                    "dt_j": float(dt_j.detach().float()[surf_idx, pair_rank].cpu().item()),
                    "dn_j": float(dn_j.detach().float()[surf_idx, pair_rank].cpu().item()),
                    "pair_dist": float(pair_dist.detach().float()[surf_idx, pair_rank].cpu().item()),
                }
                if independent_score is not None:
                    record["independent_score"] = float(independent_score.detach().float()[surf_idx, pair_rank].cpu().item())
                if coupling_gain is not None:
                    record["coupling_gain"] = float(coupling_gain.detach().float()[surf_idx, pair_rank].cpu().item())
                if rho is not None:
                    record["rho_abs_mean_for_pair"] = float(rho.detach().float()[surf_idx, pair_rank].abs().mean().cpu().item())
                self.last_top_record = record
        diag = {
            "pair_gaussian_entropy": entropy.detach(),
            "pair_gaussian_weight_max": weight.max(dim=-1).values.mean().detach(),
            "pair_gaussian_beta_eff": pt.as_tensor(beta_eff, device=h_res.device, dtype=h_res.dtype).detach(),
        }
        if sigma_res is not None:
            diag.update(
                {
                    "pair_gaussian_sigma_tangent_mean": sigma_res[:, 0].mean().detach(),
                    "pair_gaussian_sigma_tangent_min": sigma_res[:, 0].min().detach(),
                    "pair_gaussian_sigma_tangent_max": sigma_res[:, 0].max().detach(),
                    "pair_gaussian_sigma_tangent_std": sigma_res[:, 0].std(unbiased=False).detach(),
                    "pair_gaussian_sigma_normal_mean": sigma_res[:, 1].mean().detach(),
                    "pair_gaussian_sigma_normal_min": sigma_res[:, 1].min().detach(),
                    "pair_gaussian_sigma_normal_max": sigma_res[:, 1].max().detach(),
                    "pair_gaussian_sigma_normal_std": sigma_res[:, 1].std(unbiased=False).detach(),
                }
            )
        if channel is not None:
            diag["pair_gaussian_coattn_pair_weight"] = channel[:, 1].mean().detach()
            diag["pair_gaussian_coattn_single_weight"] = channel[:, 0].mean().detach()
        if rho is not None:
            diag["pair_gaussian_joint_rho_abs"] = rho.abs().mean().detach()
        if vector_condition is not None:
            diag["pair_gaussian_vector_condition_abs"] = vector_condition.abs().mean().detach()
            diag["pair_gaussian_vector_condition_std"] = vector_condition.std(unbiased=False).detach()
            diag.update(getattr(self, "_last_vector_condition_diagnostics", {}))
        if rho_dot_i is not None:
            diag["pair_gaussian_rho_dot_i_abs"] = rho_dot_i.abs().mean().detach()
            diag["pair_gaussian_rho_dot_j_abs"] = rho_dot_j.abs().mean().detach()
            diag["pair_gaussian_rho_dot_pair_abs"] = rho_dot_pair.abs().mean().detach()
        if coupling_gain is not None:
            diag["pair_gaussian_coupling_abs"] = coupling_gain.abs().mean().detach()
            diag["pair_gaussian_coupling_mean"] = coupling_gain.mean().detach()
        if content_score is not None:
            diag["pair_gaussian_content_score_mean"] = content_score.mean().detach()
            diag["pair_gaussian_content_score_std"] = content_score.std(unbiased=False).detach()
            diag["pair_gaussian_content_score_abs"] = content_score.abs().mean().detach()
            if self.content_mode == "gaussian_distance" and self.content_distance_log_weight is not None:
                diag["pair_gaussian_content_distance_weight_mean"] = pt.nn.functional.softplus(
                    self.content_distance_log_weight
                ).mean().detach()
        if self.single_compete_mode == "conservative_rescue" and pair_only and self.single_rescue_gate is not None:
            diag["pair_gaussian_single_rescue_gate_mean"] = single_gate.mean().detach()
            diag["pair_gaussian_single_rescue_beta"] = h_res.new_tensor(self.single_rescue_beta).detach()
        if self.single_compete_mode == "strong_gate" and pair_only and self.single_compete_gate is not None:
            diag["pair_gaussian_strong_gate_single_mean"] = compete[:, 0].mean().detach()
            diag["pair_gaussian_strong_gate_pair_mean"] = compete[:, 1].mean().detach()
        return out, diag


class TripletGaussianResidueToSurfaceCorrection(pt.nn.Module):
    """Three-residue joint Gaussian correction to the residue-to-surface field."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        triplet_topk=8,
        sigma_t=8.0,
        sigma_n=4.0,
        beta=0.03,
        mode="k_nearest",
        dropout=0.1,
        score_temperature=1.0,
        score_clamp=0.0,
        preselect_topm=0,
    ):
        super().__init__()
        self.triplet_topk = max(3, int(triplet_topk))
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.beta = float(beta)
        self.score_temperature = max(float(score_temperature), 1.0e-6)
        self.score_clamp = max(float(score_clamp), 0.0)
        self.preselect_topm = max(0, int(preselect_topm))
        self.mode = str(mode)
        if self.mode not in {"k_nearest", "random"}:
            raise ValueError("triplet Gaussian mode must be one of: k_nearest, random")
        self.triplet_geom_dim = 12
        triplet_input_dim = dim * 9 + self.triplet_geom_dim
        self.triplet_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(triplet_input_dim),
            pt.nn.Linear(triplet_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.rho_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(triplet_input_dim),
            pt.nn.Linear(triplet_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 9),
        )
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim),
            pt.nn.Linear(dim + geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _triplet_indices(self, dist, k):
        nearest = pt.topk(dist, k=k, dim=-1, largest=False).indices
        if self.mode == "random":
            n_res = dist.shape[1]
            offsets = pt.arange(k, device=dist.device).view(1, k)
            base = pt.argmin(dist, dim=-1, keepdim=True)
            nearest = (base + 11 * (offsets + 1) + 19 * pt.arange(dist.shape[0], device=dist.device).view(-1, 1)) % max(n_res, 1)
        combos = pt.combinations(pt.arange(k, device=dist.device), r=3)
        return nearest[:, combos[:, 0]], nearest[:, combos[:, 1]], nearest[:, combos[:, 2]]

    @staticmethod
    def _stable_triplet_corr(raw):
        raw = 0.85 * pt.tanh(raw).view(*raw.shape[:-1], 3, 3)
        r12 = raw[..., 0]
        r13 = raw[..., 1]
        partial23 = raw[..., 2]
        r23 = partial23 * ((1.0 - r12.square()).clamp_min(1e-4) * (1.0 - r13.square()).clamp_min(1e-4)).sqrt() + r12 * r13
        eye = pt.eye(3, device=raw.device, dtype=raw.dtype).view(*([1] * (raw.ndim - 1)), 3, 3)
        corr = eye.expand(*raw.shape[:-1], 3, 3).clone()
        corr[..., 0, 1] = corr[..., 1, 0] = r12
        corr[..., 0, 2] = corr[..., 2, 0] = r13
        corr[..., 1, 2] = corr[..., 2, 1] = r23
        corr = corr + 1e-4 * eye
        return corr, pt.stack([r12, r13, r23], dim=-1)

    @staticmethod
    def _joint_triplet_score(rel_i, rel_j, rel_k, surface_normal, rho_raw, sigma_t, sigma_n):
        # cuSOLVER Cholesky is not implemented for fp16; keep this tiny
        # covariance solve in fp32 under AMP and cast scores back afterwards.
        out_dtype = rel_i.dtype
        rel_i_f = rel_i.float()
        rel_j_f = rel_j.float()
        rel_k_f = rel_k.float()
        surface_normal_f = surface_normal.float()
        rho_raw_f = rho_raw.float()
        t1, t2, normal = PairGaussianResidueToSurfaceCorrection._surface_frame(surface_normal_f)
        basis = [t1.unsqueeze(1), t2.unsqueeze(1), normal.unsqueeze(1)]
        coords = []
        for rel in (rel_i_f, rel_j_f, rel_k_f):
            coords.append(pt.stack([(rel * b).sum(dim=-1) for b in basis], dim=-1))
        sigma = rel_i_f.new_tensor([max(sigma_t, 1e-6), max(sigma_t, 1e-6), max(sigma_n, 1e-6)])
        xi, xj, xk = [c / sigma.view(1, 1, 3) for c in coords]
        z = pt.stack([xi, xj, xk], dim=-1)
        corr, rho = TripletGaussianResidueToSurfaceCorrection._stable_triplet_corr(rho_raw_f)
        chol = pt.linalg.cholesky(corr)
        solved = pt.linalg.solve_triangular(chol, z.unsqueeze(-1), upper=False).squeeze(-1)
        joint_energy = 0.5 * solved.square().sum(dim=(-1, -2))
        log_det_penalty = chol.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).log().sum(dim=(-1, -2))
        joint_score = -joint_energy - log_det_penalty
        independent_score = -0.5 * z.square().sum(dim=(-1, -2))
        coupling_gain = joint_score - independent_score
        return joint_score.to(out_dtype), independent_score.to(out_dtype), coupling_gain.to(out_dtype), rho.to(out_dtype)

    def forward(self, h_res, residue_centers, surface_pos, surface_normal, surface_geom, h_surf, beta_override=None):
        if self.beta == 0.0 or h_res.shape[0] < 3:
            self.last_top_record = None
            return h_surf, {}
        beta_eff = self.beta if beta_override is None else beta_override
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.triplet_topk, residue_centers.shape[0])
        idx_i, idx_j, idx_k = self._triplet_indices(dist, k)
        rel_i = residue_centers[idx_i] - surface_pos.unsqueeze(1)
        rel_j = residue_centers[idx_j] - surface_pos.unsqueeze(1)
        rel_k = residue_centers[idx_k] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)

        def split_dist(rel):
            dn = (rel * normal).sum(dim=-1).abs()
            dt = (rel.square().sum(dim=-1).clamp_min(0.0) - dn.square()).clamp_min(0.0).sqrt()
            return dt, dn

        dt_i, dn_i = split_dist(rel_i)
        dt_j, dn_j = split_dist(rel_j)
        dt_k, dn_k = split_dist(rel_k)
        centroid = (residue_centers[idx_i] + residue_centers[idx_j] + residue_centers[idx_k]) / 3.0
        centroid_dt, centroid_dn = split_dist(centroid - surface_pos.unsqueeze(1))
        dij = (residue_centers[idx_i] - residue_centers[idx_j]).norm(dim=-1)
        dik = (residue_centers[idx_i] - residue_centers[idx_k]).norm(dim=-1)
        djk = (residue_centers[idx_j] - residue_centers[idx_k]).norm(dim=-1)
        scale = pt.stack([dij, dik, djk], dim=-1).mean().clamp_min(1e-6)
        sigma_t = max(self.sigma_t, 1e-6)
        sigma_n = max(self.sigma_n, 1e-6)
        geom = pt.stack(
            [
                dt_i / sigma_t,
                dn_i / sigma_n,
                dt_j / sigma_t,
                dn_j / sigma_n,
                dt_k / sigma_t,
                dn_k / sigma_n,
                centroid_dt / sigma_t,
                centroid_dn / sigma_n,
                dij / scale,
                dik / scale,
                djk / scale,
                (dij + dik + djk) / (3.0 * scale),
            ],
            dim=-1,
        )
        raw_triplet_count = idx_i.shape[1]
        if self.preselect_topm > 0 and raw_triplet_count > self.preselect_topm:
            v_ij = residue_centers[idx_j] - residue_centers[idx_i]
            v_ik = residue_centers[idx_k] - residue_centers[idx_i]
            triangle_area = 0.5 * pt.linalg.cross(v_ij.float(), v_ik.float(), dim=-1).norm(dim=-1).to(dtype=geom.dtype)
            min_pair_dist = pt.stack([dij, dik, djk], dim=-1).min(dim=-1).values
            centroid_prior = -0.5 * ((centroid_dt / sigma_t) ** 2 + (centroid_dn / sigma_n) ** 2)
            area_term = pt.log1p(triangle_area / scale.square().clamp_min(1e-6))
            spread_term = pt.log1p(min_pair_dist / scale.clamp_min(1e-6))
            pre_score = centroid_prior + 0.10 * area_term + 0.05 * spread_term
            keep = min(self.preselect_topm, raw_triplet_count)
            keep_idx = pt.topk(pre_score, k=keep, dim=-1, largest=True).indices

            def gather_triplet(values):
                if values.ndim == 2:
                    return values.gather(1, keep_idx)
                expand_idx = keep_idx.view(*keep_idx.shape, *([1] * (values.ndim - 2))).expand(-1, -1, *values.shape[2:])
                return values.gather(1, expand_idx)

            idx_i = gather_triplet(idx_i)
            idx_j = gather_triplet(idx_j)
            idx_k = gather_triplet(idx_k)
            rel_i = gather_triplet(rel_i)
            rel_j = gather_triplet(rel_j)
            rel_k = gather_triplet(rel_k)
            dt_i = gather_triplet(dt_i)
            dn_i = gather_triplet(dn_i)
            dt_j = gather_triplet(dt_j)
            dn_j = gather_triplet(dn_j)
            dt_k = gather_triplet(dt_k)
            dn_k = gather_triplet(dn_k)
            centroid_dt = gather_triplet(centroid_dt)
            centroid_dn = gather_triplet(centroid_dn)
            dij = gather_triplet(dij)
            dik = gather_triplet(dik)
            djk = gather_triplet(djk)
            geom = gather_triplet(geom)
        hi = h_res[idx_i]
        hj = h_res[idx_j]
        hk = h_res[idx_k]
        triplet_input = pt.cat(
            [hi, hj, hk, hi * hj, hi * hk, hj * hk, (hi - hj).abs(), (hi - hk).abs(), (hj - hk).abs(), geom],
            dim=-1,
        )
        triplet_feat = self.triplet_proj(triplet_input)
        rho_raw = self.rho_proj(triplet_input)
        score, independent_score, coupling_gain, rho = self._joint_triplet_score(rel_i, rel_j, rel_k, surface_normal, rho_raw, sigma_t, sigma_n)
        score_for_weight = score
        if self.score_clamp > 0.0:
            clamp = float(self.score_clamp)
            score_for_weight = score_for_weight.clamp(min=-clamp, max=clamp)
        weight = pt.softmax(score_for_weight / self.score_temperature, dim=-1)
        triplet_ctx = (triplet_feat * weight.unsqueeze(-1)).sum(dim=1)
        gate = self.gate(pt.cat([triplet_ctx, surface_geom], dim=-1))
        out = self.norm(h_surf + beta_eff * gate * triplet_ctx)
        entropy = -(weight.clamp_min(1e-8) * weight.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_top_record = None
        if weight.numel() > 0:
            with pt.no_grad():
                w_det = weight.detach().float()
                flat_idx = int(w_det.reshape(-1).argmax().cpu().item())
                triplet_count = int(w_det.shape[1]) if w_det.ndim == 2 else 1
                surf_idx = flat_idx // max(triplet_count, 1)
                triplet_rank = flat_idx % max(triplet_count, 1)
                self.last_top_record = {
                    "module": "triplet_gaussian",
                    "surface_index": int(surf_idx),
                    "triplet_rank": int(triplet_rank),
                    "weight": float(w_det[surf_idx, triplet_rank].cpu().item()),
                    "residue_i": int(idx_i.detach()[surf_idx, triplet_rank].cpu().item()),
                    "residue_j": int(idx_j.detach()[surf_idx, triplet_rank].cpu().item()),
                    "residue_k": int(idx_k.detach()[surf_idx, triplet_rank].cpu().item()),
                    "score_for_weight": float(score_for_weight.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "joint_score": float(score.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "independent_score": float(independent_score.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "coupling_gain": float(coupling_gain.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "rho_abs_mean_for_triplet": float(rho.detach().float()[surf_idx, triplet_rank].abs().mean().cpu().item()),
                    "dt_i": float(dt_i.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dn_i": float(dn_i.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dt_j": float(dt_j.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dn_j": float(dn_j.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dt_k": float(dt_k.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dn_k": float(dn_k.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dij": float(dij.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "dik": float(dik.detach().float()[surf_idx, triplet_rank].cpu().item()),
                    "djk": float(djk.detach().float()[surf_idx, triplet_rank].cpu().item()),
                }
        diag = {
            "triplet_gaussian_entropy": entropy.detach(),
            "triplet_gaussian_weight_max": weight.max(dim=-1).values.mean().detach(),
            "triplet_gaussian_beta_eff": pt.as_tensor(beta_eff, device=h_res.device, dtype=h_res.dtype).detach(),
            "triplet_gaussian_rho_abs": rho.abs().mean().detach(),
            "triplet_gaussian_coupling_abs": coupling_gain.abs().mean().detach(),
            "triplet_gaussian_coupling_mean": coupling_gain.mean().detach(),
            "triplet_gaussian_independent_score_mean": independent_score.mean().detach(),
            "triplet_gaussian_joint_score_mean": score.mean().detach(),
            "triplet_gaussian_score_for_weight_mean": score_for_weight.mean().detach(),
            "triplet_gaussian_score_temperature": h_res.new_tensor(float(self.score_temperature)).detach(),
            "triplet_gaussian_score_clamp": h_res.new_tensor(float(self.score_clamp)).detach(),
            "triplet_gaussian_gate_mean": gate.mean().detach(),
            "triplet_gaussian_update_norm": triplet_ctx.norm(dim=-1).mean().detach(),
            "triplet_gaussian_triplet_count": h_res.new_tensor(float(idx_i.shape[1])).detach(),
            "triplet_gaussian_raw_triplet_count": h_res.new_tensor(float(raw_triplet_count)).detach(),
            "triplet_gaussian_preselect_topm": h_res.new_tensor(float(self.preselect_topm)).detach(),
        }
        return out, diag


class GaussianResidueToSurfaceProjector(pt.nn.Module):
    def __init__(self, dim=128, geom_dim=6, residue_topk=8, sigma_t=6.0, sigma_n=3.0, dropout=0.1):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.res_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _weights(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        score = -0.5 * ((d_tangent / self.sigma_t) ** 2 + (d_normal / self.sigma_n) ** 2)
        weight = pt.softmax(score, dim=-1)
        return idx, weight, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, weight, _d_tangent, _d_normal = self._weights(surface_pos, surface_normal, residue_centers)
        h_context = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.res_proj(h_context) + self.geom_proj(surface_geom))
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


def parse_sigma_pairs(text_or_pairs):
    if isinstance(text_or_pairs, str):
        pairs = []
        for part in text_or_pairs.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                a, b = part.split(":", 1)
            elif "/" in part:
                a, b = part.split("/", 1)
            else:
                value = float(part)
                a, b = value, max(value * 0.5, 1e-3)
            pairs.append((float(a), float(b)))
        return tuple(pairs)
    return tuple((float(a), float(b)) for a, b in text_or_pairs)


def inverse_softplus(value):
    value = pt.as_tensor(value, dtype=pt.float32).clamp_min(1e-6)
    return pt.log(pt.expm1(value).clamp_min(1e-6))


class MultiScaleGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features onto surface points with several 2D Gaussian kernels.

    The old projector uses one tangent/normal bandwidth. This version keeps the
    same input/output contract but lets each surface point mix short-, mid-, and
    long-range residue context. It is still a representation change only; no
    additional loss is introduced.
    """

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((4.0, 2.0), (8.0, 4.0), (14.0, 7.0)),
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one scale.")
        self.context_proj = pt.nn.ModuleList(
            [pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim)) for _ in self.sigma_pairs]
        )
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.scale_router = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim + len(self.sigma_pairs) * 2),
            pt.nn.Linear(geom_dim + len(self.sigma_pairs) * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, len(self.sigma_pairs)),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        self.last_surface_residue_dist = dist
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        contexts = []
        stats = []
        for proj, (sigma_t, sigma_n) in zip(self.context_proj, self.sigma_pairs):
            score = -0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2)
            weight = pt.softmax(score, dim=-1)
            ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
            contexts.append(proj(ctx))
            stats.append(pt.stack([weight.max(dim=-1).values, weight.square().sum(dim=-1)], dim=-1))
        router_in = pt.cat([surface_geom, *stats], dim=-1)
        scale_weight = pt.softmax(self.scale_router(router_in), dim=-1)
        stacked = pt.stack(contexts, dim=1)
        h_context = (stacked * scale_weight.unsqueeze(-1)).sum(dim=1)
        # Return the mid-scale weights for diagnostics/backward-compatible callers.
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        sigma_t, sigma_n = self.sigma_pairs[mid]
        score = -0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2)
        weight = pt.softmax(score, dim=-1)
        h_surf0 = self.norm(h_context + self.geom_proj(surface_geom))
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class GlobalWeightedMultiScaleGaussianResidueToSurfaceProjector(MultiScaleGaussianResidueToSurfaceProjector):
    """Multiscale Gaussian projection with one learnable global scale mixture.

    This keeps the same sigma bank as the fixed multiscale projector, but removes
    per-surface-point routing so the only new degree of freedom is a global
    softmax over Gaussian scales.
    """

    def __init__(self, *args, init_logits=None, **kwargs):
        super().__init__(*args, **kwargs)
        if init_logits is None:
            init_logits = pt.zeros(len(self.sigma_pairs), dtype=pt.float32)
        self.global_scale_logits = pt.nn.Parameter(pt.as_tensor(init_logits, dtype=pt.float32).clone())

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        contexts = []
        mid_weight = None
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        for scale_idx, (proj, (sigma_t, sigma_n)) in enumerate(zip(self.context_proj, self.sigma_pairs)):
            score = -0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2)
            weight = pt.softmax(score, dim=-1)
            if scale_idx == mid:
                mid_weight = weight
            ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
            contexts.append(proj(ctx))
        scale_weight = pt.softmax(self.global_scale_logits.to(device=h_res.device, dtype=h_res.dtype), dim=-1)
        stacked = pt.stack(contexts, dim=0)
        h_context = (stacked * scale_weight.view(-1, 1, 1)).sum(dim=0)
        h_surf0 = self.norm(h_context + self.geom_proj(surface_geom))
        entropy = -(scale_weight * scale_weight.clamp_min(1e-8).log()).sum()
        self.last_diagnostics = {
            "sigma_global_weight_entropy": entropy.detach(),
            "sigma_global_weight_max": scale_weight.max().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, mid_weight


class LearnableScaleGaussianResidueToSurfaceProjector(GlobalWeightedMultiScaleGaussianResidueToSurfaceProjector):
    """Multiscale projection where the Gaussian bandwidths themselves are learned.

    Unlike ``multiscale_global``, this does not only learn a softmax over a fixed
    sigma bank. It initializes from the same bank, then optimizes the tangent and
    normal bandwidth values directly with a positive softplus parameterization.
    """

    def __init__(self, *args, sigma_min=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_min = float(sigma_min)
        init = pt.as_tensor(self.sigma_pairs, dtype=pt.float32)
        self.learned_sigma_raw = pt.nn.Parameter(inverse_softplus(init - self.sigma_min))

    def _learned_sigmas(self, device, dtype):
        return self.sigma_min + pt.nn.functional.softplus(self.learned_sigma_raw.to(device=device, dtype=dtype))

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        sigmas = self._learned_sigmas(device=h_res.device, dtype=h_res.dtype)
        contexts = []
        mid_weight = None
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        for scale_idx, proj in enumerate(self.context_proj):
            sigma_t = sigmas[scale_idx, 0].clamp_min(self.sigma_min)
            sigma_n = sigmas[scale_idx, 1].clamp_min(self.sigma_min)
            score = -0.5 * ((d_tangent / sigma_t) ** 2 + (d_normal / sigma_n) ** 2)
            weight = pt.softmax(score, dim=-1)
            if scale_idx == mid:
                mid_weight = weight
            ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
            contexts.append(proj(ctx))
        scale_weight = pt.softmax(self.global_scale_logits.to(device=h_res.device, dtype=h_res.dtype), dim=-1)
        stacked = pt.stack(contexts, dim=0)
        h_context = (stacked * scale_weight.view(-1, 1, 1)).sum(dim=0)
        h_surf0 = self.norm(h_context + self.geom_proj(surface_geom))
        entropy = -(scale_weight * scale_weight.clamp_min(1e-8).log()).sum()
        self.last_diagnostics = {
            "sigma_global_weight_entropy": entropy.detach(),
            "sigma_global_weight_max": scale_weight.max().detach(),
            "sigma_tangent_mean": sigmas[:, 0].mean().detach(),
            "sigma_tangent_min": sigmas[:, 0].min().detach(),
            "sigma_tangent_max": sigmas[:, 0].max().detach(),
            "sigma_normal_mean": sigmas[:, 1].mean().detach(),
            "sigma_normal_min": sigmas[:, 1].min().detach(),
            "sigma_normal_max": sigmas[:, 1].max().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, mid_weight


class PerResidueAdaptiveGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features with residue-conditioned Gaussian bandwidths."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((4.0, 2.0), (8.0, 4.0), (14.0, 7.0)),
        sigma_min=0.5,
        learn_amplitude=False,
        amplitude_min=0.25,
        amplitude_max=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one scale.")
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        base_sigma = pt.as_tensor(self.sigma_pairs[mid], dtype=pt.float32)
        self.sigma_min = float(sigma_min)
        self.learn_amplitude = bool(learn_amplitude)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        if self.amplitude_max <= self.amplitude_min:
            raise ValueError("amplitude_max must be greater than amplitude_min")
        self.context_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.sigma_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 2),
        )
        last = self.sigma_head[-1]
        pt.nn.init.zeros_(last.weight)
        with pt.no_grad():
            last.bias.copy_(inverse_softplus(base_sigma - self.sigma_min))
        self.amplitude_head = None
        if self.learn_amplitude:
            self.amplitude_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            amp_last = self.amplitude_head[-1]
            pt.nn.init.zeros_(amp_last.weight)
            target = (1.0 - self.amplitude_min) / max(self.amplitude_max - self.amplitude_min, 1e-6)
            target = min(max(target, 1e-4), 1.0 - 1e-4)
            with pt.no_grad():
                amp_last.bias.fill_(math.log(target / (1.0 - target)))
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        sigma_res = self.sigma_min + pt.nn.functional.softplus(self.sigma_head(h_res))
        sigma_pair = sigma_res[idx].clamp_min(self.sigma_min)
        score = -0.5 * ((d_tangent / sigma_pair[..., 0]) ** 2 + (d_normal / sigma_pair[..., 1]) ** 2)
        amplitude = None
        if self.amplitude_head is not None:
            amp_raw = self.amplitude_head(h_res).squeeze(-1)
            amplitude = self.amplitude_min + (self.amplitude_max - self.amplitude_min) * pt.sigmoid(amp_raw)
            score = score + amplitude[idx].clamp_min(1e-6).log()
        weight = pt.softmax(score, dim=-1)
        ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.context_proj(ctx) + self.geom_proj(surface_geom))
        self.last_diagnostics = {
            "sigma_tangent_mean": sigma_res[:, 0].mean().detach(),
            "sigma_tangent_min": sigma_res[:, 0].min().detach(),
            "sigma_tangent_max": sigma_res[:, 0].max().detach(),
            "sigma_tangent_std": sigma_res[:, 0].std(unbiased=False).detach(),
            "sigma_normal_mean": sigma_res[:, 1].mean().detach(),
            "sigma_normal_min": sigma_res[:, 1].min().detach(),
            "sigma_normal_max": sigma_res[:, 1].max().detach(),
            "sigma_normal_std": sigma_res[:, 1].std(unbiased=False).detach(),
        }
        if amplitude is not None:
            self.last_diagnostics.update(
                {
                    "residue_amplitude_mean": amplitude.mean().detach(),
                    "residue_amplitude_min": amplitude.min().detach(),
                    "residue_amplitude_max": amplitude.max().detach(),
                    "residue_amplitude_std": amplitude.std(unbiased=False).detach(),
                }
            )
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class FullLearnableCovarianceGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features with residue-conditioned amplitude and local 3D covariance."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((4.0, 2.0), (8.0, 4.0), (14.0, 7.0)),
        sigma_min=0.5,
        amplitude_min=0.25,
        amplitude_max=2.0,
        offdiag_scale=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one scale.")
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        base_t, base_n = self.sigma_pairs[mid]
        self.sigma_min = float(sigma_min)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self.offdiag_scale = float(offdiag_scale)
        if self.amplitude_max <= self.amplitude_min:
            raise ValueError("amplitude_max must be greater than amplitude_min")
        self.context_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.gaussian_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 7),
        )
        last = self.gaussian_head[-1]
        pt.nn.init.zeros_(last.weight)
        with pt.no_grad():
            base_diag = pt.as_tensor([base_t, base_t, base_n], dtype=pt.float32)
            last.bias[:3].copy_(inverse_softplus(base_diag - self.sigma_min))
            last.bias[3:6].zero_()
            target = (1.0 - self.amplitude_min) / max(self.amplitude_max - self.amplitude_min, 1e-6)
            target = min(max(target, 1e-4), 1.0 - 1e-4)
            last.bias[6].fill_(math.log(target / (1.0 - target)))
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal_signed = (rel * normal).sum(dim=-1)
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal_signed.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal_signed

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal_signed = self._geometry(surface_pos, surface_normal, residue_centers)

        params = self.gaussian_head(h_res)
        diag = self.sigma_min + pt.nn.functional.softplus(params[:, :3])
        offdiag = self.offdiag_scale * pt.tanh(params[:, 3:6])
        amp_raw = params[:, 6]
        amplitude = self.amplitude_min + (self.amplitude_max - self.amplitude_min) * pt.sigmoid(amp_raw)

        diag_pair = diag[idx].clamp_min(self.sigma_min)
        off_pair = offdiag[idx]
        x0 = d_tangent
        x1 = pt.zeros_like(x0)
        x2 = d_normal_signed

        l00 = diag_pair[..., 0]
        l11 = diag_pair[..., 1]
        l22 = diag_pair[..., 2]
        l10 = off_pair[..., 0]
        l20 = off_pair[..., 1]
        l21 = off_pair[..., 2]
        y0 = x0 / l00.clamp_min(1e-6)
        y1 = (x1 - l10 * y0) / l11.clamp_min(1e-6)
        y2 = (x2 - l20 * y0 - l21 * y1) / l22.clamp_min(1e-6)
        mahal = y0.square() + y1.square() + y2.square()
        score = -0.5 * mahal + amplitude[idx].clamp_min(1e-6).log()

        weight = pt.softmax(score, dim=-1)
        ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.context_proj(ctx) + self.geom_proj(surface_geom))
        self.last_diagnostics = {
            "fullcov_sigma_t1_mean": diag[:, 0].mean().detach(),
            "fullcov_sigma_t1_min": diag[:, 0].min().detach(),
            "fullcov_sigma_t1_max": diag[:, 0].max().detach(),
            "fullcov_sigma_t2_mean": diag[:, 1].mean().detach(),
            "fullcov_sigma_t2_min": diag[:, 1].min().detach(),
            "fullcov_sigma_t2_max": diag[:, 1].max().detach(),
            "fullcov_sigma_normal_mean": diag[:, 2].mean().detach(),
            "fullcov_sigma_normal_min": diag[:, 2].min().detach(),
            "fullcov_sigma_normal_max": diag[:, 2].max().detach(),
            "fullcov_offdiag_10_mean": offdiag[:, 0].mean().detach(),
            "fullcov_offdiag_20_mean": offdiag[:, 1].mean().detach(),
            "fullcov_offdiag_21_mean": offdiag[:, 2].mean().detach(),
            "residue_amplitude_mean": amplitude.mean().detach(),
            "residue_amplitude_min": amplitude.min().detach(),
            "residue_amplitude_max": amplitude.max().detach(),
            "residue_amplitude_std": amplitude.std(unbiased=False).detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class PhysChemGaussianFieldResidueToSurfaceProjector(MultiScaleGaussianResidueToSurfaceProjector):
    """Project learned residue context plus continuous physchem scalar fields."""

    accepts_phys_features = True

    def __init__(self, *args, phys_feature_dim=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.phys_feature_dim = max(1, int(phys_feature_dim))
        self.field_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(self.phys_feature_dim),
            pt.nn.Linear(self.phys_feature_dim, self.context_proj[0][1].out_features),
            pt.nn.GELU(),
            pt.nn.Linear(self.context_proj[0][1].out_features, self.context_proj[0][1].out_features),
        )

    def forward(self, h_res, residue_centers, cache, residue_phys_features=None):
        h_surf0, surface_pos, surface_normal, surface_geom, idx, weight = super().forward(h_res, residue_centers, cache)
        if residue_phys_features is None:
            phys = h_res.new_zeros((h_res.shape[0], self.phys_feature_dim))
        else:
            phys = residue_phys_features.to(device=h_res.device, dtype=h_res.dtype)
            if phys.shape[1] < self.phys_feature_dim:
                phys = pt.cat([phys, phys.new_zeros(phys.shape[0], self.phys_feature_dim - phys.shape[1])], dim=-1)
            elif phys.shape[1] > self.phys_feature_dim:
                phys = phys[:, : self.phys_feature_dim]
        field = (phys[idx.to(device=h_res.device, dtype=pt.long)] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(h_surf0 + self.field_proj(field))
        self.last_diagnostics = {
            "phys_field_abs_mean": field.abs().mean().detach(),
            "phys_field_std": field.std(unbiased=False).detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class MixtureBallGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Each residue emits a small mixture over Gaussian balls before surface pooling."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((3.0, 1.5), (5.0, 2.5), (8.0, 4.0), (14.0, 7.0)),
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one mixture component.")
        self.num_balls = len(self.sigma_pairs)
        self.context_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.ball_weight = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.num_balls),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        residue_mix = pt.softmax(self.ball_weight(h_res), dim=-1)
        local_mix = residue_mix[idx.to(device=h_res.device, dtype=pt.long)]
        scores = []
        for sigma_t, sigma_n in self.sigma_pairs:
            scores.append(-0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2))
        score = pt.stack(scores, dim=-1)
        score = score + local_mix.clamp_min(1e-8).log()
        weight = pt.softmax(pt.logsumexp(score, dim=-1), dim=-1)
        ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.context_proj(ctx) + self.geom_proj(surface_geom))
        entropy = -(residue_mix * residue_mix.clamp_min(1e-8).log()).sum(dim=-1)
        self.last_diagnostics = {
            "mixture_ball_entropy": entropy.mean().detach(),
            "mixture_ball_max_weight": residue_mix.max(dim=-1).values.mean().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class Local3DFrameResidueToSurfaceProjector(pt.nn.Module):
    """ScanNet-style local residue neighborhood reader for each surface point."""

    def __init__(self, dim=128, geom_dim=6, residue_topk=16, dropout=0.1):
        super().__init__()
        self.residue_topk = int(residue_topk)
        local_dim = dim + geom_dim + 7
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.message = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        dist_k = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist_k
        normal = surface_normal.unsqueeze(1)
        signed_normal = (rel * normal).sum(dim=-1, keepdim=True)
        d_normal = signed_normal.abs()
        d_tangent = (dist_k.square() - d_normal.square()).clamp_min(0.0).sqrt()
        geom = surface_geom.unsqueeze(1).expand(-1, k, -1)
        local = pt.cat([h_res[idx], geom, dist_k, d_tangent, signed_normal, direction], dim=-1)
        score = self.score(local).squeeze(-1)
        weight = pt.softmax(score, dim=-1)
        msg = self.message(local)
        ctx = (msg * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(ctx + self.geom_proj(surface_geom))
        entropy = -(weight * weight.clamp_min(1e-8).log()).sum(dim=-1) / max(math.log(max(k, 2)), 1e-8)
        self.last_diagnostics = {
            "local3d_attention_entropy": entropy.mean().detach(),
            "local3d_attention_max": weight.max(dim=-1).values.mean().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class Local3DAtomPointCloudResidueToSurfaceProjector(MultiScaleGaussianResidueToSurfaceProjector):
    """Fuse Gaussian residue projection with a local atom point-cloud reader."""

    accepts_atom_features = True

    def __init__(self, *args, atom_feature_dim=123, atom_topk=32, dropout=0.1, **kwargs):
        super().__init__(*args, dropout=dropout, **kwargs)
        dim = self.norm.normalized_shape[0]
        geom_dim = self.geom_proj[0].normalized_shape[0]
        self.atom_topk = int(atom_topk)
        self.atom_feature_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(int(atom_feature_dim)),
            pt.nn.Linear(int(atom_feature_dim), dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        local_dim = dim + geom_dim + 6
        self.atom_score = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.atom_message = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.atom_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.Sigmoid(),
        )
        self.atom_norm = pt.nn.LayerNorm(dim)

    def _atom_context(self, h_surf0, surface_pos, surface_normal, surface_geom, atom_pos, atom_features):
        if atom_pos is None or atom_features is None or atom_pos.numel() == 0:
            zero_ctx = h_surf0.new_zeros(h_surf0.shape)
            weight = h_surf0.new_zeros((h_surf0.shape[0], 1))
            return zero_ctx, weight
        atom_pos = atom_pos.to(device=h_surf0.device, dtype=h_surf0.dtype)
        atom_features = atom_features.to(device=h_surf0.device, dtype=h_surf0.dtype)
        k = min(self.atom_topk, int(atom_pos.shape[0]))
        dist = pt.cdist(surface_pos, atom_pos).clamp_min(0.0)
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = atom_pos[idx] - surface_pos.unsqueeze(1)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / d
        normal = surface_normal.unsqueeze(1)
        signed_normal = (rel * normal).sum(dim=-1, keepdim=True)
        d_tangent = (d.square() - signed_normal.square()).clamp_min(0.0).sqrt()
        atom_h = self.atom_feature_proj(atom_features)[idx]
        geom = surface_geom.unsqueeze(1).expand(-1, k, -1)
        local = pt.cat([atom_h, geom, d, d_tangent, signed_normal, direction], dim=-1)
        score = self.atom_score(local).squeeze(-1)
        weight = pt.softmax(score, dim=-1)
        message = self.atom_message(local)
        ctx = (message * weight.unsqueeze(-1)).sum(dim=1)
        return ctx, weight

    def forward(self, h_res, residue_centers, cache, atom_pos=None, atom_features=None):
        h_surf0, surface_pos, surface_normal, surface_geom, idx, weight = super().forward(h_res, residue_centers, cache)
        atom_ctx, atom_weight = self._atom_context(h_surf0, surface_pos, surface_normal, surface_geom, atom_pos, atom_features)
        gate = self.atom_gate(pt.cat([h_surf0, atom_ctx], dim=-1))
        h_surf0 = self.atom_norm(h_surf0 + gate * atom_ctx)
        entropy = -(atom_weight * atom_weight.clamp_min(1e-8).log()).sum(dim=-1)
        if atom_weight.shape[1] > 1:
            entropy = entropy / max(math.log(atom_weight.shape[1]), 1e-8)
        self.last_diagnostics = {
            "atom_pointcloud_attention_entropy": entropy.mean().detach(),
            "atom_pointcloud_attention_max": atom_weight.max(dim=-1).values.mean().detach(),
            "atom_pointcloud_gate_mean": gate.mean().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class Local3DPatchPointCloudResidueToSurfaceProjector(Local3DAtomPointCloudResidueToSurfaceProjector):
    """Atom point-cloud reader followed by surface-to-patch compression and gating."""

    def __init__(self, *args, patch_count=32, dropout=0.1, **kwargs):
        super().__init__(*args, dropout=dropout, **kwargs)
        dim = self.norm.normalized_shape[0]
        geom_dim = self.geom_proj[0].normalized_shape[0]
        self.patch_count = int(patch_count)
        self.patch_assignment = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim + 3),
            pt.nn.Linear(dim + geom_dim + 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.patch_count),
        )
        self.patch_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
            pt.nn.Sigmoid(),
        )
        self.patch_norm = pt.nn.LayerNorm(dim)

    def forward(self, h_res, residue_centers, cache, atom_pos=None, atom_features=None):
        h_surf0, surface_pos, surface_normal, surface_geom, idx, weight = super().forward(
            h_res,
            residue_centers,
            cache,
            atom_pos=atom_pos,
            atom_features=atom_features,
        )
        assign_logits = self.patch_assignment(pt.cat([h_surf0, surface_geom, surface_normal], dim=-1))
        assign = pt.softmax(assign_logits, dim=-1)
        usage = assign.sum(dim=0).clamp_min(1e-6)
        patch = (assign.transpose(0, 1) @ h_surf0) / usage.unsqueeze(-1)
        patch_gate = self.patch_gate(patch)
        surface_gate = assign @ patch_gate
        h_surf0 = self.patch_norm(h_surf0 * (1.0 + 0.1 * surface_gate))
        entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(self.patch_count, 2)), 1e-8)
        usage_norm = usage / usage.sum().clamp_min(1e-8)
        self.last_diagnostics.update(
            {
                "pointcloud_patch_assignment_entropy": entropy.mean().detach(),
                "pointcloud_patch_usage_active": (usage_norm > (0.25 / max(self.patch_count, 1))).float().sum().detach(),
                "pointcloud_patch_gate_mean": patch_gate.mean().detach(),
                "pointcloud_patch_gate_std": patch_gate.std(unbiased=False).detach(),
            }
        )
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class AttentionSurfaceToResiduePool(pt.nn.Module):
    def __init__(
        self,
        dim=128,
        surface_topk=24,
        dropout=0.1,
        mode="attention",
        gaussian_prior_weight=0.0,
        sigma_tangent=8.0,
        sigma_normal=4.0,
        phys_feature_dim=0,
        phys_attention_bias_weight=0.0,
        phys_context_weight=0.0,
        phys_message_weight=0.0,
        surface_vector_bridge_mode="none",
        surface_vector_dim=16,
    ):
        super().__init__()
        self.surface_topk = int(surface_topk)
        self.mode = str(mode)
        if self.mode not in {"attention", "per_surface_gaussian", "geoparam_attention"}:
            raise ValueError("surface-to-residue mode must be one of: attention, per_surface_gaussian, geoparam_attention")
        self.gaussian_prior_weight = float(gaussian_prior_weight)
        self.sigma_tangent = float(sigma_tangent)
        self.sigma_normal = float(sigma_normal)
        self.phys_feature_dim = int(phys_feature_dim)
        self.phys_attention_bias_weight = float(phys_attention_bias_weight)
        self.phys_context_weight = float(phys_context_weight)
        self.phys_message_weight = float(phys_message_weight)
        self.surface_vector_bridge_mode = str(surface_vector_bridge_mode)
        if self.surface_vector_bridge_mode not in {"none", "invariant", "equivariant"}:
            raise ValueError("surface_vector_bridge_mode must be one of: none, invariant, equivariant")
        self.surface_vector_dim = int(surface_vector_dim)
        vector_bridge_dim = 4 * self.surface_vector_dim if self.surface_vector_bridge_mode != "none" else 0
        score_input_dim = dim * 2 + 5 + vector_bridge_dim
        self.last_residue_vector_update = None
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(score_input_dim),
            pt.nn.Linear(score_input_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.context_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.surface_sigma_head = None
        if self.mode == "per_surface_gaussian":
            self.surface_sigma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
            last = self.surface_sigma_head[-1]
            pt.nn.init.zeros_(last.weight)
            pt.nn.init.zeros_(last.bias)
        self.geoparam_sigma_head = None
        self.geoparam_gamma_head = None
        if self.mode == "geoparam_attention":
            self.geoparam_sigma_min = 0.5
            self.geoparam_gamma_init = self.gaussian_prior_weight if self.gaussian_prior_weight > 0.0 else 0.25
            self.geoparam_sigma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
            sigma_last = self.geoparam_sigma_head[-1]
            pt.nn.init.zeros_(sigma_last.weight)
            base_sigma = pt.as_tensor([self.sigma_tangent, self.sigma_normal], dtype=pt.float32)
            with pt.no_grad():
                sigma_last.bias.copy_(inverse_softplus(base_sigma - self.geoparam_sigma_min))
            self.geoparam_gamma_head = pt.nn.Sequential(
                pt.nn.LayerNorm(score_input_dim),
                pt.nn.Linear(score_input_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            gamma_last = self.geoparam_gamma_head[-1]
            pt.nn.init.zeros_(gamma_last.weight)
            gamma_init = min(max(float(self.geoparam_gamma_init), 1e-4), 1.0 - 1e-4)
            with pt.no_grad():
                gamma_last.bias.fill_(math.log(gamma_init / (1.0 - gamma_init)))
        if vector_bridge_dim > 0:
            self.vector_context_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + vector_bridge_dim),
                pt.nn.Linear(dim + vector_bridge_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
        else:
            self.vector_context_proj = None
        if self.phys_feature_dim > 0:
            self.phys_pair_score = pt.nn.Sequential(
                pt.nn.LayerNorm(self.phys_feature_dim * 3 + 1),
                pt.nn.Linear(self.phys_feature_dim * 3 + 1, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            self.phys_context_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(dim + self.phys_feature_dim),
                pt.nn.Linear(dim + self.phys_feature_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.phys_edge_message = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + 5 + self.phys_feature_dim * 3 + 1),
                pt.nn.Linear(dim * 2 + 5 + self.phys_feature_dim * 3 + 1, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.phys_message_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.Sigmoid(),
            )
        else:
            self.phys_pair_score = None
            self.phys_context_proj = None
            self.phys_edge_message = None
            self.phys_message_gate = None

    def forward(
        self,
        h_res,
        h_surf,
        residue_centers,
        surface_pos,
        surface_normal,
        residue_phys_features=None,
        surface_phys_features=None,
        surface_vectors=None,
        residue_vectors=None,
        geometry_cache=None,
    ):
        self.last_residue_vector_update = None
        if geometry_cache is None:
            dist = pt.cdist(residue_centers, surface_pos).clamp_min(0.0)
            k = min(self.surface_topk, surface_pos.shape[0])
            idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
            rel = surface_pos[idx] - residue_centers.unsqueeze(1)
            d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            direction = rel / d
            normal = surface_normal[idx]
            normal_proj = (direction * normal).sum(dim=-1, keepdim=True)
            geom = pt.cat([d, direction, normal_proj], dim=-1)
        else:
            idx = geometry_cache["idx"]
            rel = geometry_cache["rel"]
            d = geometry_cache["d"]
            direction = geometry_cache["direction"]
            normal = geometry_cache["normal"]
            geom = geometry_cache["geom"]
        surf = h_surf[idx]
        vector_pair = None
        if self.vector_context_proj is not None and surface_vectors is not None:
            surf_v_all = surface_vectors.to(device=h_res.device, dtype=h_res.dtype)
            if surf_v_all.ndim == 3 and surf_v_all.shape[0] == surface_pos.shape[0]:
                surf_v = surf_v_all[idx]
                v_norm = surf_v.norm(dim=-1)
                v_dir = (surf_v * direction.unsqueeze(-2)).sum(dim=-1)
                v_normal = (surf_v * normal.unsqueeze(-2)).sum(dim=-1)
                vector_pair = pt.cat([v_norm, v_dir, v_dir.abs(), v_normal], dim=-1)
        res_expand = h_res.unsqueeze(1).expand_as(surf)
        score_parts = [res_expand, surf, geom]
        if vector_pair is not None:
            score_parts.append(vector_pair)
        score_input = pt.cat(score_parts, dim=-1)
        if self.mode == "per_surface_gaussian":
            d_normal = (rel * normal).sum(dim=-1).abs()
            d_tangent = (d.squeeze(-1).square() - d_normal.square()).clamp_min(0.0).sqrt()
            raw_sigma = self.surface_sigma_head(h_surf).to(device=h_res.device, dtype=h_res.dtype)
            # Initialize at the old SOTA prior scale: tangent 8A, normal 4A.
            sigma_t = 2.0 + 12.0 * pt.sigmoid(raw_sigma[:, 0])
            sigma_n = 1.0 + 6.0 * pt.sigmoid(raw_sigma[:, 1])
            local_sigma_t = sigma_t[idx].clamp_min(1e-6)
            local_sigma_n = sigma_n[idx].clamp_min(1e-6)
            logits = -0.5 * ((d_tangent / local_sigma_t) ** 2 + (d_normal / local_sigma_n) ** 2)
            self.last_diagnostics = {
                "surface_to_residue_sigma_tangent_mean": sigma_t.mean().detach(),
                "surface_to_residue_sigma_tangent_std": sigma_t.std(unbiased=False).detach(),
                "surface_to_residue_sigma_normal_mean": sigma_n.mean().detach(),
                "surface_to_residue_sigma_normal_std": sigma_n.std(unbiased=False).detach(),
            }
        else:
            logits = self.score(score_input).squeeze(-1)
            self.last_diagnostics = {}
        if self.mode == "geoparam_attention":
            d_normal = (rel * normal).sum(dim=-1).abs()
            d_tangent = (d.squeeze(-1).square() - d_normal.square()).clamp_min(0.0).sqrt()
            sigma_res = self.geoparam_sigma_min + pt.nn.functional.softplus(self.geoparam_sigma_head(h_res))
            sigma_t = sigma_res[:, 0].clamp_min(1e-6)
            sigma_n = sigma_res[:, 1].clamp_min(1e-6)
            prior = -0.5 * (
                (d_tangent / sigma_t.unsqueeze(1)) ** 2
                + (d_normal / sigma_n.unsqueeze(1)) ** 2
            )
            gamma = pt.sigmoid(self.geoparam_gamma_head(score_input)).squeeze(-1)
            logits = logits + gamma * prior
            self.last_diagnostics = {
                "surface_to_residue_geoparam_sigma_tangent_mean": sigma_t.mean().detach(),
                "surface_to_residue_geoparam_sigma_tangent_min": sigma_t.min().detach(),
                "surface_to_residue_geoparam_sigma_tangent_max": sigma_t.max().detach(),
                "surface_to_residue_geoparam_sigma_tangent_std": sigma_t.std(unbiased=False).detach(),
                "surface_to_residue_geoparam_sigma_normal_mean": sigma_n.mean().detach(),
                "surface_to_residue_geoparam_sigma_normal_min": sigma_n.min().detach(),
                "surface_to_residue_geoparam_sigma_normal_max": sigma_n.max().detach(),
                "surface_to_residue_geoparam_sigma_normal_std": sigma_n.std(unbiased=False).detach(),
                "surface_to_residue_geoparam_gamma_mean": gamma.mean().detach(),
                "surface_to_residue_geoparam_gamma_std": gamma.std(unbiased=False).detach(),
                "surface_to_residue_geoparam_prior_mean": prior.mean().detach(),
                "surface_to_residue_geoparam_prior_std": prior.std(unbiased=False).detach(),
            }
        if self.mode == "attention" and self.gaussian_prior_weight > 0:
            d_normal = (rel * normal).sum(dim=-1).abs()
            d_tangent = (d.squeeze(-1).square() - d_normal.square()).clamp_min(0.0).sqrt()
            prior = -0.5 * (
                (d_tangent / max(self.sigma_tangent, 1e-6)) ** 2
                + (d_normal / max(self.sigma_normal, 1e-6)) ** 2
            )
            logits = logits + self.gaussian_prior_weight * prior
        use_phys = (
            self.phys_pair_score is not None
            and self.phys_attention_bias_weight != 0.0
            and residue_phys_features is not None
            and surface_phys_features is not None
        )
        surf_phys = None
        if use_phys:
            res_phys = residue_phys_features.to(device=h_res.device, dtype=h_res.dtype)
            surf_phys_all = surface_phys_features.to(device=h_res.device, dtype=h_res.dtype)
            surf_phys = surf_phys_all[idx]
            res_phys_pair = res_phys.unsqueeze(1).expand_as(surf_phys)
            denom = res_phys_pair.norm(dim=-1, keepdim=True).clamp_min(1e-6) * surf_phys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            phys_cos = (res_phys_pair * surf_phys).sum(dim=-1, keepdim=True) / denom
            phys_pair = pt.cat([res_phys_pair, surf_phys, (res_phys_pair - surf_phys).abs(), phys_cos], dim=-1)
            logits = logits + self.phys_attention_bias_weight * self.phys_pair_score(phys_pair).squeeze(-1)
        else:
            phys_pair = None
        attn = pt.softmax(logits, dim=-1)
        ctx = (surf * attn.unsqueeze(-1)).sum(dim=1)
        out = self.context_proj(ctx)
        if self.vector_context_proj is not None and vector_pair is not None:
            vector_ctx = (vector_pair * attn.unsqueeze(-1)).sum(dim=1)
            out = out + self.vector_context_proj(pt.cat([ctx, vector_ctx], dim=-1))
            if self.surface_vector_bridge_mode == "equivariant" and residue_vectors is not None:
                raw_vector_ctx = (surf_v * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                if raw_vector_ctx.shape == residue_vectors.shape:
                    self.last_residue_vector_update = raw_vector_ctx
        if (
            self.phys_context_proj is not None
            and self.phys_context_weight != 0.0
            and residue_phys_features is not None
            and surface_phys_features is not None
        ):
            if surf_phys is None:
                surf_phys = surface_phys_features.to(device=h_res.device, dtype=h_res.dtype)[idx]
            phys_ctx = (surf_phys * attn.unsqueeze(-1)).sum(dim=1)
            out = out + self.phys_context_weight * self.phys_context_proj(pt.cat([ctx, phys_ctx], dim=-1))
        if (
            self.phys_edge_message is not None
            and self.phys_message_weight != 0.0
            and residue_phys_features is not None
            and surface_phys_features is not None
        ):
            if surf_phys is None:
                surf_phys = surface_phys_features.to(device=h_res.device, dtype=h_res.dtype)[idx]
            if phys_pair is None:
                res_phys = residue_phys_features.to(device=h_res.device, dtype=h_res.dtype)
                res_phys_pair = res_phys.unsqueeze(1).expand_as(surf_phys)
                denom = res_phys_pair.norm(dim=-1, keepdim=True).clamp_min(1e-6) * surf_phys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                phys_cos = (res_phys_pair * surf_phys).sum(dim=-1, keepdim=True) / denom
                phys_pair = pt.cat([res_phys_pair, surf_phys, (res_phys_pair - surf_phys).abs(), phys_cos], dim=-1)
            edge_msg = self.phys_edge_message(pt.cat([res_expand, surf, geom, phys_pair], dim=-1))
            msg_ctx = (edge_msg * attn.unsqueeze(-1)).sum(dim=1)
            msg_gate = self.phys_message_gate(pt.cat([h_res, msg_ctx], dim=-1))
            out = out + self.phys_message_weight * msg_gate * msg_ctx
        return out, attn


class AttentionResidueToSurfacePool(pt.nn.Module):
    """Residue-to-surface cross update used by bidirectional co-update blocks."""

    def __init__(self, dim=128, residue_topk=16, dropout=0.1):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2 + 5),
            pt.nn.Linear(dim * 2 + 5, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.context_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )

    def forward(self, h_surf, h_res, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        res = h_res[idx]
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / d
        normal = surface_normal.unsqueeze(1).expand(-1, k, -1)
        normal_proj = (direction * normal).sum(dim=-1, keepdim=True)
        geom = pt.cat([d, direction, normal_proj], dim=-1)
        surf_expand = h_surf.unsqueeze(1).expand_as(res)
        logits = self.score(pt.cat([surf_expand, res, geom], dim=-1)).squeeze(-1)
        attn = pt.softmax(logits, dim=-1)
        ctx = (res * attn.unsqueeze(-1)).sum(dim=1)
        return self.context_proj(ctx), attn


class SurfaceBusFromViews(pt.nn.Module):
    """Surface-point arbitration over view-specific residue states."""

    def __init__(self, dim=128, view_count=4, residue_topk=16, dropout=0.1):
        super().__init__()
        self.view_count = max(1, int(view_count))
        self.readers = pt.nn.ModuleList(
            [
                AttentionResidueToSurfacePool(dim=dim, residue_topk=residue_topk, dropout=dropout)
                for _ in range(self.view_count)
            ]
        )
        self.view_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * self.view_count + 6),
            pt.nn.Linear(dim * self.view_count + 6, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.view_count),
        )
        self.out_norm = pt.nn.LayerNorm(dim)
        self.last_diagnostics = {}

    def forward(self, h_surf_base, view_stack, surface_pos, surface_normal, surface_geom, residue_centers):
        if view_stack is None or view_stack.numel() == 0:
            self.last_diagnostics = {}
            return h_surf_base
        usable_views = min(int(view_stack.shape[0]), self.view_count)
        view_msgs = []
        attn_entropies = []
        for view_idx in range(usable_views):
            msg, attn = self.readers[view_idx](
                h_surf_base,
                view_stack[view_idx],
                surface_pos,
                surface_normal,
                residue_centers,
            )
            view_msgs.append(msg)
            attn_entropies.append(-(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean())
        while len(view_msgs) < self.view_count:
            view_msgs.append(h_surf_base.new_zeros(h_surf_base.shape))
        stacked = pt.stack(view_msgs, dim=1)
        gate_logits = self.view_gate(pt.cat([stacked.reshape(stacked.shape[0], -1), surface_geom], dim=-1))
        if usable_views < self.view_count:
            gate_logits[:, usable_views:] = -1.0e4
        gate = pt.softmax(gate_logits, dim=-1)
        bus = (gate.unsqueeze(-1) * stacked).sum(dim=1)
        entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
        self.last_diagnostics = {
            "surface_bus_view_entropy": entropy.detach(),
            "surface_bus_view_max": gate.max(dim=-1).values.mean().detach(),
            "surface_bus_update_norm": bus.norm(dim=-1).mean().detach(),
            "surface_bus_view_count": h_surf_base.new_tensor(float(usable_views)).detach(),
        }
        if attn_entropies:
            self.last_diagnostics["surface_bus_residue_entropy"] = pt.stack(attn_entropies).mean().detach()
        return self.out_norm(bus)


class StructuralCommunityDynamicAttention(pt.nn.Module):
    """Dynamic structural-community routing between residues and surface tokens."""

    def __init__(self, dim=128, num_communities=16, geom_bias=0.25, dropout=0.1):
        super().__init__()
        self.num_communities = max(1, int(num_communities))
        self.geom_bias = float(geom_bias)
        self.surface_score = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2 + 5),
            pt.nn.Linear(dim * 2 + 5, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.surface_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.surface_gate = pt.nn.Sequential(pt.nn.LayerNorm(dim * 2), pt.nn.Linear(dim * 2, dim), pt.nn.Sigmoid())
        self.residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.residue_gate = pt.nn.Sequential(pt.nn.LayerNorm(dim * 2), pt.nn.Linear(dim * 2, dim), pt.nn.Sigmoid())
        self.surface_norm = pt.nn.LayerNorm(dim)
        self.residue_norm = pt.nn.LayerNorm(dim)
        self.last_diagnostics = {}

    def _community_ids(self, centers):
        n = int(centers.shape[0])
        c = min(self.num_communities, max(n, 1))
        if n <= c:
            return pt.arange(n, device=centers.device, dtype=pt.long), c
        center = centers.mean(dim=0, keepdim=True)
        first = (centers - center).norm(dim=-1).argmax()
        seeds = [first]
        min_dist = pt.cdist(centers, centers[first : first + 1]).squeeze(-1)
        for _ in range(1, c):
            nxt = min_dist.argmax()
            seeds.append(nxt)
            dist_new = pt.cdist(centers, centers[nxt : nxt + 1]).squeeze(-1)
            min_dist = pt.minimum(min_dist, dist_new)
        seed_idx = pt.stack(seeds).to(device=centers.device, dtype=pt.long)
        dist_seed = pt.cdist(centers, centers[seed_idx])
        return dist_seed.argmin(dim=-1).to(dtype=pt.long), c

    def forward(self, h_res, h_surf, residue_centers, surface_pos, surface_normal):
        if h_res.shape[0] == 0 or h_surf.shape[0] == 0:
            self.last_diagnostics = {}
            return h_res, h_surf
        centers = residue_centers.to(device=h_res.device, dtype=h_res.dtype)
        ids, count = self._community_ids(centers)
        comm_tokens = []
        comm_centers = []
        active = []
        for cid in range(count):
            mask = ids == cid
            active.append(mask.to(dtype=h_res.dtype).sum())
            if pt.any(mask):
                comm_tokens.append(h_res[mask].mean(dim=0))
                comm_centers.append(centers[mask].mean(dim=0))
            else:
                comm_tokens.append(h_res.new_zeros(h_res.shape[-1]))
                comm_centers.append(centers.mean(dim=0))
        comm = pt.stack(comm_tokens, dim=0)
        comm_pos = pt.stack(comm_centers, dim=0)
        rel = comm_pos.unsqueeze(0) - surface_pos.unsqueeze(1)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / d
        normal = surface_normal.unsqueeze(1).expand(-1, count, -1)
        normal_proj = (direction * normal).sum(dim=-1, keepdim=True)
        geom = pt.cat([d, direction, normal_proj], dim=-1)
        surf_expand = h_surf.unsqueeze(1).expand(-1, count, -1)
        comm_expand = comm.unsqueeze(0).expand(h_surf.shape[0], -1, -1)
        logits = self.surface_score(pt.cat([surf_expand, comm_expand, geom], dim=-1)).squeeze(-1)
        if self.geom_bias != 0.0:
            logits = logits - self.geom_bias * d.squeeze(-1) / d.detach().mean().clamp_min(1e-6)
        attn = pt.softmax(logits, dim=-1)
        surf_ctx = (attn.unsqueeze(-1) * comm_expand).sum(dim=1)
        surf_input = pt.cat([h_surf, surf_ctx], dim=-1)
        h_surf_out = self.surface_norm(h_surf + self.surface_gate(surf_input) * self.surface_update(surf_input))
        res_ctx = comm[ids]
        res_input = pt.cat([h_res, res_ctx], dim=-1)
        h_res_out = self.residue_norm(h_res + self.residue_gate(res_input) * self.residue_update(res_input))
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        active_t = pt.stack(active) if active else h_res.new_zeros(1)
        self.last_diagnostics = {
            "struct_comm_active_mean": (active_t > 0).to(dtype=h_res.dtype).sum().detach(),
            "struct_comm_surface_entropy": entropy.detach(),
            "struct_comm_surface_gate_mean": self.surface_gate(surf_input).detach().mean(),
            "struct_comm_residue_gate_mean": self.residue_gate(res_input).detach().mean(),
        }
        return h_res_out, h_surf_out


class MotifPatchCrossAttention(pt.nn.Module):
    """Residue motif tokens interacting with existing surface patch tokens."""

    def __init__(self, dim=128, num_motifs=32, mode="dual_cross", beta=0.1, heads=4, dropout=0.1, geom_bias=0.1):
        super().__init__()
        self.num_motifs = max(1, int(num_motifs))
        self.mode = str(mode)
        if self.mode not in {"feedback", "dual_cross"}:
            raise ValueError("motif_patch_mode must be one of: none, feedback, dual_cross")
        self.beta = float(beta)
        self.geom_bias = float(geom_bias)
        heads = max(1, int(heads))
        if dim % heads != 0:
            heads = 1
        self.motif_self_attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.motif_from_patch = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.patch_from_motif = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.residue_from_patch = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.motif_norm = pt.nn.LayerNorm(dim)
        self.patch_norm = pt.nn.LayerNorm(dim)
        self.residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.context_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.context_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.residue_norm = pt.nn.LayerNorm(dim)
        self.context_norm = pt.nn.LayerNorm(dim)
        self.motif_aux_head = InterfaceHead(dim=dim, dropout=dropout)
        self.patch_aux_head = InterfaceHead(dim=dim, dropout=dropout)
        self.last_diagnostics = {}

    def _motif_ids(self, centers):
        n = int(centers.shape[0])
        c = min(self.num_motifs, max(n, 1))
        if n <= c:
            return pt.arange(n, device=centers.device, dtype=pt.long), c
        center = centers.mean(dim=0, keepdim=True)
        first = (centers - center).norm(dim=-1).argmax()
        seeds = [first]
        min_dist = pt.cdist(centers, centers[first : first + 1]).squeeze(-1)
        for _ in range(1, c):
            nxt = min_dist.argmax()
            seeds.append(nxt)
            dist_new = pt.cdist(centers, centers[nxt : nxt + 1]).squeeze(-1)
            min_dist = pt.minimum(min_dist, dist_new)
        seed_idx = pt.stack(seeds).to(device=centers.device, dtype=pt.long)
        dist_seed = pt.cdist(centers, centers[seed_idx])
        return dist_seed.argmin(dim=-1).to(dtype=pt.long), c

    def forward(self, h_res, centers, patch_tokens, surf_context=None):
        self.last_diagnostics = {}
        if patch_tokens is None or patch_tokens.numel() == 0 or h_res.shape[0] == 0:
            return h_res, surf_context, {}
        centers = centers.to(device=h_res.device, dtype=h_res.dtype)
        patch_tokens = patch_tokens.to(device=h_res.device, dtype=h_res.dtype)
        ids, count = self._motif_ids(centers)
        motif_tokens = []
        motif_centers = []
        active = []
        primary = []
        for cid in range(count):
            mask = ids == cid
            active.append(mask.to(dtype=h_res.dtype).sum())
            if pt.any(mask):
                motif_tokens.append(h_res[mask].mean(dim=0))
                motif_centers.append(centers[mask].mean(dim=0))
                local_idx = pt.nonzero(mask, as_tuple=False).squeeze(-1)
                primary.append(local_idx[0])
            else:
                motif_tokens.append(h_res.new_zeros(h_res.shape[-1]))
                motif_centers.append(centers.mean(dim=0))
                primary.append(h_res.new_tensor(0, dtype=pt.long))
        motif = pt.stack(motif_tokens, dim=0)
        motif_pos = pt.stack(motif_centers, dim=0)
        primary_idx = pt.stack(primary).to(device=h_res.device, dtype=pt.long)
        motif_self, _ = self.motif_self_attn(motif.unsqueeze(0), motif.unsqueeze(0), motif.unsqueeze(0), need_weights=False)
        motif = self.motif_norm(motif + motif_self.squeeze(0))
        motif_ctx, motif_attn = self.motif_from_patch(
            motif.unsqueeze(0),
            patch_tokens.unsqueeze(0),
            patch_tokens.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        motif = self.motif_norm(motif + self.beta * motif_ctx.squeeze(0))
        patch_updated = patch_tokens
        patch_attn_entropy = h_res.new_tensor(0.0)
        if self.mode == "dual_cross":
            patch_ctx, patch_attn = self.patch_from_motif(
                patch_tokens.unsqueeze(0),
                motif.unsqueeze(0),
                motif.unsqueeze(0),
                need_weights=True,
                average_attn_weights=True,
            )
            patch_updated = self.patch_norm(patch_tokens + self.beta * patch_ctx.squeeze(0))
            patch_attn_entropy = -(
                patch_attn.squeeze(0).clamp_min(1e-8) * patch_attn.squeeze(0).clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        patch_res_ctx, patch_res_attn = self.residue_from_patch(
            h_res.unsqueeze(0),
            patch_updated.unsqueeze(0),
            patch_updated.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        patch_res_ctx = patch_res_ctx.squeeze(0)
        motif_res_ctx = motif[ids]
        res_input = pt.cat([h_res, motif_res_ctx, patch_res_ctx], dim=-1)
        res_update = self.residue_update(res_input)
        res_gate = self.residue_gate(res_input)
        h_res_out = self.residue_norm(h_res + self.beta * res_gate * res_update)
        surf_context_out = surf_context
        if surf_context is not None:
            ctx_input = pt.cat([surf_context, motif_res_ctx, patch_res_ctx], dim=-1)
            ctx_update = self.context_update(ctx_input)
            ctx_gate = self.context_gate(ctx_input)
            surf_context_out = self.context_norm(surf_context + self.beta * ctx_gate * ctx_update)
        active_t = pt.stack(active) if active else h_res.new_zeros(1)
        motif_attn_entropy = -(
            motif_attn.squeeze(0).clamp_min(1e-8) * motif_attn.squeeze(0).clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        patch_res_entropy = -(
            patch_res_attn.squeeze(0).clamp_min(1e-8) * patch_res_attn.squeeze(0).clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        diag = {
            "motif_patch_motif_logits": self.motif_aux_head(motif),
            "motif_patch_motif_primary_idx": primary_idx,
            "motif_patch_patch_logits": self.patch_aux_head(patch_updated),
            "motif_patch_motif_count": h_res.new_tensor(float(count)).detach(),
            "motif_patch_active_mean": active_t.detach().float().mean(),
            "motif_patch_motif_patch_entropy": motif_attn_entropy.detach(),
            "motif_patch_patch_motif_entropy": patch_attn_entropy.detach(),
            "motif_patch_residue_patch_entropy": patch_res_entropy.detach(),
            "motif_patch_residue_gate_mean": res_gate.detach().mean(),
        }
        if surf_context is not None:
            diag["motif_patch_context_gate_mean"] = ctx_gate.detach().mean()
        self.last_diagnostics = {k: v for k, v in diag.items() if not k.endswith("_logits") and not k.endswith("_idx")}
        return h_res_out, surf_context_out, diag


class S1SurfaceFieldPredictor(pt.nn.Module):
    """A4-conditioned Surface Interface Field.

    Default loss is intentionally simple: final residue BCE plus a small base
    A4 auxiliary BCE. The surface branch is supervised through residue labels
    unless the training script explicitly adds pseudo-label losses later.
    """

    def __init__(
        self,
        atom_feature_dim=123,
        dim=128,
        vector_dim=16,
        graph_views=("seq", "k16", "k32", "exposure"),
        view_layers=2,
        rbf_bins=16,
        dropout=0.1,
        router_mode="global",
        use_augmented_features=True,
        residue_feature_dim=43,
        use_true_features=False,
        true_feature_dim=10,
        surface_points=1024,
        surface_graph_k=16,
        surface_extra_graph_k=0,
        surface_extra_graph_ks="",
        surface_graph_flow_mode="parallel",
        surface_layers=2,
        surface_residue_topk=8,
        residue_surface_topk=24,
        sigma_tangent=6.0,
        sigma_normal=3.0,
        projection_mode="single",
        multiscale_sigmas="4:2,8:4,14:7",
        surface_attention_prior_weight=0.0,
        surface_attention_sigma_tangent=8.0,
        surface_attention_sigma_normal=4.0,
        surface_to_residue_mode="attention",
        phys_feature_dim=0,
        phys_attention_bias_weight=0.0,
        phys_context_weight=0.0,
        phys_message_weight=0.0,
        surface_patch_mode="none",
        surface_patch_count=32,
        surface_patch_beta=0.1,
        surface_patch_attention_heads=4,
        surface_lappe_dim=0,
        surface_lappe_mode="none",
        surface_spectral_filter_dim=0,
        internal_operator_mode="none",
        internal_operator_k=16,
        internal_operator_layers=2,
        internal_operator_beta=0.1,
        surface_encoder_mode="graph",
        surface_fusion_mode="delta",
        global_feedback_rounds=1,
        global_feedback_interleave_residue_graph=False,
        surface_global_feedback_before_residue=False,
        surface_global_feedback_rounds=1,
        core_cycles=1,
        surface_vector_bridge_mode="none",
        surface_co_update_mode="none",
        surface_co_update_layers=1,
        surface_vector_state_mode="reinit",
        patch_state_mode="stateless",
        surface_serial_stage_mode="none",
        surface_round_end_r2s_mode="attention",
        post_final_residue_global_feedback_rounds=0,
        final_residue_skip_mode="none",
        final_residue_skip_source="multiview",
        final_residue_skip_beta=0.0,
        final_residue_skip_init_zero=False,
        pair_gaussian_mode="none",
        pair_gaussian_topk=8,
        pair_gaussian_beta=0.0,
        pair_gaussian_sigma_tangent=8.0,
        pair_gaussian_sigma_normal=4.0,
        pair_gaussian_sigma_mode="fixed",
        pair_gaussian_sigma_min=0.5,
        pair_gaussian_beta_mode="fixed",
        pair_gaussian_beta_min=0.0,
        pair_gaussian_beta_max=0.3,
        pair_gaussian_beta_init=0.05,
        pair_gaussian_content_mode="none",
        pair_gaussian_content_weight=1.0,
        pair_gaussian_rho_condition_mode="none",
        pair_gaussian_vector_condition_mode="none",
        pair_gaussian_single_compete_mode="none",
        pair_gaussian_single_rescue_beta=0.25,
        pair_gaussian_chunk_size=0,
        pair_gaussian_chunk_checkpoint=True,
        pair_gaussian_chunk_diagnostics="full",
        pair_gaussian_only=False,
        pair_gaussian_fusion_mode="init_only",
        pair_gaussian_joint_mode="independent",
        pair_gaussian_multiview_mode="none",
        pair_gaussian_multiview_beta=0.1,
        pair_gaussian_multiview_chunk_size=256,
        triplet_gaussian_mode="none",
        triplet_gaussian_topk=8,
        triplet_gaussian_beta=0.0,
        triplet_gaussian_beta_mode="fixed",
        triplet_gaussian_beta_min=0.0,
        triplet_gaussian_beta_max=0.2,
        triplet_gaussian_beta_init=0.03,
        triplet_gaussian_sigma_tangent=8.0,
        triplet_gaussian_sigma_normal=4.0,
        triplet_gaussian_fusion_mode="init_only",
        triplet_gaussian_score_temperature=1.0,
        triplet_gaussian_score_clamp=0.0,
        triplet_gaussian_preselect_topm=0,
        residue_amplitude_mode="none",
        residue_amplitude_min=0.25,
        residue_amplitude_max=2.0,
        surface_delta_scale=0.2,
        residue_vector_surface_mode="none",
        residue_vector_surface_topk=16,
        teacher_hidden_dim=0,
        gear_mode="none",
        gear_relation_k=32,
        gear_layers=2,
        gear_view_mode="none",
        oagnn_mode="none",
        oagnn_relation_k=32,
        oagnn_layers=2,
        token_encoder_mode="none",
        token_encoder_layers=2,
        token_encoder_heads=4,
        edge_pair_mode="none",
        cross_view_edge_mode="none",
        cross_view_edge_beta=0.1,
        cross_view_edge_heads=4,
        residue_encoder_mode="multiview",
        residue_graph_flow_mode="parallel",
        sequence_pos_enc_mode="none",
        sequence_pos_enc_dim=16,
        atom_residue_encoder_mode="mean",
        atom_graph_k=8,
        atom_contact_k=16,
        atom_contact_radius=5.0,
        atom_pre_encoder_mode="none",
        atom_pre_dim=64,
        atom_pre_layers=2,
        atom_pre_topk=16,
        atom_pre_cutoff=5.0,
        atom_pre_dropout=0.05,
        post_final_fourview_mode="none",
        post_final_fourview_beta=1.0,
        surface_bus_mode="none",
        surface_bus_beta=1.0,
        surface_bus_topk=16,
        struct_community_mode="none",
        struct_community_count=16,
        struct_community_geom_bias=0.25,
        stochastic_head_mode="none",
        ctmc_diffusion_steps=3,
        ctmc_diffusion_gamma=0.25,
        ctmc_diffusion_topk=16,
        poisson_lambda_max=20.0,
        pocket_mass_topk=16,
        motif_patch_mode="none",
        motif_patch_count=32,
        motif_patch_beta=0.1,
        motif_patch_attention_heads=4,
        motif_patch_geom_bias=0.1,
        semantic_patch_mode="none",
        semantic_patch_query_count=8,
        semantic_patch_beta=0.03,
        semantic_patch_radius_min=2.0,
        semantic_patch_radius_max=16.0,
        semantic_patch_adjacency_weight=0.25,
        semantic_patch_active_topk=4,
        semantic_patch_graph_hops=2,
        semantic_patch_geodesic_max_steps=96,
        semantic_patch_geodesic_normal_penalty=1.0,
        semantic_patch_geodesic_tangent_penalty=1.0,
        semantic_patch_detach_backbone=True,
        semantic_patch_boundary_mix=0.1,
        semantic_patch_boundary_temperature=0.7,
    ):
        super().__init__()
        self.use_augmented_features = bool(use_augmented_features)
        self.residue_feature_dim = int(residue_feature_dim)
        self.use_true_features = bool(use_true_features)
        self.true_feature_dim = int(true_feature_dim)
        self.phys_feature_dim = int(phys_feature_dim)
        extra_dim = 0
        if self.use_augmented_features:
            extra_dim += self.residue_feature_dim
        if self.use_true_features:
            extra_dim += self.true_feature_dim
        self.atom_pre_encoder_mode = str(atom_pre_encoder_mode)
        if self.atom_pre_encoder_mode not in {"none", "mpnn"}:
            raise ValueError("atom_pre_encoder_mode must be one of: none, mpnn")
        self.atom_pre_encoder = None
        if self.atom_pre_encoder_mode == "mpnn":
            self.atom_pre_encoder = AtomPreResidueEncoder(
                atom_feature_dim=atom_feature_dim,
                dim=dim,
                atom_dim=atom_pre_dim,
                layers=atom_pre_layers,
                topk=atom_pre_topk,
                cutoff=atom_pre_cutoff,
                dropout=atom_pre_dropout,
            )
            extra_dim += int(dim)
        self.last_extra_diagnostics = {}
        self.vector_dim = int(vector_dim)
        self.oagnn_mode = str(oagnn_mode)
        if self.oagnn_mode not in {"none", "oa_edge", "official"}:
            raise ValueError("oagnn_mode must be one of: none, oa_edge, official")
        self.residue_encoder_mode = str(residue_encoder_mode)
        if self.residue_encoder_mode not in {"multiview", "multiplex"}:
            raise ValueError("residue_encoder_mode must be one of: multiview, multiplex")
        self.residue_graph_flow_mode = str(residue_graph_flow_mode)
        serial_graph_flow_views = {
            "k16_then_k32": ("k16", "k32"),
            "k8_then_k16_then_k32_then_k64": ("k8", "k16", "k32", "k64"),
        }
        if self.residue_graph_flow_mode not in {"parallel", *serial_graph_flow_views.keys()}:
            raise ValueError(
                "residue_graph_flow_mode must be one of: parallel, "
                + ", ".join(sorted(serial_graph_flow_views))
            )
        if self.residue_graph_flow_mode != "parallel" and self.residue_encoder_mode != "multiview":
            raise ValueError("serial residue_graph_flow_mode requires residue_encoder_mode=multiview")
        if self.residue_graph_flow_mode != "parallel" and str(cross_view_edge_mode) != "none":
            raise ValueError("serial residue_graph_flow_mode requires cross_view_edge_mode=none")
        encoder_view_layers = int(oagnn_layers) if self.oagnn_mode != "none" else int(view_layers)
        if self.residue_encoder_mode == "multiplex":
            encoder_cls = StaticResidueMultiplexGVPEncoder
        elif self.residue_graph_flow_mode in serial_graph_flow_views:
            encoder_cls = SerialK16K32MultiViewGVPEncoder
        else:
            encoder_cls = StaticMultiViewGVPEncoder
        encoder_kwargs = dict(
            atom_feature_dim=atom_feature_dim,
            dim=dim,
            vector_dim=vector_dim,
            graph_views=graph_views,
            view_layers=encoder_view_layers,
            rbf_bins=rbf_bins,
            dropout=dropout,
            extra_residue_feature_dim=extra_dim,
            router_mode=router_mode,
            gear_view_mode=gear_view_mode,
            oagnn_mode=self.oagnn_mode,
            token_encoder_mode=token_encoder_mode,
            token_encoder_layers=token_encoder_layers,
            token_encoder_heads=token_encoder_heads,
            edge_pair_mode=edge_pair_mode,
            cross_view_edge_mode=cross_view_edge_mode,
            cross_view_edge_beta=cross_view_edge_beta,
            cross_view_edge_heads=cross_view_edge_heads,
            sequence_pos_enc_mode=sequence_pos_enc_mode,
            sequence_pos_enc_dim=sequence_pos_enc_dim,
            atom_residue_encoder_mode=atom_residue_encoder_mode,
            atom_graph_k=atom_graph_k,
            atom_contact_k=atom_contact_k,
            atom_contact_radius=atom_contact_radius,
        )
        if self.residue_graph_flow_mode in serial_graph_flow_views:
            encoder_kwargs["serial_views"] = serial_graph_flow_views[self.residue_graph_flow_mode]
        self.encoder = encoder_cls(**encoder_kwargs)
        self.global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.gear_mode = str(gear_mode)
        self.gear_refiner = None
        if self.gear_mode != "none":
            if self.gear_mode not in {"rel_edge", "edge_msg"}:
                raise ValueError("gear_mode must be one of: none, rel_edge, edge_msg")
            self.gear_refiner = GearStyleResidueGraphRefiner(
                dim=dim,
                relation_k=gear_relation_k,
                layers=gear_layers,
                dropout=dropout,
                mode=self.gear_mode,
            )
        self.base_head = InterfaceHead(dim=dim, dropout=dropout)
        self.aux_head = InterfaceHead(dim=dim, dropout=dropout)
        self.atom_pre_aux_head = InterfaceHead(dim=dim, dropout=dropout) if self.atom_pre_encoder is not None else None
        self.stochastic_head_mode = str(stochastic_head_mode)
        if self.stochastic_head_mode not in {"none", "poisson", "poisson_pocket", "ctmc_poisson", "event_competition"}:
            raise ValueError("stochastic_head_mode must be one of: none, poisson, poisson_pocket, ctmc_poisson, event_competition")
        self.ctmc_diffusion_steps = max(1, int(ctmc_diffusion_steps))
        self.ctmc_diffusion_gamma = float(ctmc_diffusion_gamma)
        self.ctmc_diffusion_topk = max(1, int(ctmc_diffusion_topk))
        self.poisson_lambda_max = float(poisson_lambda_max)
        self.pocket_mass_topk = max(1, int(pocket_mass_topk))
        self.ctmc_rate = None
        self.ctmc_gate = None
        if self.stochastic_head_mode == "ctmc_poisson":
            self.ctmc_rate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + 1),
                pt.nn.Linear(dim * 2 + 1, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            self.ctmc_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
                pt.nn.Sigmoid(),
            )
        self.event_competition_head = None
        if self.stochastic_head_mode == "event_competition":
            self.event_competition_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 4),
            )
        self.final_delta = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.surface_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
            pt.nn.Sigmoid(),
        )
        self.post_final_fourview_mode = str(post_final_fourview_mode)
        if self.post_final_fourview_mode not in {"none", "refine_dual_env"}:
            raise ValueError("post_final_fourview_mode must be one of: none, refine_dual_env")
        self.post_final_fourview_beta = float(post_final_fourview_beta)
        self.post_final_graph_head = None
        self.post_final_dual_gate = None
        if self.post_final_fourview_mode != "none":
            self.post_final_graph_head = InterfaceHead(dim=dim, dropout=dropout)
            self.post_final_dual_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 3),
                pt.nn.Linear(dim * 3, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
        self.surface_bus_mode = str(surface_bus_mode)
        if self.surface_bus_mode not in {"none", "static", "static_pre_global", "recurrent"}:
            raise ValueError("surface_bus_mode must be one of: none, static, static_pre_global, recurrent")
        if self.surface_bus_mode != "none" and self.residue_encoder_mode != "multiview":
            raise ValueError("surface_bus_mode requires residue_encoder_mode=multiview.")
        self.surface_bus_beta = float(surface_bus_beta)
        self.surface_bus = None
        self.surface_bus_static_norm = None
        self.surface_bus_view_update = None
        self.surface_bus_view_gate = None
        self.surface_bus_view_norm = None
        view_count = len(tuple(graph_views))
        if self.surface_bus_mode != "none":
            self.surface_bus = SurfaceBusFromViews(
                dim=dim,
                view_count=view_count,
                residue_topk=surface_bus_topk,
                dropout=dropout,
            )
            self.surface_bus_static_norm = pt.nn.LayerNorm(dim)
            self.surface_bus_view_update = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.surface_bus_view_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
                pt.nn.Sigmoid(),
            )
            self.surface_bus_view_norm = pt.nn.LayerNorm(dim)
        self.aug_feature_builder = None
        if self.use_augmented_features:
            self.aug_feature_builder = AugFeatureBuilder(atom_feature_dim=atom_feature_dim, residue_feature_dim=self.residue_feature_dim)
        self.surface_points = int(surface_points)
        self.surface_graph_k = int(surface_graph_k)
        self.surface_extra_graph_k = max(0, int(surface_extra_graph_k))
        self.surface_extra_graph_ks = _parse_positive_ints(surface_extra_graph_ks)
        self.surface_graph_flow_mode = str(surface_graph_flow_mode)
        if self.surface_graph_flow_mode not in {"parallel", "k16_then_k32"}:
            raise ValueError("surface_graph_flow_mode must be one of: parallel, k16_then_k32")
        self.surface_delta_scale = float(surface_delta_scale)
        self.surface_lappe_dim = max(0, int(surface_lappe_dim))
        self.surface_lappe_mode = str(surface_lappe_mode)
        if self.surface_lappe_mode not in {"none", "encoder", "patch", "both"}:
            raise ValueError("surface_lappe_mode must be one of: none, encoder, patch, both")
        if self.surface_lappe_dim <= 0:
            self.surface_lappe_mode = "none"
        self.surface_lappe_proj = None
        if self.surface_lappe_mode in {"encoder", "both"}:
            self.surface_lappe_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(self.surface_lappe_dim),
                pt.nn.Linear(self.surface_lappe_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
        self.surface_spectral_filter_dim = max(0, int(surface_spectral_filter_dim))
        self.surface_spectral_filter = None
        if self.surface_spectral_filter_dim > 0:
            self.surface_spectral_filter = SurfaceLaplacianSpectralFilter(
                dim=dim,
                lappe_dim=self.surface_spectral_filter_dim,
                dropout=dropout,
            )
        self.internal_operator_mode = str(internal_operator_mode)
        if self.internal_operator_mode not in {"none", "residue_laplacian"}:
            raise ValueError("internal_operator_mode must be one of: none, residue_laplacian")
        self.internal_operator = None
        if self.internal_operator_mode == "residue_laplacian":
            self.internal_operator = ResidueInternalLaplacianOperator(
                dim=dim,
                relation_k=internal_operator_k,
                layers=internal_operator_layers,
                beta=internal_operator_beta,
                dropout=dropout,
            )
        self.surface_encoder_mode = str(surface_encoder_mode)
        if self.surface_encoder_mode not in {"graph", "attention", "gvp"}:
            raise ValueError("surface_encoder_mode must be one of: graph, attention, gvp")
        if self.surface_graph_flow_mode == "k16_then_k32":
            if self.surface_encoder_mode == "attention":
                raise ValueError("surface_graph_flow_mode=k16_then_k32 requires graph or gvp surface_encoder_mode")
            if self.surface_graph_k != 16 or 32 not in set(self.surface_extra_graph_ks):
                raise ValueError("surface_graph_flow_mode=k16_then_k32 requires s1_surface_graph_k=16 and surface_extra_graph_ks including 32")
        self.surface_fusion_mode = str(surface_fusion_mode)
        if self.surface_fusion_mode not in {"delta", "same_level"}:
            raise ValueError("surface_fusion_mode must be one of: delta, same_level")
        self.global_feedback_rounds = max(1, int(global_feedback_rounds))
        self.global_feedback_interleave_residue_graph = bool(global_feedback_interleave_residue_graph)
        self.surface_global_feedback_before_residue = bool(surface_global_feedback_before_residue)
        self.surface_global_feedback_rounds = max(1, int(surface_global_feedback_rounds))
        self.surface_global_feedback = None
        if self.surface_global_feedback_before_residue:
            self.surface_global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.core_cycles = max(1, int(core_cycles))
        self.surface_vector_bridge_mode = str(surface_vector_bridge_mode)
        if self.surface_vector_bridge_mode not in {"none", "invariant", "equivariant"}:
            raise ValueError("surface_vector_bridge_mode must be one of: none, invariant, equivariant")
        self.surface_co_update_mode = str(surface_co_update_mode)
        if self.surface_co_update_mode not in {"none", "s2r", "bidir"}:
            raise ValueError("surface_co_update_mode must be one of: none, s2r, bidir")
        self.surface_co_update_layers = max(1, int(surface_co_update_layers))
        self.surface_serial_stage_mode = str(surface_serial_stage_mode)
        if self.surface_serial_stage_mode not in {"none", "pair_then_triplet", "perresidue_then_pair"}:
            raise ValueError("surface_serial_stage_mode must be one of: none, pair_then_triplet, perresidue_then_pair")
        self.surface_round_end_r2s_mode = str(surface_round_end_r2s_mode)
        if self.surface_round_end_r2s_mode not in {"attention", "skip"}:
            raise ValueError("surface_round_end_r2s_mode must be one of: attention, skip")
        self.post_final_residue_global_feedback_rounds = max(0, int(post_final_residue_global_feedback_rounds))
        self.post_final_residue_global_feedback = None
        if self.post_final_residue_global_feedback_rounds > 0:
            self.post_final_residue_global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.final_residue_skip_mode = str(final_residue_skip_mode)
        if self.final_residue_skip_mode not in {"none", "gated"}:
            raise ValueError("final_residue_skip_mode must be one of: none, gated")
        self.final_residue_skip_source = str(final_residue_skip_source)
        if self.final_residue_skip_source not in {"multiview"}:
            raise ValueError("final_residue_skip_source must be one of: multiview")
        self.final_residue_skip_beta = float(final_residue_skip_beta)
        self.final_residue_skip_proj = None
        self.final_residue_skip_gate = None
        self.final_residue_skip_norm = None
        if self.final_residue_skip_mode == "gated":
            self.final_residue_skip_proj = pt.nn.Linear(dim, dim)
            self.final_residue_skip_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 4),
                pt.nn.Linear(dim * 4, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.final_residue_skip_norm = pt.nn.LayerNorm(dim)
            if bool(final_residue_skip_init_zero):
                last = self.final_residue_skip_gate[-1]
                pt.nn.init.zeros_(last.weight)
                pt.nn.init.constant_(last.bias, -2.0)
        self.struct_community_mode = str(struct_community_mode)
        if self.struct_community_mode not in {"none", "dynamic_attn"}:
            raise ValueError("struct_community_mode must be one of: none, dynamic_attn")
        self.struct_community = None
        if self.struct_community_mode == "dynamic_attn":
            self.struct_community = StructuralCommunityDynamicAttention(
                dim=dim,
                num_communities=struct_community_count,
                geom_bias=struct_community_geom_bias,
                dropout=dropout,
            )
        self.motif_patch_mode = str(motif_patch_mode)
        if self.motif_patch_mode not in {"none", "feedback", "dual_cross"}:
            raise ValueError("motif_patch_mode must be one of: none, feedback, dual_cross")
        self.motif_patch = None
        if self.motif_patch_mode != "none":
            self.motif_patch = MotifPatchCrossAttention(
                dim=dim,
                num_motifs=motif_patch_count,
                mode=self.motif_patch_mode,
                beta=motif_patch_beta,
                heads=motif_patch_attention_heads,
                dropout=dropout,
                geom_bias=motif_patch_geom_bias,
            )
        self.surface_vector_state_mode = str(surface_vector_state_mode)
        if self.surface_vector_state_mode not in {"reinit", "persistent"}:
            raise ValueError("surface_vector_state_mode must be one of: reinit, persistent")
        self.residue_vector_surface_mode = str(residue_vector_surface_mode)
        if self.residue_vector_surface_mode not in {"none", "cross_attn"}:
            raise ValueError("residue_vector_surface_mode must be one of: none, cross_attn")
        self.residue_vector_surface_topk = max(1, int(residue_vector_surface_topk))
        self.patch_state_mode = str(patch_state_mode)
        if self.patch_state_mode not in {"stateless", "persistent"}:
            raise ValueError("patch_state_mode must be one of: stateless, persistent")
        self.pair_gaussian_mode = str(pair_gaussian_mode)
        if self.pair_gaussian_mode not in {"none", "k_nearest", "random"}:
            raise ValueError("pair_gaussian_mode must be one of: none, k_nearest, random")
        self.pair_gaussian_only = bool(pair_gaussian_only)
        self.pair_gaussian_fusion_mode = str(pair_gaussian_fusion_mode)
        if self.pair_gaussian_fusion_mode not in {"init_only", "deep", "coattn"}:
            raise ValueError("pair_gaussian_fusion_mode must be one of: init_only, deep, coattn")
        self.pair_gaussian_joint_mode = str(pair_gaussian_joint_mode)
        if self.pair_gaussian_joint_mode not in {"independent", "joint_residual", "joint_only"}:
            raise ValueError("pair_gaussian_joint_mode must be one of: independent, joint_residual, joint_only")
        self.pair_gaussian_multiview_mode = str(pair_gaussian_multiview_mode)
        if self.pair_gaussian_multiview_mode not in {"none", "init", "deep"}:
            raise ValueError("pair_gaussian_multiview_mode must be one of: none, init, deep")
        self.pair_gaussian_multiview_beta = float(pair_gaussian_multiview_beta)
        self.pair_gaussian_multiview_chunk_size = max(1, int(pair_gaussian_multiview_chunk_size))
        self.pair_gaussian_beta_mode = str(pair_gaussian_beta_mode)
        if self.pair_gaussian_beta_mode not in {"fixed", "learned_deep"}:
            raise ValueError("pair_gaussian_beta_mode must be one of: fixed, learned_deep")
        self.pair_gaussian_beta_min = float(pair_gaussian_beta_min)
        self.pair_gaussian_beta_max = float(pair_gaussian_beta_max)
        if self.pair_gaussian_beta_max <= self.pair_gaussian_beta_min:
            raise ValueError("pair_gaussian_beta_max must be greater than pair_gaussian_beta_min")
        self.pair_gaussian_beta_init = float(pair_gaussian_beta_init)
        self.pair_gaussian_deep_beta_logits = None
        self.pair_gaussian_content_mode = str(pair_gaussian_content_mode)
        self.pair_gaussian_content_weight = float(pair_gaussian_content_weight)
        self.pair_gaussian_rho_condition_mode = str(pair_gaussian_rho_condition_mode)
        if self.pair_gaussian_rho_condition_mode not in {"none", "surface_dot"}:
            raise ValueError("pair_gaussian_rho_condition_mode must be one of: none, surface_dot")
        self.pair_gaussian_vector_condition_mode = str(pair_gaussian_vector_condition_mode)
        if self.pair_gaussian_vector_condition_mode not in {"none", "surface_frame"}:
            raise ValueError("pair_gaussian_vector_condition_mode must be one of: none, surface_frame")
        self.pair_gaussian_single_compete_mode = str(pair_gaussian_single_compete_mode)
        if self.pair_gaussian_single_compete_mode not in {"none", "conservative_rescue", "strong_gate"}:
            raise ValueError("pair_gaussian_single_compete_mode must be one of: none, conservative_rescue, strong_gate")
        self.pair_gaussian_single_rescue_beta = float(pair_gaussian_single_rescue_beta)
        self.pair_gaussian_chunk_size = max(0, int(pair_gaussian_chunk_size))
        self.pair_gaussian_chunk_checkpoint = bool(pair_gaussian_chunk_checkpoint)
        self.pair_gaussian_chunk_diagnostics = str(pair_gaussian_chunk_diagnostics)
        if self.pair_gaussian_chunk_diagnostics not in {"full", "none"}:
            raise ValueError("pair_gaussian_chunk_diagnostics must be one of: full, none")
        self.triplet_gaussian_mode = str(triplet_gaussian_mode)
        if self.triplet_gaussian_mode not in {"none", "k_nearest", "random"}:
            raise ValueError("triplet_gaussian_mode must be one of: none, k_nearest, random")
        self.triplet_gaussian_beta_mode = str(triplet_gaussian_beta_mode)
        if self.triplet_gaussian_beta_mode not in {"fixed", "learned_deep"}:
            raise ValueError("triplet_gaussian_beta_mode must be one of: fixed, learned_deep")
        self.triplet_gaussian_beta_min = float(triplet_gaussian_beta_min)
        self.triplet_gaussian_beta_max = float(triplet_gaussian_beta_max)
        if self.triplet_gaussian_beta_max <= self.triplet_gaussian_beta_min:
            raise ValueError("triplet_gaussian_beta_max must be greater than triplet_gaussian_beta_min")
        self.triplet_gaussian_beta_init = float(triplet_gaussian_beta_init)
        self.triplet_gaussian_fusion_mode = str(triplet_gaussian_fusion_mode)
        if self.triplet_gaussian_fusion_mode not in {"init_only", "deep"}:
            raise ValueError("triplet_gaussian_fusion_mode must be one of: init_only, deep")
        self.triplet_gaussian_deep_beta_logits = None
        self.residue_amplitude_mode = str(residue_amplitude_mode)
        if self.residue_amplitude_mode not in {"none", "bounded"}:
            raise ValueError("residue_amplitude_mode must be one of: none, bounded")
        self.projection_mode = str(projection_mode)
        if self.projection_mode == "multiscale":
            self.res_to_surface = MultiScaleGaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                dropout=dropout,
            )
        elif self.projection_mode == "single":
            self.res_to_surface = GaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=surface_residue_topk,
                sigma_t=sigma_tangent,
                sigma_n=sigma_normal,
                dropout=dropout,
            )
        elif self.projection_mode == "multiscale_global":
            self.res_to_surface = GlobalWeightedMultiScaleGaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                dropout=dropout,
            )
        elif self.projection_mode in {"multiscale_learned", "anisotropic_sigma"}:
            self.res_to_surface = LearnableScaleGaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                dropout=dropout,
            )
        elif self.projection_mode == "per_residue_sigma":
            self.res_to_surface = PerResidueAdaptiveGaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                learn_amplitude=self.residue_amplitude_mode == "bounded",
                amplitude_min=residue_amplitude_min,
                amplitude_max=residue_amplitude_max,
                dropout=dropout,
            )
        elif self.projection_mode == "full_learnable_covariance":
            self.res_to_surface = FullLearnableCovarianceGaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                amplitude_min=residue_amplitude_min,
                amplitude_max=residue_amplitude_max,
                dropout=dropout,
            )
        elif self.projection_mode == "phys_gaussian_field":
            self.res_to_surface = PhysChemGaussianFieldResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                dropout=dropout,
                phys_feature_dim=max(phys_feature_dim, true_feature_dim, 1),
            )
        elif self.projection_mode == "mixture_ball_k4":
            self.res_to_surface = MixtureBallGaussianResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs="3:1.5,5:2.5,8:4,14:7",
                dropout=dropout,
            )
        elif self.projection_mode == "local3d_frame":
            self.res_to_surface = Local3DFrameResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                dropout=dropout,
            )
        elif self.projection_mode == "local3d_atom_pointcloud":
            self.res_to_surface = Local3DAtomPointCloudResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                dropout=dropout,
                atom_feature_dim=atom_feature_dim,
                atom_topk=32,
            )
        elif self.projection_mode == "local3d_patch_pointcloud":
            self.res_to_surface = Local3DPatchPointCloudResidueToSurfaceProjector(
                dim=dim,
                geom_dim=6,
                residue_topk=max(surface_residue_topk, 16),
                sigma_pairs=multiscale_sigmas,
                dropout=dropout,
                atom_feature_dim=atom_feature_dim,
                atom_topk=32,
                patch_count=surface_patch_count,
            )
        else:
            raise ValueError(
                "projection_mode must be one of: single, multiscale, multiscale_global, "
                "multiscale_learned, anisotropic_sigma, per_residue_sigma, "
                "phys_gaussian_field, mixture_ball_k4, local3d_frame, "
                "local3d_atom_pointcloud, local3d_patch_pointcloud"
            )
        if self.surface_encoder_mode == "gvp":
            self.surface_encoder = SurfacePointGVPGraphEncoder(
                dim=dim,
                vector_dim=vector_dim,
                geom_dim=6,
                dropout=dropout,
                layers=surface_layers,
            )
        else:
            self.surface_encoder = SurfacePointGraphEncoder(dim=dim, geom_dim=6, dropout=dropout, layers=surface_layers)
        if self.surface_extra_graph_ks and self.surface_encoder_mode == "attention":
            raise ValueError("surface_extra_graph_ks requires graph or gvp surface_encoder_mode.")
        self.surface_extra_encoders = pt.nn.ModuleList()
        self.surface_graph_logits = None
        self.surface_graph_fusion_norm = None
        if self.surface_extra_graph_ks:
            for _k in self.surface_extra_graph_ks:
                if self.surface_encoder_mode == "gvp":
                    self.surface_extra_encoders.append(
                        SurfacePointGVPGraphEncoder(
                            dim=dim,
                            vector_dim=vector_dim,
                            geom_dim=6,
                            dropout=dropout,
                            layers=surface_layers,
                        )
                    )
                else:
                    self.surface_extra_encoders.append(
                        SurfacePointGraphEncoder(dim=dim, geom_dim=6, dropout=dropout, layers=surface_layers)
                    )
            self.surface_graph_logits = pt.nn.Parameter(pt.zeros(1 + len(self.surface_extra_graph_ks)))
            self.surface_graph_fusion_norm = pt.nn.LayerNorm(dim)
        self.last_surface_graph_diagnostics = {}
        heads = 4 if dim % 4 == 0 else 1
        self.surface_attention_geom = pt.nn.Sequential(pt.nn.LayerNorm(6), pt.nn.Linear(6, dim))
        self.surface_attention_encoder = pt.nn.TransformerEncoder(
            pt.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=max(1, int(surface_layers)),
        )
        self.surface_to_residue = AttentionSurfaceToResiduePool(
            dim=dim,
            surface_topk=residue_surface_topk,
            dropout=dropout,
            mode=surface_to_residue_mode,
            gaussian_prior_weight=surface_attention_prior_weight,
            sigma_tangent=surface_attention_sigma_tangent,
            sigma_normal=surface_attention_sigma_normal,
            phys_feature_dim=phys_feature_dim,
            phys_attention_bias_weight=phys_attention_bias_weight,
            phys_context_weight=phys_context_weight,
            phys_message_weight=phys_message_weight,
            surface_vector_bridge_mode=self.surface_vector_bridge_mode,
            surface_vector_dim=vector_dim,
        )
        self.residue_vector_to_surface = None
        if self.residue_vector_surface_mode == "cross_attn":
            self.residue_vector_to_surface = ResidueVectorToSurfaceCrossAttention(
                dim=dim,
                vector_dim=vector_dim,
                geom_dim=6,
                residue_topk=self.residue_vector_surface_topk,
                dropout=dropout,
            )
        self.residue_to_surface = AttentionResidueToSurfacePool(
            dim=dim,
            residue_topk=surface_residue_topk,
            dropout=dropout,
        )
        self.surface_patch_cluster = None
        if str(surface_patch_mode) != "none":
            self.surface_patch_cluster = DynamicSurfacePatchCluster(
                dim=dim,
                geom_dim=6,
                num_patches=surface_patch_count,
                mode=surface_patch_mode,
                beta=surface_patch_beta,
                dropout=dropout,
                attention_heads=surface_patch_attention_heads,
                lappe_dim=self.surface_lappe_dim if self.surface_lappe_mode in {"patch", "both"} else 0,
            )
        self.semantic_patch_mode = str(semantic_patch_mode)
        self.semantic_patch_head = None
        if self.semantic_patch_mode != "none":
            if str(surface_patch_mode) != "connected_pool_adj":
                raise ValueError("semantic patch experiments require surface_patch_mode=connected_pool_adj")
            self.semantic_patch_head = SemanticConnectedPatchHead(
                dim=dim,
                query_count=semantic_patch_query_count,
                mode=self.semantic_patch_mode,
                beta=semantic_patch_beta,
                radius_min=semantic_patch_radius_min,
                radius_max=semantic_patch_radius_max,
                adjacency_weight=semantic_patch_adjacency_weight,
                active_topk=semantic_patch_active_topk,
                graph_hops=semantic_patch_graph_hops,
                geodesic_max_steps=semantic_patch_geodesic_max_steps,
                geodesic_normal_penalty=semantic_patch_geodesic_normal_penalty,
                geodesic_tangent_penalty=semantic_patch_geodesic_tangent_penalty,
                detach_backbone=semantic_patch_detach_backbone,
                boundary_mix=semantic_patch_boundary_mix,
                boundary_temperature=semantic_patch_boundary_temperature,
                dropout=dropout,
                attention_heads=surface_patch_attention_heads,
            )
        self.pair_gaussian = None
        self.pair_gaussian_multiview = None
        self.pair_gaussian_multiview_mix = None
        self.pair_gaussian_multiview_gate = None
        self.pair_gaussian_multiview_norm = None
        self.triplet_gaussian = None
        if self.pair_gaussian_mode != "none" and float(pair_gaussian_beta) != 0.0:
            self.pair_gaussian = PairGaussianResidueToSurfaceCorrection(
                dim=dim,
                geom_dim=6,
                pair_topk=pair_gaussian_topk,
                sigma_t=pair_gaussian_sigma_tangent,
                sigma_n=pair_gaussian_sigma_normal,
                sigma_mode=pair_gaussian_sigma_mode,
                sigma_min=pair_gaussian_sigma_min,
                beta=pair_gaussian_beta,
                mode=self.pair_gaussian_mode,
                joint_mode=self.pair_gaussian_joint_mode,
                fusion_mode="coattn" if self.pair_gaussian_fusion_mode == "coattn" else "residual",
                content_mode=self.pair_gaussian_content_mode,
                content_weight=self.pair_gaussian_content_weight,
                rho_condition_mode=self.pair_gaussian_rho_condition_mode,
                vector_condition_mode=pair_gaussian_vector_condition_mode,
                single_compete_mode=self.pair_gaussian_single_compete_mode,
                single_rescue_beta=self.pair_gaussian_single_rescue_beta,
                vector_dim=vector_dim,
                dropout=dropout,
            )
            if self.pair_gaussian_beta_mode == "learned_deep":
                beta_init = min(max(self.pair_gaussian_beta_init, self.pair_gaussian_beta_min + 1e-6), self.pair_gaussian_beta_max - 1e-6)
                beta_ratio = (beta_init - self.pair_gaussian_beta_min) / (self.pair_gaussian_beta_max - self.pair_gaussian_beta_min)
                beta_logit = math.log(beta_ratio / max(1.0 - beta_ratio, 1e-12))
                deep_rounds = max(1, self.surface_co_update_layers - 1 if self.surface_co_update_mode != "none" else self.core_cycles - 1)
                self.pair_gaussian_deep_beta_logits = pt.nn.Parameter(pt.full((deep_rounds,), float(beta_logit)))
            if self.pair_gaussian_multiview_mode != "none" and self.pair_gaussian_multiview_beta != 0.0:
                view_count = len(tuple(graph_views))
                self.pair_gaussian_multiview = pt.nn.ModuleList(
                    [
                        PairGaussianResidueToSurfaceCorrection(
                            dim=dim,
                            geom_dim=6,
                            pair_topk=pair_gaussian_topk,
                            sigma_t=pair_gaussian_sigma_tangent,
                            sigma_n=pair_gaussian_sigma_normal,
                            sigma_mode=pair_gaussian_sigma_mode,
                            sigma_min=pair_gaussian_sigma_min,
                            beta=pair_gaussian_beta,
                            mode=self.pair_gaussian_mode,
                            joint_mode=self.pair_gaussian_joint_mode,
                            fusion_mode="residual",
                            content_mode=self.pair_gaussian_content_mode,
                            content_weight=self.pair_gaussian_content_weight,
                            rho_condition_mode=self.pair_gaussian_rho_condition_mode,
                            vector_condition_mode=pair_gaussian_vector_condition_mode,
                            single_compete_mode=self.pair_gaussian_single_compete_mode,
                            single_rescue_beta=self.pair_gaussian_single_rescue_beta,
                            vector_dim=vector_dim,
                            dropout=dropout,
                        )
                        for _ in range(view_count)
                    ]
                )
                self.pair_gaussian_multiview_mix = pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * view_count),
                    pt.nn.Linear(dim * view_count, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                self.pair_gaussian_multiview_gate = pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 2 + 6),
                    pt.nn.Linear(dim * 2 + 6, dim),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                    pt.nn.Sigmoid(),
                )
                self.pair_gaussian_multiview_norm = pt.nn.LayerNorm(dim)
        if self.triplet_gaussian_mode != "none" and float(triplet_gaussian_beta) != 0.0:
            self.triplet_gaussian = TripletGaussianResidueToSurfaceCorrection(
                dim=dim,
                geom_dim=6,
                triplet_topk=triplet_gaussian_topk,
                sigma_t=triplet_gaussian_sigma_tangent,
                sigma_n=triplet_gaussian_sigma_normal,
                beta=triplet_gaussian_beta,
                mode=self.triplet_gaussian_mode,
                dropout=dropout,
                score_temperature=triplet_gaussian_score_temperature,
                score_clamp=triplet_gaussian_score_clamp,
                preselect_topm=triplet_gaussian_preselect_topm,
            )
            if self.triplet_gaussian_beta_mode == "learned_deep":
                beta_init = min(
                    max(self.triplet_gaussian_beta_init, self.triplet_gaussian_beta_min + 1e-6),
                    self.triplet_gaussian_beta_max - 1e-6,
                )
                beta_ratio = (beta_init - self.triplet_gaussian_beta_min) / (
                    self.triplet_gaussian_beta_max - self.triplet_gaussian_beta_min
                )
                beta_logit = math.log(beta_ratio / max(1.0 - beta_ratio, 1e-6))
                deep_rounds = max(1, self.core_cycles * (self.global_feedback_rounds + self.surface_global_feedback_rounds))
                self.triplet_gaussian_deep_beta_logits = pt.nn.Parameter(pt.full((deep_rounds,), float(beta_logit)))
        self.surface_head = InterfaceHead(dim=dim, dropout=dropout)
        self.same_level_head = InterfaceHead(dim=dim, dropout=dropout)
        self.same_level_fusion = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.core_residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.core_residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.core_norm = pt.nn.LayerNorm(dim)
        self.surface_cross_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.surface_cross_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.surface_cross_norm = pt.nn.LayerNorm(dim)
        self.residue_vector_scalar_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(vector_dim),
            pt.nn.Linear(vector_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.residue_vector_scalar_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.teacher_adapter = TeacherHiddenAdapter(dim, teacher_hidden_dim, dropout=dropout) if int(teacher_hidden_dim) > 0 else None


    def _poisson_logits_from_raw(self, raw_logits):
        lam = pt.nn.functional.softplus(raw_logits)
        if self.poisson_lambda_max > 0:
            lam = lam.clamp_max(float(self.poisson_lambda_max))
        odds = pt.expm1(lam).clamp_min(1e-8)
        return odds.log(), lam

    def _ctmc_diffuse_residue(self, h_res, centers):
        if self.ctmc_rate is None or h_res.shape[0] <= 1:
            return h_res, {}
        n = h_res.shape[0]
        k = min(self.ctmc_diffusion_topk, max(n - 1, 1))
        dist = pt.cdist(centers, centers).clamp_min(0.0)
        dist = dist + pt.eye(n, device=dist.device, dtype=dist.dtype) * 1.0e6
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        h_i = h_res.unsqueeze(1).expand(-1, k, -1)
        h_j = h_res[idx]
        d = dist.gather(1, idx).unsqueeze(-1) / 10.0
        rate = pt.nn.functional.softplus(self.ctmc_rate(pt.cat([h_i, h_j, d], dim=-1)).squeeze(-1))
        weight = rate / rate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        h = h_res
        gamma = max(0.0, min(float(self.ctmc_diffusion_gamma), 1.0))
        for _ in range(self.ctmc_diffusion_steps):
            msg = (weight.unsqueeze(-1) * h[idx]).sum(dim=1)
            h = (1.0 - gamma) * h + gamma * msg
        gate = self.ctmc_gate(pt.cat([h_res, h], dim=-1))
        out = h_res + gate * (h - h_res)
        weight_safe = weight.clamp_min(1e-8)
        diag = {
            "ctmc_rate_mean": rate.detach().mean(),
            "ctmc_rate_std": rate.detach().std(unbiased=False),
            "ctmc_rate_max": rate.detach().max(),
            "ctmc_gate_mean": gate.detach().mean(),
            "ctmc_gate_std": gate.detach().std(unbiased=False),
            "ctmc_gate_min": gate.detach().min(),
            "ctmc_gate_max": gate.detach().max(),
            "ctmc_update_norm": (out - h_res).detach().norm(dim=-1).mean(),
            "ctmc_update_rel_norm": ((out - h_res).detach().norm(dim=-1) / h_res.detach().norm(dim=-1).clamp_min(1e-6)).mean(),
            "ctmc_weight_entropy": (-(weight_safe * weight_safe.log()).sum(dim=-1)).detach().mean(),
            "ctmc_weight_max": weight.detach().max(dim=-1).values.mean(),
        }
        return out, diag

    def _stochastic_transform_logits(self, raw_logits, h_for_head):
        if self.stochastic_head_mode == "none":
            return raw_logits, {}
        if self.stochastic_head_mode in {"poisson", "poisson_pocket", "ctmc_poisson"}:
            logits, lam = self._poisson_logits_from_raw(raw_logits)
            return logits, {"poisson_lambda": lam.detach()}
        if self.stochastic_head_mode == "event_competition" and self.event_competition_head is not None:
            rates = pt.nn.functional.softplus(self.event_competition_head(h_for_head)).clamp_max(float(self.poisson_lambda_max))
            bind = rates[:, 0] + rates[:, 1]
            total = rates.sum(dim=-1).clamp_min(1e-8)
            p_event = bind / total
            p_contact = 1.0 - pt.exp(-bind)
            prob = (p_event * p_contact).clamp(1e-6, 1.0 - 1e-6)
            logits = pt.logit(prob)
            return logits, {
                "event_rate_open": rates[:, 0].detach(),
                "event_rate_contact": rates[:, 1].detach(),
                "event_rate_rigid": rates[:, 2].detach(),
                "event_rate_collapse": rates[:, 3].detach(),
            }
        return raw_logits, {}

    def _compose_extra_features(
        self, X, q, m=None, graph_cache=None, true_feat=None,
        atom_to_residue_index=None, num_residues=None, static_geometry=None,
    ):
        parts = []
        self.last_extra_diagnostics = {}
        self.last_atom_pre_residue_feat = None
        if self.atom_pre_encoder is not None:
            if m is None:
                raise ValueError("atom_pre_encoder is not supported by the dense-M-free clean path.")
            atom_residue_feat = self.atom_pre_encoder(X, q, m)
            self.last_atom_pre_residue_feat = atom_residue_feat
            parts.append(atom_residue_feat)
            for key, value in getattr(self.atom_pre_encoder, "last_diagnostics", {}).items():
                self.last_extra_diagnostics[key] = value.detach()
        if self.aug_feature_builder is not None:
            if static_geometry is not None and "aug_features" in static_geometry:
                parts.append(static_geometry["aug_features"].to(device=X.device, dtype=X.dtype))
            else:
                parts.append(self.aug_feature_builder(
                    X, q, m, graph_cache=graph_cache,
                    atom_to_residue_index=atom_to_residue_index, num_residues=num_residues,
                    centers_override=static_geometry.get("centers") if static_geometry is not None else None,
                ))
        if self.use_true_features:
            if true_feat is None:
                true_feat = X.new_zeros((int(num_residues if num_residues is not None else m.shape[1]), self.true_feature_dim))
            true_feat = true_feat.to(device=X.device, dtype=X.dtype)
            if true_feat.shape[1] < self.true_feature_dim:
                true_feat = pt.cat([true_feat, true_feat.new_zeros(true_feat.shape[0], self.true_feature_dim - true_feat.shape[1])], dim=-1)
            elif true_feat.shape[1] > self.true_feature_dim:
                true_feat = true_feat[:, : self.true_feature_dim]
            parts.append(true_feat)
        if not parts:
            return None
        return pt.cat([p.to(device=X.device, dtype=X.dtype) for p in parts], dim=-1)

    def _prepare_phys_features(self, true_feat, device, dtype):
        if self.phys_feature_dim <= 0 or true_feat is None:
            return None
        feat = true_feat.to(device=device, dtype=dtype)
        if feat.shape[1] < self.phys_feature_dim:
            feat = pt.cat([feat, feat.new_zeros(feat.shape[0], self.phys_feature_dim - feat.shape[1])], dim=-1)
        elif feat.shape[1] > self.phys_feature_dim:
            feat = feat[:, : self.phys_feature_dim]
        return feat

    def _fuse_view_states(self, view_stack, h_query=None):
        encoder = self.encoder
        view_outputs = [view_stack[idx] for idx in range(int(view_stack.shape[0]))]
        stacked = view_stack
        router_map = None
        if encoder.router_mode == "global":
            weights = pt.softmax(encoder.view_logits, dim=0).to(dtype=stacked.dtype, device=stacked.device)
            weights = weights[: stacked.shape[0]]
            weights = weights / weights.sum().clamp_min(1e-8)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            router_entropy = -(weights * weights.clamp_min(encoder.router_entropy_eps).log()).sum()
            router_map = weights.unsqueeze(0).expand(stacked.shape[1], -1)
        elif encoder.router_mode == "per_residue":
            per_residue = pt.cat(view_outputs, dim=-1)
            router_map = pt.softmax(encoder.view_router(per_residue), dim=-1)
            router_map = router_map[:, : stacked.shape[0]]
            router_map = router_map / router_map.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            fused = (router_map.unsqueeze(-1) * stacked.permute(1, 0, 2)).sum(dim=1)
            router_entropy = -(router_map * router_map.clamp_min(encoder.router_entropy_eps).log()).sum(dim=-1).mean()
        else:
            view_tokens = stacked.permute(1, 0, 2).contiguous()
            base_query = h_query if h_query is not None else view_tokens.mean(dim=1)
            query = encoder.residue_query(base_query).unsqueeze(1)
            fused = base_query
            for attn, update, norm in zip(encoder.view_attn_rounds, encoder.residue_updates, encoder.attn_round_norms):
                attn_out, attn_weights = attn(query, view_tokens, view_tokens, need_weights=True, average_attn_weights=True)
                query = norm(query + update(pt.cat([query, attn_out], dim=-1)))
                fused = query.squeeze(1)
                router_map = attn_weights.squeeze(1)
            router_entropy = -(router_map * router_map.clamp_min(encoder.router_entropy_eps).log()).sum(dim=-1).mean()
        return encoder.fusion_norm(fused), router_map, router_entropy

    def _surface_bus_pre_global_views(self, view_stack, mask):
        if view_stack is None or view_stack.numel() == 0:
            return view_stack, None
        updated = []
        deltas = []
        for view_idx in range(int(view_stack.shape[0])):
            before = view_stack[view_idx].unsqueeze(0)
            after = self.global_feedback(before, mask).squeeze(0)
            updated.append(after)
            deltas.append((after - view_stack[view_idx]).norm(dim=-1).mean())
        return pt.stack(updated, dim=0), pt.stack(deltas).mean() if deltas else None

    def _apply_surface_bus_static(self, h_surf_base, view_stack, surface_pos, surface_normal, surface_geom, centers, mask=None):
        if self.surface_bus is None or view_stack is None or view_stack.numel() == 0:
            return h_surf_base
        bus_views = view_stack
        pre_global_delta = None
        if self.surface_bus_mode == "static_pre_global":
            bus_views, pre_global_delta = self._surface_bus_pre_global_views(view_stack, mask)
        bus = self.surface_bus(
            h_surf_base,
            bus_views,
            surface_pos,
            surface_normal,
            surface_geom,
            centers,
        )
        if pre_global_delta is not None:
            self.surface_bus.last_diagnostics["surface_bus_pre_global_update_norm"] = pre_global_delta.detach()
        return self.surface_bus_static_norm(h_surf_base + self.surface_bus_beta * (bus - h_surf_base))

    def _advance_surface_bus_view_states(self, view_stack, surf_context, residue_vectors, centers, graph_cache=None, true_feat=None, chain_ids=None):
        encoder = self.encoder
        if not hasattr(encoder, "views"):
            raise ValueError("recurrent surface_bus_mode requires multiview encoder views.")
        if getattr(encoder, "cross_view_edge_mode", "none") != "none":
            raise ValueError("recurrent surface_bus_mode currently requires cross_view_edge_mode=none.")
        if getattr(encoder, "oagnn_mode", "none") == "official":
            raise ValueError("recurrent surface_bus_mode currently does not support oagnn_mode=official.")
        centers = centers.to(device=view_stack.device, dtype=view_stack.dtype)
        if residue_vectors is None:
            v0 = view_stack.new_zeros((view_stack.shape[1], self.vector_dim, 3))
        else:
            v0 = residue_vectors.to(device=view_stack.device, dtype=view_stack.dtype)
        updated_views = []
        update_norms = []
        gate_means = []
        for view_idx, name in enumerate(encoder.graph_views[: int(view_stack.shape[0])]):
            view_state = view_stack[view_idx]
            view_input = pt.cat([view_state, surf_context], dim=-1)
            view_update = self.surface_bus_view_update(view_input)
            view_gate = self.surface_bus_view_gate(view_input)
            view_state = self.surface_bus_view_norm(view_state + self.surface_bus_beta * view_gate * view_update)
            update_norms.append(view_update.detach().norm(dim=-1).mean())
            gate_means.append(view_gate.detach().mean())
            nn_idx, nn_dist = encoder._view_edges(name, centers, graph_cache, true_feature_cache=true_feat, chain_ids=chain_ids)
            nn_idx = nn_idx.to(device=view_state.device, dtype=pt.long)
            nn_dist = nn_dist.to(device=view_state.device, dtype=view_state.dtype)
            edge_s, edge_v = encoder.feature_builder.edge_features(centers, nn_idx, nn_dist)
            rel_type = encoder._edge_relation_types(name, centers, nn_idx, nn_dist)
            updated_views.append(
                encoder.views[name](
                    view_state,
                    v0,
                    nn_idx,
                    edge_s,
                    edge_v,
                    rel_type=rel_type,
                    centers=centers,
                )
            )
        next_stack = pt.stack(updated_views, dim=0)
        diag = {
            "surface_bus_recurrent_rounds": view_stack.new_tensor(1.0).detach(),
            "surface_bus_view_update_norm": pt.stack(update_norms).mean().detach() if update_norms else view_stack.new_tensor(0.0).detach(),
            "surface_bus_view_gate_mean": pt.stack(gate_means).mean().detach() if gate_means else view_stack.new_tensor(0.0).detach(),
        }
        return next_stack, diag

    def _interleaved_residue_graph_refine(self, h_res, residue_vectors, centers, graph_cache=None, true_feat=None, chain_ids=None):
        encoder = self.encoder
        if not hasattr(encoder, "views"):
            raise ValueError("global-feedback interleaved residue graph currently requires multiview residue_encoder_mode.")
        h = h_res.squeeze(0)
        centers = centers.to(device=h.device, dtype=h.dtype)
        if residue_vectors is None:
            v0 = h.new_zeros((h.shape[0], self.vector_dim, 3))
        else:
            v0 = residue_vectors.to(device=h.device, dtype=h.dtype)
        view_inputs = {}
        for name in encoder.graph_views:
            nn_idx, nn_dist = encoder._view_edges(name, centers, graph_cache, true_feature_cache=true_feat, chain_ids=chain_ids)
            nn_idx = nn_idx.to(device=h.device, dtype=pt.long)
            nn_dist = nn_dist.to(device=h.device, dtype=h.dtype)
            edge_s, edge_v = encoder.feature_builder.edge_features(centers, nn_idx, nn_dist)
            rel_type = encoder._edge_relation_types(name, centers, nn_idx, nn_dist)
            view_inputs[name] = {
                "nn_idx": nn_idx,
                "edge_s": edge_s,
                "edge_v": edge_v,
                "rel_type": rel_type,
            }
        if getattr(encoder, "cross_view_edge_mode", "none") == "none":
            view_outputs = []
            for name, values in view_inputs.items():
                view_outputs.append(
                    encoder.views[name](
                        h,
                        v0,
                        values["nn_idx"],
                        values["edge_s"],
                        values["edge_v"],
                        rel_type=values["rel_type"],
                        centers=centers,
                    )
                )
        else:
            view_outputs = encoder._forward_cross_view(h, v0, view_inputs, centers=centers)
        stacked = pt.stack(view_outputs, dim=0)
        if encoder.router_mode == "global":
            weights = pt.softmax(encoder.view_logits, dim=0).to(dtype=stacked.dtype, device=stacked.device)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            router_entropy = -(weights * weights.clamp_min(encoder.router_entropy_eps).log()).sum()
        elif encoder.router_mode == "per_residue":
            per_residue = pt.cat(view_outputs, dim=-1)
            router_map = pt.softmax(encoder.view_router(per_residue), dim=-1)
            fused = (router_map.unsqueeze(-1) * stacked.permute(1, 0, 2)).sum(dim=1)
            router_entropy = -(router_map * router_map.clamp_min(encoder.router_entropy_eps).log()).sum(dim=-1).mean()
        else:
            view_tokens = stacked.permute(1, 0, 2).contiguous()
            query = encoder.residue_query(h).unsqueeze(1)
            router_map = None
            fused = None
            for attn, update, norm in zip(encoder.view_attn_rounds, encoder.residue_updates, encoder.attn_round_norms):
                attn_out, attn_weights = attn(query, view_tokens, view_tokens, need_weights=True, average_attn_weights=True)
                query = norm(query + update(pt.cat([query, attn_out], dim=-1)))
                fused = query.squeeze(1)
                router_map = attn_weights.squeeze(1)
            router_entropy = -(router_map * router_map.clamp_min(encoder.router_entropy_eps).log()).sum(dim=-1).mean()
        return encoder.fusion_norm(fused).unsqueeze(0), router_entropy

    def _surface_cache(self, cache, X, q, m):
        if cache is not None:
            return cache
        if m is None:
            raise ValueError("A precomputed S1 surface cache is required by the dense-M-free clean path.")
        item = {
            "X": X.detach().float().cpu(),
            "q": q.detach().float().cpu(),
            "M": m.detach().float().cpu(),
        }
        return build_s1_surface_proxy_cache(item, num_surface_points=self.surface_points, graph_k=self.surface_graph_k)

    def _surface_graph_for_encoder(self, cache, surface_pos):
        if self.surface_extra_graph_ks:
            return self._surface_graph_for_k(cache, surface_pos, self.surface_graph_k)
        if self.surface_extra_graph_k <= 0:
            return cache.get("surface_graph_idx"), cache.get("surface_graph_dist")
        return self._surface_graph_for_k(cache, surface_pos, self.surface_extra_graph_k)

    def _surface_graph_for_k(self, cache, surface_pos, graph_k):
        graph_k = int(graph_k)
        if graph_k <= 0:
            return cache.get("surface_graph_idx"), cache.get("surface_graph_dist")
        if graph_k == self.surface_graph_k:
            graph_idx = cache.get("surface_graph_idx")
            graph_dist = cache.get("surface_graph_dist")
            if graph_idx is not None and graph_idx.numel() > 0:
                return graph_idx, graph_dist
        p = int(surface_pos.shape[0])
        if p <= 1:
            return surface_pos.new_empty((p, 0), dtype=pt.long), surface_pos.new_empty((p, 0))
        k = min(graph_k + 1, p)
        dist = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
        graph_dist, graph_idx = pt.topk(dist, k=k, dim=-1, largest=False)
        return graph_idx[:, 1:].contiguous(), graph_dist[:, 1:].contiguous()

    def _encode_surface_multigraph(
        self,
        h_surf_state,
        surface_geom,
        surface_pos,
        surface_normal,
        cache,
        surface_initial_vectors=None,
        graph_bundle=None,
    ):
        encoders = [self.surface_encoder, *list(self.surface_extra_encoders)]
        graph_ks = [self.surface_graph_k, *list(self.surface_extra_graph_ks)]
        if self.surface_graph_flow_mode == "k16_then_k32":
            h = h_surf_state
            vectors = []
            used_ks = []
            for encoder, graph_k in zip(encoders, graph_ks):
                graph_idx, graph_dist = (
                    graph_bundle[int(graph_k)] if graph_bundle is not None and int(graph_k) in graph_bundle
                    else self._surface_graph_for_k(cache, surface_pos, graph_k)
                )
                if self.surface_encoder_mode == "gvp":
                    h = encoder(
                        h,
                        surface_geom,
                        surface_pos,
                        surface_normal,
                        graph_idx,
                        graph_dist=graph_dist,
                        initial_vectors=surface_initial_vectors,
                    )
                    vec = getattr(encoder, "last_surface_vectors", None)
                    if vec is not None:
                        vectors.append(vec)
                        surface_initial_vectors = vec
                else:
                    h = encoder(
                        h,
                        surface_geom,
                        surface_pos,
                        surface_normal,
                        graph_idx,
                        graph_dist=graph_dist,
                    )
                used_ks.append(int(graph_k))
            if self.surface_encoder_mode == "gvp" and vectors:
                self.surface_encoder.last_surface_vectors = vectors[-1]
            diagnostics = {
                "surface_graph_view_count": h_surf_state.new_tensor(float(len(used_ks))).detach(),
                "surface_graph_flow_serial": h_surf_state.new_tensor(1.0).detach(),
            }
            for i, graph_k in enumerate(used_ks):
                diagnostics[f"surface_graph_serial_step{i + 1}_k"] = h_surf_state.new_tensor(float(graph_k)).detach()
            self.last_surface_graph_diagnostics = diagnostics
            return h
        outputs = []
        vectors = []
        used_ks = []
        for encoder, graph_k in zip(encoders, graph_ks):
            graph_idx, graph_dist = (
                graph_bundle[int(graph_k)] if graph_bundle is not None and int(graph_k) in graph_bundle
                else self._surface_graph_for_k(cache, surface_pos, graph_k)
            )
            if self.surface_encoder_mode == "gvp":
                out = encoder(
                    h_surf_state,
                    surface_geom,
                    surface_pos,
                    surface_normal,
                    graph_idx,
                    graph_dist=graph_dist,
                    initial_vectors=surface_initial_vectors,
                )
                vec = getattr(encoder, "last_surface_vectors", None)
                if vec is not None:
                    vectors.append(vec)
            else:
                out = encoder(
                    h_surf_state,
                    surface_geom,
                    surface_pos,
                    surface_normal,
                    graph_idx,
                    graph_dist=graph_dist,
                )
            outputs.append(out)
            used_ks.append(int(graph_k))
        if len(outputs) == 1:
            self.last_surface_graph_diagnostics = {
                "surface_graph_view_count": h_surf_state.new_tensor(1.0).detach(),
            }
            return outputs[0]
        weights = pt.softmax(self.surface_graph_logits[: len(outputs)], dim=0).to(device=h_surf_state.device, dtype=h_surf_state.dtype)
        stacked = pt.stack(outputs, dim=0)
        fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
        fused = self.surface_graph_fusion_norm(fused)
        if self.surface_encoder_mode == "gvp" and len(vectors) == len(outputs):
            vec_stack = pt.stack(vectors, dim=0)
            fused_vec = (weights.view(-1, 1, 1, 1) * vec_stack).sum(dim=0)
            self.surface_encoder.last_surface_vectors = fused_vec
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum()
        diagnostics = {
            "surface_graph_view_count": h_surf_state.new_tensor(float(len(outputs))).detach(),
            "surface_graph_router_entropy": entropy.detach(),
        }
        for graph_k, weight in zip(used_ks, weights):
            diagnostics[f"surface_graph_weight_k{graph_k}"] = weight.detach()
        self.last_surface_graph_diagnostics = diagnostics
        return fused

    def _apply_multiview_pair_gaussian(
        self,
        h_surf_state,
        view_stack,
        centers,
        surface_pos,
        surface_normal,
        surface_geom,
    ):
        if self.pair_gaussian_multiview is None or view_stack is None or view_stack.numel() == 0:
            return h_surf_state, {}
        view_surfaces = []
        diag_values = {}
        usable_views = min(int(view_stack.shape[0]), len(self.pair_gaussian_multiview))
        for view_idx in range(usable_views):
            chunk_surfaces = []
            module = self.pair_gaussian_multiview[view_idx]
            for start in range(0, int(surface_pos.shape[0]), self.pair_gaussian_multiview_chunk_size):
                end = min(start + self.pair_gaussian_multiview_chunk_size, int(surface_pos.shape[0]))
                zero_base = h_surf_state.new_zeros((end - start, h_surf_state.shape[-1]))
                def run_chunk(h_view, centers_arg, pos_arg, normal_arg, geom_arg, base_arg):
                    out_arg, _diag_arg = module(
                        h_view,
                        centers_arg,
                        pos_arg,
                        normal_arg,
                        geom_arg,
                        base_arg,
                        pair_only=True,
                    )
                    return out_arg

                if self.training:
                    view_surface = torch_checkpoint(
                        run_chunk,
                        view_stack[view_idx],
                        centers,
                        surface_pos[start:end],
                        surface_normal[start:end],
                        surface_geom[start:end],
                        zero_base,
                        use_reentrant=False,
                    )
                    with pt.no_grad():
                        _unused_surface, view_diag = module(
                            view_stack[view_idx].detach(),
                            centers.detach(),
                            surface_pos[start:end].detach(),
                            surface_normal[start:end].detach(),
                            surface_geom[start:end].detach(),
                            zero_base.detach(),
                            pair_only=True,
                        )
                else:
                    view_surface, view_diag = module(
                        view_stack[view_idx],
                        centers,
                        surface_pos[start:end],
                        surface_normal[start:end],
                        surface_geom[start:end],
                        zero_base,
                        pair_only=True,
                    )
                chunk_surfaces.append(view_surface)
                for key, value in view_diag.items():
                    diag_values.setdefault(f"pair_gaussian_multiview_{key}", []).append(value.detach())
            view_surfaces.append(pt.cat(chunk_surfaces, dim=0))
        if not view_surfaces:
            return h_surf_state, {}
        view_mix_input = pt.cat(view_surfaces, dim=-1)
        expected_dim = self.pair_gaussian_multiview_mix[0].normalized_shape[0]
        if view_mix_input.shape[-1] < expected_dim:
            pad = view_mix_input.new_zeros((view_mix_input.shape[0], expected_dim - view_mix_input.shape[-1]))
            view_mix_input = pt.cat([view_mix_input, pad], dim=-1)
        elif view_mix_input.shape[-1] > expected_dim:
            view_mix_input = view_mix_input[:, :expected_dim]
        view_mix = self.pair_gaussian_multiview_mix(view_mix_input)
        gate = self.pair_gaussian_multiview_gate(pt.cat([h_surf_state, view_mix, surface_geom], dim=-1))
        out = self.pair_gaussian_multiview_norm(h_surf_state + self.pair_gaussian_multiview_beta * gate * view_mix)
        diag = {
            "pair_gaussian_multiview_gate_mean": gate.mean().detach(),
            "pair_gaussian_multiview_update_norm": view_mix.norm(dim=-1).mean().detach(),
            "pair_gaussian_multiview_view_count": h_surf_state.new_tensor(float(usable_views)).detach(),
        }
        for key, values in diag_values.items():
            if values:
                diag[key] = pt.stack(values).mean().detach()
        return out, diag

    @staticmethod
    def _merge_chunk_diagnostics(weighted_diags):
        merged = {}
        totals = {}
        for diag, weight in weighted_diags:
            if not diag or weight <= 0:
                continue
            for key, value in diag.items():
                value = value.detach().reshape(()).to(dtype=pt.float32)
                merged[key] = merged.get(key, value.new_zeros(())) + value * float(weight)
                totals[key] = totals.get(key, 0.0) + float(weight)
        return {key: value / max(totals[key], 1.0) for key, value in merged.items()}

    def _pair_gaussian_pair_dist_scale(self, centers, surface_pos, precomputed_dist=None, precomputed_pair_indices=None):
        if precomputed_pair_indices is None:
            dist = precomputed_dist
            if dist is None:
                dist = pt.cdist(surface_pos, centers).clamp_min(0.0)
            k = min(self.pair_gaussian.pair_topk, centers.shape[0])
            idx_i, idx_j = self.pair_gaussian._pair_indices(dist, k)
        else:
            idx_i, idx_j = precomputed_pair_indices
        pair_vec = centers[idx_i] - centers[idx_j]
        return pair_vec.norm(dim=-1).clamp_min(1e-6).mean().clamp_min(1e-6)

    def _apply_pair_gaussian_chunked(
        self,
        h_res,
        centers,
        surface_pos,
        surface_normal,
        surface_geom,
        h_surf_state,
        residue_vectors=None,
        pair_only=False,
        beta_override=None,
        precomputed_dist=None,
        precomputed_pair_indices=None,
    ):
        if self.pair_gaussian is None:
            return h_surf_state, {}
        sigma_res_override = None
        if getattr(self.pair_gaussian, "sigma_mode", "") == "reuse_single" and hasattr(self.res_to_surface, "sigma_head"):
            sigma_res_override = self.res_to_surface.sigma_min + pt.nn.functional.softplus(self.res_to_surface.sigma_head(h_res))
        chunk_size = int(self.pair_gaussian_chunk_size)
        self._last_pair_gaussian_top_record = None
        if (
            chunk_size <= 0
            or int(surface_pos.shape[0]) <= chunk_size
            or getattr(self.pair_gaussian, "mode", "") == "random"
        ):
            out, diag = self.pair_gaussian(
                h_res,
                centers,
                surface_pos,
                surface_normal,
                surface_geom,
                h_surf_state,
                residue_vectors=residue_vectors,
                pair_only=pair_only,
                beta_override=beta_override,
                sigma_res_override=sigma_res_override,
                precomputed_dist=precomputed_dist,
                precomputed_pair_indices=precomputed_pair_indices,
            )
            self._last_pair_gaussian_top_record = getattr(self.pair_gaussian, "last_top_record", None)
            return out, diag

        chunk_surfaces = []
        weighted_diags = []
        best_pair_record = None
        pair_dist_scale = self._pair_gaussian_pair_dist_scale(
            centers, surface_pos, precomputed_dist=precomputed_dist,
            precomputed_pair_indices=precomputed_pair_indices,
        )
        for start in range(0, int(surface_pos.shape[0]), chunk_size):
            end = min(start + chunk_size, int(surface_pos.shape[0]))
            pos_chunk = surface_pos[start:end]
            normal_chunk = surface_normal[start:end]
            geom_chunk = surface_geom[start:end]
            h_surf_chunk = h_surf_state[start:end]

            def run_chunk(h_res_arg, centers_arg, pos_arg, normal_arg, geom_arg, h_surf_arg, res_vec_arg, scale_arg, sigma_arg):
                out_arg, _diag_arg = self.pair_gaussian(
                    h_res_arg,
                    centers_arg,
                    pos_arg,
                    normal_arg,
                    geom_arg,
                    h_surf_arg,
                    residue_vectors=res_vec_arg,
                    pair_only=pair_only,
                    beta_override=beta_override,
                    pair_dist_scale=scale_arg,
                    sigma_res_override=sigma_arg if sigma_arg.numel() else None,
                    precomputed_dist=precomputed_dist[start:end] if precomputed_dist is not None else None,
                    precomputed_pair_indices=(
                        (precomputed_pair_indices[0][start:end], precomputed_pair_indices[1][start:end])
                        if precomputed_pair_indices is not None else None
                    ),
                )
                return out_arg

            if self.training and self.pair_gaussian_chunk_checkpoint:
                pair_surface = torch_checkpoint(
                    run_chunk,
                    h_res,
                    centers,
                    pos_chunk,
                    normal_chunk,
                    geom_chunk,
                    h_surf_chunk,
                    residue_vectors,
                    pair_dist_scale,
                    sigma_res_override if sigma_res_override is not None else h_res.new_empty(0, 2),
                    use_reentrant=False,
                )
                if self.pair_gaussian_chunk_diagnostics == "full":
                    with pt.no_grad():
                        _unused_surface, pair_diag = self.pair_gaussian(
                            h_res.detach(),
                            centers.detach(),
                            pos_chunk.detach(),
                            normal_chunk.detach(),
                            geom_chunk.detach(),
                            h_surf_chunk.detach(),
                            residue_vectors=residue_vectors.detach() if residue_vectors is not None else None,
                            pair_only=pair_only,
                            beta_override=beta_override.detach() if pt.is_tensor(beta_override) else beta_override,
                            pair_dist_scale=pair_dist_scale.detach(),
                            sigma_res_override=sigma_res_override.detach() if sigma_res_override is not None else None,
                            precomputed_dist=precomputed_dist[start:end].detach() if precomputed_dist is not None else None,
                            precomputed_pair_indices=(
                                (precomputed_pair_indices[0][start:end], precomputed_pair_indices[1][start:end])
                                if precomputed_pair_indices is not None else None
                            ),
                        )
                else:
                    pair_diag = {}
            else:
                pair_surface, pair_diag = self.pair_gaussian(
                    h_res,
                    centers,
                    pos_chunk,
                    normal_chunk,
                    geom_chunk,
                    h_surf_chunk,
                    residue_vectors=residue_vectors,
                    pair_only=pair_only,
                    beta_override=beta_override,
                    pair_dist_scale=pair_dist_scale,
                    sigma_res_override=sigma_res_override,
                    precomputed_dist=precomputed_dist[start:end] if precomputed_dist is not None else None,
                    precomputed_pair_indices=(
                        (precomputed_pair_indices[0][start:end], precomputed_pair_indices[1][start:end])
                        if precomputed_pair_indices is not None else None
                    ),
                )
                if self.training and self.pair_gaussian_chunk_diagnostics == "none":
                    pair_diag = {}
            chunk_surfaces.append(pair_surface)
            weighted_diags.append((pair_diag, end - start))
            rec = getattr(self.pair_gaussian, "last_top_record", None)
            if rec is not None:
                rec = dict(rec)
                rec["surface_index"] = int(rec.get("surface_index", 0)) + int(start)
                if best_pair_record is None or float(rec.get("weight", -1.0)) > float(best_pair_record.get("weight", -1.0)):
                    best_pair_record = rec

        self._last_pair_gaussian_top_record = best_pair_record
        return pt.cat(chunk_surfaces, dim=0), self._merge_chunk_diagnostics(weighted_diags)

    def forward(
        self,
        X,
        ids_topk,
        q0,
        M,
        atom_counts,
        residue_counts,
        residue_graph_caches=None,
        s1_surface_caches=None,
        true_residue_features=None,
        residue_chain_ids=None,
        teacher_hidden=None,
        return_embeddings=False,
        atom_to_residue_index=None,
        s1_static_geometry_caches=None,
    ):
        logits = []
        prior_logits = []
        base_logits = []
        aux_logits = []
        atom_pre_aux_logits = []
        surface_logits = []
        surface_pooled_logits = []
        gate_values = []
        delta_values = []
        router_maps = []
        router_entropies = []
        residue_embeddings = []
        surface_embeddings = []
        patch_entropies = []
        patch_active_counts = []
        patch_usage_values = []
        patch_surface_gates = []
        patch_gate_values = []
        patch_primary_residue_idx = []
        motif_patch_motif_logits = []
        motif_patch_motif_primary_idx = []
        motif_patch_patch_logits = []
        motif_patch_patch_primary_idx = []
        semantic_patch_outputs = []
        sigma_diag_values = {}
        gaussian_top_records = []
        atom_offset = 0
        residue_offset = 0
        last_weights = None
        for sample_idx in range(len(atom_counts)):
            atom_count = int(atom_counts[sample_idx].item())
            residue_count = int(residue_counts[sample_idx].item())
            x = X[atom_offset : atom_offset + atom_count]
            q = q0[atom_offset : atom_offset + atom_count]
            m = None
            atom_residue_index = None
            if atom_to_residue_index is not None:
                atom_residue_index = atom_to_residue_index[atom_offset : atom_offset + atom_count] - residue_offset
            elif M is not None:
                m = M[atom_offset : atom_offset + atom_count, residue_offset : residue_offset + residue_count]
            else:
                raise ValueError("Either M or atom_to_residue_index must be provided.")
            graph_cache = residue_graph_caches[sample_idx] if residue_graph_caches is not None else None
            static_geometry = s1_static_geometry_caches[sample_idx] if s1_static_geometry_caches is not None else None
            true_feat = None
            if true_residue_features is not None:
                true_feat = true_residue_features[residue_offset : residue_offset + residue_count]
            chain_ids = None
            if residue_chain_ids is not None:
                chain_ids = residue_chain_ids[residue_offset : residue_offset + residue_count]
            extra = self._compose_extra_features(
                x, q, m, graph_cache=graph_cache, true_feat=true_feat,
                atom_to_residue_index=atom_residue_index, num_residues=residue_count,
                static_geometry=static_geometry,
            )
            atom_pre_feat = getattr(self, "last_atom_pre_residue_feat", None)
            if self.atom_pre_aux_head is not None and atom_pre_feat is not None:
                atom_pre_aux_logits.append(self.atom_pre_aux_head(atom_pre_feat))
            for key, value in getattr(self, "last_extra_diagnostics", {}).items():
                value = value.detach().to(device=x.device, dtype=pt.float32).reshape(())
                sigma_diag_values.setdefault(key, []).append(value)
            h_local, _view_stack, weights, router_map, router_entropy = self.encoder(
                x,
                q,
                m,
                graph_cache=graph_cache,
                extra_residue_features=extra,
                true_feature_cache=true_feat,
                chain_ids=chain_ids,
                atom_to_residue_index=atom_residue_index,
                num_residues=residue_count,
                static_geometry=static_geometry,
            )
            view_stack = _view_stack.to(device=h_local.device, dtype=h_local.dtype) if _view_stack is not None else None
            residue_vectors = getattr(self.encoder, "last_residue_vectors", None)
            if residue_vectors is not None:
                residue_vectors = residue_vectors.to(device=h_local.device, dtype=h_local.dtype)
            for key, value in getattr(self.encoder, "last_diagnostics", {}).items():
                value = value.detach().to(device=h_local.device, dtype=pt.float32).reshape(())
                sigma_diag_values.setdefault(key, []).append(value)
            mask = pt.ones((1, residue_count), dtype=h_local.dtype, device=h_local.device)
            h_res = h_local
            centers = (
                static_geometry["centers"].to(device=x.device, dtype=x.dtype)
                if static_geometry is not None
                else self.encoder.feature_builder.centers_from_atom_mask(
                    x, m, atom_to_residue_index=atom_residue_index, num_residues=residue_count
                )
            )
            if self.global_feedback_interleave_residue_graph and self.global_feedback_rounds >= 2:
                h_res = self.global_feedback(h_res, mask)
                h_res, interleave_entropy = self._interleaved_residue_graph_refine(
                    h_res,
                    residue_vectors,
                    centers,
                    graph_cache=graph_cache,
                    true_feat=true_feat,
                    chain_ids=chain_ids,
                )
                sigma_diag_values.setdefault("interleave_residue_graph_router_entropy", []).append(
                    interleave_entropy.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                for _ in range(self.global_feedback_rounds - 1):
                    h_res = self.global_feedback(h_res, mask)
            else:
                for _ in range(self.global_feedback_rounds):
                    h_res = self.global_feedback(h_res, mask)
            h_res = h_res.squeeze(0)
            h_res_multiview_skip = h_res
            if self.gear_refiner is not None:
                h_res = self.gear_refiner(h_res, centers, graph_cache=graph_cache)
                for key, value in getattr(self.gear_refiner, "last_diagnostics", {}).items():
                    value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                    sigma_diag_values.setdefault(key, []).append(value)
            if teacher_hidden is not None and self.teacher_adapter is not None:
                th = teacher_hidden[residue_offset : residue_offset + residue_count]
                h_res = self.teacher_adapter(h_res, th)
            if self.internal_operator is not None:
                h_res = self.internal_operator(h_res, centers, graph_cache=graph_cache)
                for key, value in getattr(self.internal_operator, "last_diagnostics", {}).items():
                    value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                    sigma_diag_values.setdefault(key, []).append(value)
            if self.training:
                fork_devices = [h_res.device.index] if h_res.is_cuda else []
                with pt.random.fork_rng(devices=fork_devices, enabled=True):
                    prior_logit = self.base_head(h_res)
            else:
                prior_logit = self.base_head(h_res)
            view_states = view_stack
            cache = self._surface_cache(
                s1_surface_caches[sample_idx] if s1_surface_caches is not None else None,
                x,
                q,
                m,
            )
            phys_feat = self._prepare_phys_features(true_feat, device=h_res.device, dtype=h_res.dtype)
            surface_lappe = cache.get("surface_lappe")
            if self.surface_lappe_dim > 0:
                if surface_lappe is None:
                    surface_lappe = h_res.new_zeros((int(cache["surface_pos"].shape[0]), self.surface_lappe_dim))
                else:
                    surface_lappe = surface_lappe.to(device=h_res.device, dtype=h_res.dtype)
                    if surface_lappe.shape[-1] < self.surface_lappe_dim:
                        surface_lappe = pt.cat(
                            [
                                surface_lappe,
                                surface_lappe.new_zeros((surface_lappe.shape[0], self.surface_lappe_dim - surface_lappe.shape[-1])),
                            ],
                            dim=-1,
                        )
                    elif surface_lappe.shape[-1] > self.surface_lappe_dim:
                        surface_lappe = surface_lappe[:, : self.surface_lappe_dim]
            else:
                surface_lappe = None
            patch_diag = {}
            surf_attn = None
            if self.surface_co_update_mode == "none":
                update_rounds = self.core_cycles
            else:
                update_rounds = self.surface_co_update_layers
            if self.surface_serial_stage_mode != "none":
                update_rounds = update_rounds * 2
            h_surf_state = None
            surface_vector_state = None
            patch_state = None
            connected_patch_partition = None
            surface_graph_bundle = None
            surface_runtime_geometry = static_geometry if not self.training else None
            for core_idx in range(update_rounds):
                stage_idx = core_idx % 2 if self.surface_serial_stage_mode != "none" else 0
                should_project_surface = core_idx == 0 or self.surface_co_update_mode == "none"
                if self.surface_serial_stage_mode == "perresidue_then_pair" and stage_idx == 0:
                    should_project_surface = True
                if should_project_surface:
                    if getattr(self.res_to_surface, "accepts_phys_features", False):
                        h_surf0, surface_pos, surface_normal, surface_geom, rs_idx, rs_weight = self.res_to_surface(
                            h_res,
                            centers,
                            cache,
                            residue_phys_features=phys_feat,
                        )
                    elif getattr(self.res_to_surface, "accepts_atom_features", False):
                        h_surf0, surface_pos, surface_normal, surface_geom, rs_idx, rs_weight = self.res_to_surface(
                            h_res,
                            centers,
                            cache,
                            atom_pos=x,
                            atom_features=q,
                        )
                    else:
                        h_surf0, surface_pos, surface_normal, surface_geom, rs_idx, rs_weight = self.res_to_surface(h_res, centers, cache)
                    if not self.training and surface_runtime_geometry is None:
                        surface_residue_dist = getattr(self.res_to_surface, "last_surface_residue_dist", None)
                        if surface_residue_dist is None or surface_residue_dist.shape != (surface_pos.shape[0], centers.shape[0]):
                            surface_residue_dist = pt.cdist(surface_pos, centers).clamp_min(0.0)
                        res_surface_dist = pt.cdist(centers, surface_pos).clamp_min(0.0)
                        surface_topk = min(self.surface_to_residue.surface_topk, surface_pos.shape[0])
                        surface_idx = pt.topk(res_surface_dist, k=surface_topk, dim=-1, largest=False).indices
                        surface_rel = surface_pos[surface_idx] - centers.unsqueeze(1)
                        surface_d = surface_rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        surface_direction = surface_rel / surface_d
                        local_normal = surface_normal[surface_idx]
                        normal_proj = (surface_direction * local_normal).sum(dim=-1, keepdim=True)
                        surface_runtime_geometry = {
                            "surface_to_residue": {
                                "idx": surface_idx,
                                "rel": surface_rel,
                                "d": surface_d,
                                "direction": surface_direction,
                                "normal": local_normal,
                                "geom": pt.cat([surface_d, surface_direction, normal_proj], dim=-1),
                            },
                        }
                        if self.pair_gaussian is not None:
                            pair_k = min(self.pair_gaussian.pair_topk, centers.shape[0])
                            surface_runtime_geometry["pair_indices"] = self.pair_gaussian._pair_indices(
                                surface_residue_dist, pair_k
                            )
                        surface_graph_bundle = {}
                        for graph_k in (self.surface_graph_k, *self.surface_extra_graph_ks):
                            cached_idx = surface_runtime_geometry.get(f"surface_graph_idx_k{int(graph_k)}")
                            cached_dist = surface_runtime_geometry.get(f"surface_graph_dist_k{int(graph_k)}")
                            if cached_idx is not None and cached_dist is not None:
                                surface_graph_bundle[int(graph_k)] = (
                                    cached_idx.to(device=surface_pos.device, dtype=pt.long),
                                    cached_dist.to(device=surface_pos.device, dtype=surface_pos.dtype),
                                )
                            else:
                                surface_graph_bundle[int(graph_k)] = self._surface_graph_for_k(cache, surface_pos, graph_k)
                    if not self.training and surface_graph_bundle is None:
                        surface_graph_bundle = {}
                        for graph_k in (self.surface_graph_k, *self.surface_extra_graph_ks):
                            cached_idx = surface_runtime_geometry.get(f"surface_graph_idx_k{int(graph_k)}")
                            cached_dist = surface_runtime_geometry.get(f"surface_graph_dist_k{int(graph_k)}")
                            if cached_idx is not None and cached_dist is not None:
                                surface_graph_bundle[int(graph_k)] = (
                                    cached_idx.to(device=surface_pos.device, dtype=pt.long),
                                    cached_dist.to(device=surface_pos.device, dtype=surface_pos.dtype),
                                )
                            else:
                                surface_graph_bundle[int(graph_k)] = self._surface_graph_for_k(cache, surface_pos, graph_k)
                    if self.surface_lappe_proj is not None and surface_lappe is not None:
                        h_surf0 = h_surf0 + self.surface_lappe_proj(surface_lappe)
                    h_surf_state = h_surf0
                    for key, value in getattr(self.res_to_surface, "last_diagnostics", {}).items():
                        value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        sigma_diag_values.setdefault(key, []).append(value)
                    if self.surface_bus_mode in {"static", "static_pre_global"}:
                        h_surf_state = self._apply_surface_bus_static(
                            h_surf_state,
                            view_stack,
                            surface_pos,
                            surface_normal,
                            surface_geom,
                            centers,
                            mask=mask,
                        )
                        for key, value in getattr(self.surface_bus, "last_diagnostics", {}).items():
                            value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                            sigma_diag_values.setdefault(key, []).append(value)
                if self.surface_bus_mode == "recurrent":
                    h_surf_state = self._apply_surface_bus_static(
                        h_surf_state,
                        view_states,
                        surface_pos,
                        surface_normal,
                        surface_geom,
                        centers,
                        mask=mask,
                    )
                    for key, value in getattr(self.surface_bus, "last_diagnostics", {}).items():
                        value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        sigma_diag_values.setdefault(key, []).append(value)
                pair_stage_active = core_idx == 0 or self.pair_gaussian_fusion_mode in {"deep", "coattn"}
                if self.surface_serial_stage_mode == "pair_then_triplet":
                    pair_stage_active = stage_idx == 0
                elif self.surface_serial_stage_mode == "perresidue_then_pair":
                    pair_stage_active = stage_idx == 1
                if self.pair_gaussian is not None and pair_stage_active:
                    pair_beta_override = None
                    if self.pair_gaussian_beta_mode == "learned_deep":
                        if core_idx == 0:
                            pair_beta_override = h_res.new_tensor(1.0)
                        elif self.pair_gaussian_deep_beta_logits is not None:
                            beta_idx = min(core_idx - 1, int(self.pair_gaussian_deep_beta_logits.numel()) - 1)
                            beta_logit = self.pair_gaussian_deep_beta_logits[beta_idx].to(device=h_res.device, dtype=h_res.dtype)
                            pair_beta_override = self.pair_gaussian_beta_min + (
                                self.pair_gaussian_beta_max - self.pair_gaussian_beta_min
                            ) * pt.sigmoid(beta_logit)
                    h_surf_state, pair_diag = self._apply_pair_gaussian_chunked(
                        h_res,
                        centers,
                        surface_pos,
                        surface_normal,
                        surface_geom,
                        h_surf_state,
                        residue_vectors=residue_vectors,
                        pair_only=(True if self.surface_serial_stage_mode == "pair_then_triplet" else (False if self.surface_serial_stage_mode == "perresidue_then_pair" else self.pair_gaussian_only and core_idx == 0 and self.surface_bus_mode == "none")),
                        beta_override=pair_beta_override,
                        precomputed_pair_indices=(
                            surface_runtime_geometry.get("pair_indices")
                            if surface_runtime_geometry is not None else None
                        ),
                    )
                    if pair_beta_override is not None:
                        pair_diag[f"pair_gaussian_beta_round{core_idx}"] = pair_beta_override.detach()
                        if core_idx == 0:
                            pair_diag["pair_gaussian_beta_init"] = pair_beta_override.detach()
                        else:
                            pair_diag["pair_gaussian_beta_deep"] = pair_beta_override.detach()
                    pair_record = getattr(self, "_last_pair_gaussian_top_record", None)
                    if pair_record is not None:
                        pair_record = dict(pair_record)
                        pair_record.update({"sample_index": int(sample_idx), "core_round": int(core_idx), "serial_stage": int(stage_idx)})
                        for _rk in ("residue_i", "residue_j", "residue_k"):
                            if _rk in pair_record:
                                pair_record[_rk + "_global"] = int(pair_record[_rk]) + int(residue_offset)
                        gaussian_top_records.append(pair_record)
                    for key, value in pair_diag.items():
                        value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        sigma_diag_values.setdefault(key, []).append(value)
                triplet_stage_active = core_idx == 0 or self.triplet_gaussian_fusion_mode == "deep"
                if self.surface_serial_stage_mode == "pair_then_triplet":
                    triplet_stage_active = stage_idx == 1
                elif self.surface_serial_stage_mode == "perresidue_then_pair":
                    triplet_stage_active = False
                if self.triplet_gaussian is not None and triplet_stage_active:
                    triplet_beta_override = None
                    if self.triplet_gaussian_beta_mode == "learned_deep":
                        if core_idx == 0:
                            triplet_beta_override = h_res.new_tensor(self.triplet_gaussian_beta_init)
                        elif self.triplet_gaussian_deep_beta_logits is not None:
                            beta_idx = min(core_idx - 1, int(self.triplet_gaussian_deep_beta_logits.numel()) - 1)
                            beta_logit = self.triplet_gaussian_deep_beta_logits[beta_idx].to(device=h_res.device, dtype=h_res.dtype)
                            triplet_beta_override = self.triplet_gaussian_beta_min + (
                                self.triplet_gaussian_beta_max - self.triplet_gaussian_beta_min
                            ) * pt.sigmoid(beta_logit)
                    h_surf_state, triplet_diag = self.triplet_gaussian(
                        h_res,
                        centers,
                        surface_pos,
                        surface_normal,
                        surface_geom,
                        h_surf_state,
                        beta_override=triplet_beta_override,
                    )
                    if triplet_beta_override is not None:
                        triplet_diag[f"triplet_gaussian_beta_round{core_idx}"] = triplet_beta_override.detach()
                        if core_idx == 0:
                            triplet_diag["triplet_gaussian_beta_init"] = triplet_beta_override.detach()
                        else:
                            triplet_diag["triplet_gaussian_beta_deep"] = triplet_beta_override.detach()
                    triplet_record = getattr(self.triplet_gaussian, "last_top_record", None)
                    if triplet_record is not None:
                        triplet_record = dict(triplet_record)
                        triplet_record.update({"sample_index": int(sample_idx), "core_round": int(core_idx), "serial_stage": int(stage_idx)})
                        for _rk in ("residue_i", "residue_j", "residue_k"):
                            if _rk in triplet_record:
                                triplet_record[_rk + "_global"] = int(triplet_record[_rk]) + int(residue_offset)
                        gaussian_top_records.append(triplet_record)
                    for key, value in triplet_diag.items():
                        value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        sigma_diag_values.setdefault(key, []).append(value)
                if self.pair_gaussian_multiview is not None and (
                    core_idx == 0 or self.pair_gaussian_multiview_mode == "deep"
                ):
                    h_surf_state, multiview_pair_diag = self._apply_multiview_pair_gaussian(
                        h_surf_state,
                        view_stack,
                        centers,
                        surface_pos,
                        surface_normal,
                        surface_geom,
                    )
                    for key, value in multiview_pair_diag.items():
                        value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        sigma_diag_values.setdefault(key, []).append(value)
                if self.surface_encoder_mode == "attention":
                    h_attn = h_surf_state + self.surface_attention_geom(surface_geom.to(device=h_surf_state.device, dtype=h_surf_state.dtype))
                    h_surf = self.surface_attention_encoder(h_attn.unsqueeze(0)).squeeze(0)
                else:
                    surface_initial_vectors = surface_vector_state if self.surface_vector_state_mode == "persistent" else None
                    if self.residue_vector_to_surface is not None and self.surface_encoder_mode == "gvp":
                        cross_vectors = self.residue_vector_to_surface(
                            h_surf_state,
                            h_res,
                            residue_vectors,
                            surface_pos,
                            surface_normal,
                            surface_geom,
                            centers,
                        )
                        if cross_vectors is not None:
                            surface_initial_vectors = cross_vectors
                        for key, value in getattr(self.residue_vector_to_surface, "last_diagnostics", {}).items():
                            value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                            sigma_diag_values.setdefault(key, []).append(value)
                    if self.surface_extra_graph_ks:
                        h_surf = self._encode_surface_multigraph(
                            h_surf_state,
                            surface_geom,
                            surface_pos,
                            surface_normal,
                            cache,
                            surface_initial_vectors=surface_initial_vectors,
                            graph_bundle=surface_graph_bundle,
                        )
                        for key, value in getattr(self, "last_surface_graph_diagnostics", {}).items():
                            value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                            sigma_diag_values.setdefault(key, []).append(value)
                    elif self.surface_encoder_mode == "gvp":
                        surface_graph_idx, surface_graph_dist = self._surface_graph_for_encoder(cache, surface_pos)
                        h_surf = self.surface_encoder(
                            h_surf_state,
                            surface_geom,
                            surface_pos,
                            surface_normal,
                            surface_graph_idx,
                            graph_dist=surface_graph_dist,
                            initial_vectors=surface_initial_vectors,
                        )
                    else:
                        surface_graph_idx, surface_graph_dist = self._surface_graph_for_encoder(cache, surface_pos)
                        h_surf = self.surface_encoder(
                            h_surf_state,
                            surface_geom,
                            surface_pos,
                            surface_normal,
                            surface_graph_idx,
                            graph_dist=surface_graph_dist,
                        )
                if self.surface_spectral_filter is not None:
                    h_surf = self.surface_spectral_filter(h_surf, surface_lappe)
                surface_vectors = None
                if self.surface_vector_bridge_mode != "none" and self.surface_encoder_mode == "gvp":
                    surface_vectors = getattr(self.surface_encoder, "last_surface_vectors", None)
                    if self.surface_vector_state_mode == "persistent":
                        surface_vector_state = surface_vectors
                patch_diag = {}
                patch_stage_active = self.surface_serial_stage_mode == "none" or stage_idx == 1
                if patch_stage_active and self.surface_patch_cluster is not None and self.surface_patch_cluster.mode == "gate":
                    h_surf, _unused_context, patch_diag = self.surface_patch_cluster(
                        h_res,
                        h_surf,
                        surface_pos,
                        surface_normal,
                        surface_geom,
                        rs_idx,
                        rs_weight,
                        surf_context=None,
                        patch_state=patch_state if self.patch_state_mode == "persistent" else None,
                        surface_lappe=surface_lappe,
                    )
                    if self.patch_state_mode == "persistent" and "patch_tokens" in patch_diag:
                        patch_state = patch_diag["patch_tokens"]
                if self.struct_community is not None:
                    h_res, h_surf = self.struct_community(
                        h_res,
                        h_surf,
                        centers,
                        surface_pos,
                        surface_normal,
                    )
                    for key, value in getattr(self.struct_community, "last_diagnostics", {}).items():
                        value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        sigma_diag_values.setdefault(key, []).append(value)
                surface_phys_feat = None
                if phys_feat is not None:
                    surface_phys_feat = (phys_feat[rs_idx.to(device=h_res.device, dtype=pt.long)] * rs_weight.to(device=h_res.device, dtype=h_res.dtype).unsqueeze(-1)).sum(dim=1)
                if self.surface_global_feedback is not None:
                    h_surf_before_global = h_surf
                    surface_mask = h_surf.new_ones((1, h_surf.shape[0]))
                    h_surf_global = h_surf.unsqueeze(0)
                    for _ in range(self.surface_global_feedback_rounds):
                        h_surf_global = self.surface_global_feedback(h_surf_global, surface_mask)
                    h_surf = h_surf_global.squeeze(0)
                    update_norm = (h_surf - h_surf_before_global).norm(dim=-1).mean()
                    sigma_diag_values.setdefault("surface_global_feedback_update_norm", []).append(
                        update_norm.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                    )
                surf_context, surf_attn = self.surface_to_residue(
                    h_res,
                    h_surf,
                    centers,
                    surface_pos,
                    surface_normal,
                    residue_phys_features=phys_feat,
                    surface_phys_features=surface_phys_feat,
                    surface_vectors=surface_vectors,
                    residue_vectors=residue_vectors,
                    geometry_cache=(
                        surface_runtime_geometry.get("surface_to_residue")
                        if surface_runtime_geometry is not None else None
                    ),
                )
                for key, value in getattr(self.surface_to_residue, "last_diagnostics", {}).items():
                    value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                    sigma_diag_values.setdefault(key, []).append(value)
                if (
                    self.surface_vector_bridge_mode == "equivariant"
                    and residue_vectors is not None
                    and getattr(self.surface_to_residue, "last_residue_vector_update", None) is not None
                ):
                    residue_vectors = residue_vectors + self.surface_to_residue.last_residue_vector_update
                    vector_scalar = self.residue_vector_scalar_proj(pt.linalg.norm(residue_vectors, dim=-1))
                    vector_gate = self.residue_vector_scalar_gate(pt.cat([h_res, vector_scalar], dim=-1))
                    h_res = h_res + vector_gate * vector_scalar
                if patch_stage_active and self.surface_patch_cluster is not None and self.surface_patch_cluster.mode in {"attn", "self_attn", "replace_attn", "independent_attn", "connected_pool", "connected_pool_adj"}:
                    if self.surface_patch_cluster.connected_mode and connected_patch_partition is None:
                        connected_graph_idx, _connected_graph_dist = self._surface_graph_for_k(
                            cache, surface_pos, self.surface_graph_k
                        )
                        connected_patch_partition = self.surface_patch_cluster.connected_partition(
                            surface_pos, connected_graph_idx
                        )
                    h_surf, surf_context, patch_diag = self.surface_patch_cluster(
                        h_res,
                        h_surf,
                        surface_pos,
                        surface_normal,
                        surface_geom,
                        rs_idx,
                        rs_weight,
                        surf_context=surf_context,
                        patch_state=patch_state if self.patch_state_mode == "persistent" else None,
                        surface_lappe=surface_lappe,
                        connected_partition=connected_patch_partition,
                    )
                    if self.patch_state_mode == "persistent" and "patch_tokens" in patch_diag:
                        patch_state = patch_diag["patch_tokens"]
                    if self.motif_patch is not None and "patch_tokens" in patch_diag:
                        h_res, surf_context, motif_patch_diag = self.motif_patch(
                            h_res,
                            centers,
                            patch_diag["patch_tokens"],
                            surf_context=surf_context,
                        )
                        patch_diag.update(motif_patch_diag)
                        for key, value in getattr(self.motif_patch, "last_diagnostics", {}).items():
                            value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                            sigma_diag_values.setdefault(key, []).append(value)
                if core_idx + 1 < update_rounds:
                    if self.surface_bus_mode == "recurrent":
                        view_states, bus_recurrent_diag = self._advance_surface_bus_view_states(
                            view_states,
                            surf_context,
                            residue_vectors,
                            centers,
                            graph_cache=graph_cache,
                            true_feat=true_feat,
                            chain_ids=chain_ids,
                        )
                        for key, value in bus_recurrent_diag.items():
                            value = value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                            sigma_diag_values.setdefault(key, []).append(value)
                        h_res, router_map, router_entropy = self._fuse_view_states(view_states, h_query=h_res)
                        sigma_diag_values.setdefault("surface_bus_view_router_entropy", []).append(
                            router_entropy.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        )
                    else:
                        core_input = pt.cat([h_res, surf_context], dim=-1)
                        h_res = self.core_norm(h_res + self.core_residue_gate(core_input) * self.core_residue_update(core_input))
                    if self.surface_round_end_r2s_mode == "skip" or self.surface_serial_stage_mode != "none":
                        h_surf_state = h_surf
                    elif self.surface_co_update_mode == "bidir":
                        surface_context, _surface_cross_attn = self.residue_to_surface(
                            h_surf,
                            h_res,
                            surface_pos,
                            surface_normal,
                            centers,
                        )
                        surface_input = pt.cat([h_surf, surface_context], dim=-1)
                        h_surf_state = self.surface_cross_norm(
                            h_surf + self.surface_cross_gate(surface_input) * self.surface_cross_update(surface_input)
                        )
                    else:
                        h_surf_state = h_surf
            if self.post_final_residue_global_feedback is not None:
                h_res_post = h_res.unsqueeze(0)
                post_mask = h_res.new_ones((1, h_res.shape[0]))
                h_res_before_post = h_res
                for _ in range(self.post_final_residue_global_feedback_rounds):
                    h_res_post = self.post_final_residue_global_feedback(h_res_post, post_mask)
                h_res = h_res_post.squeeze(0)
                sigma_diag_values.setdefault("post_final_residue_global_feedback_update_norm", []).append(
                    (h_res - h_res_before_post).norm(dim=-1).mean().detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
            final_res_skip_pre_h = None
            if self.final_residue_skip_mode == "gated":
                skip_source = h_res_multiview_skip if self.final_residue_skip_source == "multiview" else None
                if skip_source is not None:
                    z = self.final_residue_skip_proj(skip_source.to(device=h_res.device, dtype=h_res.dtype))
                    skip_gate = pt.sigmoid(self.final_residue_skip_gate(pt.cat([h_res, z, h_res * z, h_res - z], dim=-1)))
                    skip_update = self.final_residue_skip_beta * skip_gate * z
                    final_res_skip_pre_h = h_res
                    h_res = self.final_residue_skip_norm(h_res + skip_update)
                    skip_update_norm = skip_update.norm(dim=-1)
                    denom = final_res_skip_pre_h.norm(dim=-1).clamp_min(1e-6) * z.norm(dim=-1).clamp_min(1e-6)
                    skip_cos = ((final_res_skip_pre_h * z).sum(dim=-1) / denom).clamp(-1.0, 1.0)
                    sigma_diag_values.setdefault("final_res_skip_gate_mean", []).append(
                        skip_gate.detach().to(device=h_res.device, dtype=pt.float32).mean().reshape(())
                    )
                    sigma_diag_values.setdefault("final_res_skip_gate_std", []).append(
                        skip_gate.detach().to(device=h_res.device, dtype=pt.float32).std(unbiased=False).reshape(())
                    )
                    sigma_diag_values.setdefault("final_res_skip_update_norm", []).append(
                        skip_update_norm.detach().to(device=h_res.device, dtype=pt.float32).mean().reshape(())
                    )
                    sigma_diag_values.setdefault("final_res_skip_cos_to_final", []).append(
                        skip_cos.detach().to(device=h_res.device, dtype=pt.float32).mean().reshape(())
                    )
            surf_logit = self.surface_head(h_surf)
            base_logit = self.base_head(h_res)
            delta = self.final_delta(pt.cat([h_res, surf_context], dim=-1)).squeeze(-1)
            gate = self.surface_gate(pt.cat([h_res, surf_context], dim=-1)).squeeze(-1)
            if self.surface_fusion_mode == "same_level":
                fused_h = self.same_level_fusion(pt.cat([h_res, surf_context], dim=-1))
                final_logit = self.same_level_head(fused_h)
            else:
                final_logit = base_logit + self.surface_delta_scale * gate * delta
            if final_res_skip_pre_h is not None:
                if self.surface_fusion_mode == "same_level":
                    pre_fused_h = self.same_level_fusion(pt.cat([final_res_skip_pre_h, surf_context], dim=-1))
                    pre_final_logit = self.same_level_head(pre_fused_h)
                else:
                    pre_base_logit = self.base_head(final_res_skip_pre_h)
                    pre_delta = self.final_delta(pt.cat([final_res_skip_pre_h, surf_context], dim=-1)).squeeze(-1)
                    pre_gate = self.surface_gate(pt.cat([final_res_skip_pre_h, surf_context], dim=-1)).squeeze(-1)
                    pre_final_logit = pre_base_logit + self.surface_delta_scale * pre_gate * pre_delta
                delta_logit_abs = (final_logit - pre_final_logit).detach().abs().to(device=h_res.device, dtype=pt.float32)
                sigma_diag_values.setdefault("final_res_skip_delta_logit_abs", []).append(delta_logit_abs.mean().reshape(()))
            surface_branch_logit = final_logit
            if self.stochastic_head_mode == "ctmc_poisson":
                h_before_ctmc = h_res
                h_res, ctmc_diag = self._ctmc_diffuse_residue(h_res, centers)
                for key, value in ctmc_diag.items():
                    sigma_diag_values.setdefault(key, []).append(value.to(device=h_res.device, dtype=pt.float32).reshape(()))
                ctmc_base_logit = self.base_head(h_res)
                ctmc_delta = self.final_delta(pt.cat([h_res, surf_context], dim=-1)).squeeze(-1)
                ctmc_gate = self.surface_gate(pt.cat([h_res, surf_context], dim=-1)).squeeze(-1)
                if self.surface_fusion_mode == "same_level":
                    ctmc_fused_h = self.same_level_fusion(pt.cat([h_res, surf_context], dim=-1))
                    final_logit = self.same_level_head(ctmc_fused_h)
                else:
                    final_logit = ctmc_base_logit + self.surface_delta_scale * ctmc_gate * ctmc_delta
                base_logit = ctmc_base_logit
                gate = ctmc_gate
                delta = ctmc_delta
                surface_branch_logit = final_logit
                sigma_diag_values.setdefault("ctmc_logit_delta_abs", []).append(
                    (final_logit - base_logit).detach().abs().mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                sigma_diag_values.setdefault("ctmc_hidden_delta_norm", []).append(
                    (h_res - h_before_ctmc).detach().norm(dim=-1).mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )

            if self.post_final_fourview_mode == "refine_dual_env":
                graph_refined_b, post_entropy = self._interleaved_residue_graph_refine(
                    h_res.unsqueeze(0),
                    residue_vectors,
                    centers,
                    graph_cache=graph_cache,
                    true_feat=true_feat,
                    chain_ids=chain_ids,
                )
                graph_refined = graph_refined_b.squeeze(0)
                graph_env = graph_refined - h_res
                graph_branch = h_res + self.post_final_fourview_beta * graph_env
                graph_logit = self.post_final_graph_head(graph_branch)
                dual_gate = pt.softmax(
                    self.post_final_dual_gate(pt.cat([h_res, surf_context, graph_env], dim=-1)),
                    dim=-1,
                )
                final_logit = dual_gate[:, 0] * surface_branch_logit + dual_gate[:, 1] * graph_logit
                sigma_diag_values.setdefault("post_final_fourview_router_entropy", []).append(
                    post_entropy.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                sigma_diag_values.setdefault("post_final_fourview_update_norm", []).append(
                    graph_env.detach().norm(dim=-1).mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                sigma_diag_values.setdefault("post_final_dual_surface_gate_mean", []).append(
                    dual_gate[:, 0].detach().mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                sigma_diag_values.setdefault("post_final_dual_graph_gate_mean", []).append(
                    dual_gate[:, 1].detach().mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                sigma_diag_values.setdefault("post_final_graph_logit_abs", []).append(
                    graph_logit.detach().abs().mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
                sigma_diag_values.setdefault("post_final_surface_logit_abs", []).append(
                    surface_branch_logit.detach().abs().mean().to(device=h_res.device, dtype=pt.float32).reshape(())
                )
            if self.semantic_patch_head is not None:
                required_patch_keys = {
                    "patch_tokens", "patch_positions", "patch_assignment", "patch_adjacency",
                    "patch_surface_graph_idx",
                }
                missing_patch_keys = required_patch_keys.difference(patch_diag)
                if missing_patch_keys:
                    raise RuntimeError(
                        "semantic patch head missing connected patch outputs: "
                        + ", ".join(sorted(missing_patch_keys))
                    )
                semantic_output = self.semantic_patch_head(
                    h_res=h_res,
                    h_surf=h_surf,
                    centers=centers,
                    surface_pos=surface_pos,
                    surface_normal=surface_normal,
                    surface_graph_idx=patch_diag["patch_surface_graph_idx"],
                    patch_tokens=patch_diag["patch_tokens"],
                    patch_pos=patch_diag["patch_positions"],
                    assign=patch_diag["patch_assignment"],
                    adjacency=patch_diag["patch_adjacency"],
                    rs_idx=rs_idx,
                    rs_weight=rs_weight,
                )
                final_logit = final_logit + semantic_output["residual"]
                semantic_patch_outputs.append(semantic_output)
                sigma_diag_values.setdefault("semantic_patch_residual_abs", []).append(
                    semantic_output["residual"].detach().abs().mean().float().reshape(())
                )
                for semantic_key in (
                    "semantic_residual_beta",
                    "semantic_active_region_count",
                    "semantic_active_region_mass",
                    "semantic_region_existence_mean",
                    "semantic_region_amplitude_mean",
                    "semantic_boundary_fraction",
                    "semantic_boundary_shift",
                    "semantic_boundary_tv",
                ):
                    semantic_value = semantic_output.get(semantic_key)
                    if semantic_value is not None:
                        sigma_diag_values.setdefault(semantic_key, []).append(
                            semantic_value.detach().float().mean().reshape(())
                        )
            final_logit, stochastic_diag = self._stochastic_transform_logits(final_logit, h_res)
            for key, value in stochastic_diag.items():
                sigma_diag_values.setdefault(key + "_mean", []).append(value.float().mean().to(device=h_res.device).reshape(()))
                sigma_diag_values.setdefault(key + "_min", []).append(value.float().min().to(device=h_res.device).reshape(()))
                sigma_diag_values.setdefault(key + "_max", []).append(value.float().max().to(device=h_res.device).reshape(()))
                sigma_diag_values.setdefault(key + "_std", []).append(value.float().std(unbiased=False).to(device=h_res.device).reshape(()))

            # Diagnostic pooled surface score, using the same attention neighborhood
            # as the surface-to-residue context.
            if surface_runtime_geometry is not None and "surface_to_residue" in surface_runtime_geometry:
                idx = surface_runtime_geometry["surface_to_residue"]["idx"]
            else:
                pooled_dist = pt.cdist(centers, surface_pos).clamp_min(0.0)
                pooled_k = min(self.surface_to_residue.surface_topk, surface_pos.shape[0])
                idx = pt.topk(pooled_dist, k=pooled_k, dim=-1, largest=False).indices
            pooled = (surf_logit[idx] * surf_attn).sum(dim=-1)

            logits.append(final_logit)
            prior_logits.append(prior_logit)
            base_logits.append(base_logit)
            aux_logits.append(self.aux_head(h_local).squeeze(0))
            surface_logits.append(surf_logit)
            surface_pooled_logits.append(pooled)
            gate_values.append(gate.detach())
            delta_values.append(delta.detach())
            if return_embeddings or self.stochastic_head_mode == "poisson_pocket":
                residue_embeddings.append(h_res)
                surface_embeddings.append(h_surf)
            if patch_diag:
                patch_entropies.append(patch_diag["patch_assignment_entropy"].detach())
                patch_active_counts.append(patch_diag["patch_usage_active"].detach())
                patch_usage_values.append(patch_diag["patch_usage"].detach())
                if "patch_surface_gate" in patch_diag:
                    patch_surface_gates.append(patch_diag["patch_surface_gate"].detach())
                if "patch_gate" in patch_diag:
                    patch_gate_values.append(patch_diag["patch_gate"].detach())
                for key, value in patch_diag.items():
                    if key in {
                        "patch_assignment_entropy", "patch_usage", "patch_usage_active",
                        "patch_surface_gate", "patch_gate", "patch_tokens", "patch_positions",
                        "patch_assignment", "patch_adjacency", "patch_token_primary_residue_idx",
                    }:
                        continue
                    if isinstance(value, pt.Tensor) and value.numel() == 1:
                        sigma_diag_values.setdefault(key, []).append(
                            value.detach().to(device=h_res.device, dtype=pt.float32).reshape(())
                        )
                patch_primary_residue_idx.append(rs_idx[:, 0].detach().to(device=h_res.device, dtype=pt.long) + residue_offset)
                if "motif_patch_motif_logits" in patch_diag:
                    motif_patch_motif_logits.append(patch_diag["motif_patch_motif_logits"])
                    motif_patch_motif_primary_idx.append(
                        patch_diag["motif_patch_motif_primary_idx"].to(device=h_res.device, dtype=pt.long) + residue_offset
                    )
                if "motif_patch_patch_logits" in patch_diag and "patch_token_primary_residue_idx" in patch_diag:
                    motif_patch_patch_logits.append(patch_diag["motif_patch_patch_logits"])
                    motif_patch_patch_primary_idx.append(
                        patch_diag["patch_token_primary_residue_idx"].to(device=h_res.device, dtype=pt.long) + residue_offset
                    )
            router_maps.append(router_map.detach() if router_map is not None else pt.empty(0, device=h_res.device))
            router_entropies.append(router_entropy.detach() if router_entropy is not None else pt.tensor(0.0, device=h_res.device))
            last_weights = weights
            atom_offset += atom_count
            residue_offset += residue_count
        out = {
            "logits": pt.cat(logits, dim=0),
            "prior_logits": pt.cat(prior_logits, dim=0),
            "surface_conditioned_logits": pt.cat(base_logits, dim=0),
            "base_logits": pt.cat(base_logits, dim=0),
            "surface_logits": pt.cat(surface_logits, dim=0),
            "surface_pooled_logits": pt.cat(surface_pooled_logits, dim=0),
            "surface_gate": pt.cat(gate_values, dim=0),
            "surface_delta": pt.cat(delta_values, dim=0),
            "view_weights": last_weights,
            "view_router": pt.cat([x for x in router_maps], dim=0) if router_maps and router_maps[0].numel() else None,
            "view_entropy": pt.stack(router_entropies).mean() if router_entropies else None,
            "view_names": getattr(self.encoder, "serial_view_order", self.encoder.graph_views),
        }
        if atom_pre_aux_logits:
            out["atom_pre_logits"] = pt.cat(atom_pre_aux_logits, dim=0)
        if patch_entropies:
            out["patch_assignment_entropy"] = pt.cat(patch_entropies, dim=0)
            out["patch_usage_active"] = pt.stack(patch_active_counts).float()
            out["patch_usage"] = pt.cat(patch_usage_values, dim=0)
            if patch_surface_gates:
                out["patch_surface_gate"] = pt.cat(patch_surface_gates, dim=0)
            if patch_gate_values:
                out["patch_gate"] = pt.cat(patch_gate_values, dim=0)
            out["patch_primary_residue_idx"] = pt.cat(patch_primary_residue_idx, dim=0)
        if motif_patch_motif_logits:
            out["motif_patch_motif_logits"] = pt.cat(motif_patch_motif_logits, dim=0)
            out["motif_patch_motif_primary_idx"] = pt.cat(motif_patch_motif_primary_idx, dim=0)
        if motif_patch_patch_logits:
            out["motif_patch_patch_logits"] = pt.cat(motif_patch_patch_logits, dim=0)
            out["motif_patch_patch_primary_idx"] = pt.cat(motif_patch_patch_primary_idx, dim=0)
        if semantic_patch_outputs:
            out["semantic_patch_outputs"] = semantic_patch_outputs
        if gaussian_top_records:
            out["gaussian_top_records"] = gaussian_top_records
        for key, values in sigma_diag_values.items():
            out[key] = pt.stack(values).mean()
        if return_embeddings or self.stochastic_head_mode == "poisson_pocket":
            out["residue_embeddings"] = pt.cat(residue_embeddings, dim=0)
            out["surface_embeddings"] = pt.cat(surface_embeddings, dim=0)
        return out
