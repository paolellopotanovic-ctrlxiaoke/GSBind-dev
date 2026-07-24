"""Residue and atom encoders used by M4.

Mechanically extracted from the validated research implementation. Tensor
operations and module attribute names are intentionally unchanged.
"""

import inspect
import math

import torch as pt

from .residue_backbone import StaticMultiViewGVPEncoder


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
