import math

import torch as pt

from apex_model import FullGVPMessageLayer, GVPTupleLayerNorm, masked_mean


def parse_k_schedule(text_or_values):
    if isinstance(text_or_values, str):
        return tuple(int(x) for x in text_or_values.split(",") if str(x).strip())
    return tuple(int(x) for x in text_or_values)


class GenericGVPEncoder(pt.nn.Module):
    """Receptor-only multi-scale residue GVP encoder.

    This model predicts generic interface propensity from the apo/AF2 receptor
    structure alone. It deliberately has no partner input and no pair-level loss.
    """

    def __init__(
        self,
        atom_feature_dim,
        dim=128,
        vector_dim=16,
        k_schedule=(16, 16, 24, 24, 32, 32, 48, 48),
        rbf_bins=16,
        dropout=0.1,
        aux_layer=4,
    ):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.dim = int(dim)
        self.vector_dim = int(vector_dim)
        self.k_schedule = parse_k_schedule(k_schedule)
        if not self.k_schedule:
            raise ValueError("k_schedule must contain at least one k.")
        self.aux_layer = int(aux_layer)
        self.rbf_bins = int(rbf_bins)
        centers = pt.linspace(0.0, 30.0, rbf_bins)
        self.register_buffer("rbf_centers", centers)
        self.node_scalar_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(atom_feature_dim + 6),
            pt.nn.Linear(atom_feature_dim + 6, dim),
            pt.nn.SiLU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.node_vector_proj = pt.nn.Linear(4, vector_dim, bias=False)
        self.edge_scalar_proj = pt.nn.Sequential(
            pt.nn.Linear(rbf_bins + 2, dim),
            pt.nn.SiLU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.edge_vector_proj = pt.nn.Linear(1, vector_dim, bias=False)
        self.input_norm = GVPTupleLayerNorm(dim, vector_dim)
        self.layers = pt.nn.ModuleList(
            [FullGVPMessageLayer((dim, vector_dim), (dim, vector_dim), dropout=dropout) for _ in self.k_schedule]
        )
        self.local_out = pt.nn.Sequential(pt.nn.LayerNorm(dim + vector_dim), pt.nn.Linear(dim + vector_dim, dim))

    def centers_from_atom_mask(self, X, M):
        weights = M.to(dtype=X.dtype, device=X.device)
        denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return weights.transpose(0, 1).matmul(X) / denom

    def rbf(self, dist):
        widths = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        return pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(dist.device)) / widths) ** 2)

    def aggregate_atom_features(self, q_atom, M):
        weights = M.to(dtype=q_atom.dtype, device=q_atom.device)
        denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        mean_feat = weights.transpose(0, 1).matmul(q_atom) / denom
        atom_count = weights.sum(dim=0).log1p().unsqueeze(-1)
        has_atoms = (weights.sum(dim=0) > 0).to(dtype=q_atom.dtype).unsqueeze(-1)
        atom_offset = self.atom_feature_dim - 64
        if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
            bb = weights.transpose(0, 1).matmul(q_atom[:, atom_offset : atom_offset + 4]).clamp_max(1.0)
        else:
            bb = q_atom.new_zeros(M.shape[1], 4)
        bb_quality = bb.mean(dim=-1, keepdim=True)
        return pt.cat([mean_feat, atom_count, has_atoms, bb_quality, bb[:, :3]], dim=-1)

    def backbone_vectors(self, X, M, q_atom):
        centers = self.centers_from_atom_mask(X, M)
        n_res = M.shape[1]
        atom_offset = self.atom_feature_dim - 64
        vecs = []
        if atom_offset >= 0 and q_atom.shape[1] >= atom_offset + 4:
            for j in range(4):
                atom_w = M.to(dtype=X.dtype, device=X.device) * q_atom[:, atom_offset + j : atom_offset + j + 1].to(
                    dtype=X.dtype
                )
                denom = atom_w.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
                pos = atom_w.transpose(0, 1).matmul(X) / denom
                v = pos - centers
                present = (atom_w.sum(dim=0) > 0).to(dtype=X.dtype).unsqueeze(-1)
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                vecs.append(v * present)
        if not vecs:
            vecs = [X.new_zeros(n_res, 3) for _ in range(4)]
        return pt.stack(vecs, dim=-2), centers

    def cached_knn(self, n, device, graph_cache, k):
        if not graph_cache:
            return None, None
        idx = graph_cache.get(f"knn_idx_k{int(k)}")
        dist = graph_cache.get(f"knn_dist_k{int(k)}")
        if idx is None or dist is None:
            return None, None
        idx = idx.to(device=device, dtype=pt.long)
        dist = dist.to(device=device)
        if idx.shape[0] != n or dist.shape[0] != n:
            return None, None
        max_k = min(int(k), max(n - 1, 0), idx.shape[1], dist.shape[1])
        if max_k <= 0:
            return idx[:, :0], dist[:, :0]
        return idx[:, :max_k].contiguous(), dist[:, :max_k].contiguous()

    def build_edges(self, centers, graph_cache, k):
        n = centers.shape[0]
        if n <= 1:
            empty_idx = pt.empty((n, 0), dtype=pt.long, device=centers.device)
            empty_dist = centers.new_empty((n, 0))
            return empty_idx, empty_dist, centers.new_empty((n, 0, self.dim)), centers.new_empty((n, 0, self.vector_dim, 3))
        nn_idx, nn_dist = self.cached_knn(n, centers.device, graph_cache, k)
        if nn_idx is None or nn_dist is None:
            dist = pt.cdist(centers, centers).clamp_min(0.0)
            kk = min(int(k) + 1, n)
            nn_dist, nn_idx = pt.topk(dist, k=kk, dim=-1, largest=False)
            nn_dist = nn_dist[:, 1:]
            nn_idx = nn_idx[:, 1:]
        nn_dist = nn_dist.to(dtype=centers.dtype)
        direction = centers[nn_idx] - centers.unsqueeze(1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        edge_v = self.edge_vector_proj(direction.unsqueeze(-2).transpose(-1, -2)).transpose(-1, -2)
        rel = (nn_idx - pt.arange(n, device=centers.device).unsqueeze(-1)).float()
        rel_abs = pt.log1p(rel.abs()).unsqueeze(-1) / math.log(max(n, 2))
        rel_sign = rel.sign().unsqueeze(-1)
        edge_s = self.edge_scalar_proj(pt.cat([self.rbf(nn_dist), rel_abs, rel_sign], dim=-1))
        return nn_idx, nn_dist, edge_s, edge_v

    def forward(self, X, q_atom, M, graph_cache=None):
        scalar_node = self.aggregate_atom_features(q_atom, M)
        s = self.node_scalar_proj(scalar_node)
        base_vec, centers = self.backbone_vectors(X, M, q_atom)
        v = self.node_vector_proj(base_vec.transpose(-1, -2)).transpose(-1, -2)
        s, v = self.input_norm(s, v)
        aux_local = None
        for i, (layer, k) in enumerate(zip(self.layers, self.k_schedule), start=1):
            nn_idx, _nn_dist, edge_s, edge_v = self.build_edges(centers, graph_cache, k)
            if nn_idx.shape[1] > 0:
                s, v = layer(s, v, edge_s, edge_v, nn_idx)
            if i == self.aux_layer:
                aux_local = self.local_out(pt.cat([s, pt.linalg.norm(v, dim=-1)], dim=-1))
        local = self.local_out(pt.cat([s, pt.linalg.norm(v, dim=-1)], dim=-1))
        if aux_local is None:
            aux_local = local
        return local.unsqueeze(0), aux_local.unsqueeze(0)


class GlobalContextFeedback(pt.nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h, mask):
        mask = mask.to(dtype=h.dtype, device=h.device)
        mean = masked_mean(h, mask, dim=1)
        masked_h = h.masked_fill(mask.unsqueeze(-1) < 0.5, pt.finfo(h.dtype).min)
        maxv = masked_h.max(dim=1).values
        maxv = pt.where(pt.isfinite(maxv), maxv, pt.zeros_like(maxv))
        g = 0.5 * (mean + maxv)
        g_expand = g.unsqueeze(1).expand_as(h)
        x = pt.cat([h, g_expand, h * g_expand], dim=-1)
        return self.norm(h + self.gate(x) * self.update(x))


class InterfaceHead(pt.nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class GenericGVPPredictor(pt.nn.Module):
    def __init__(
        self,
        atom_feature_dim=123,
        dim=128,
        vector_dim=16,
        k_schedule=(16, 16, 24, 24, 32, 32, 48, 48),
        rbf_bins=16,
        dropout=0.1,
        aux_layer=4,
    ):
        super().__init__()
        self.encoder = GenericGVPEncoder(
            atom_feature_dim=atom_feature_dim,
            dim=dim,
            vector_dim=vector_dim,
            k_schedule=k_schedule,
            rbf_bins=rbf_bins,
            dropout=dropout,
            aux_layer=aux_layer,
        )
        self.global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.primary_head = InterfaceHead(dim=dim, dropout=dropout)
        self.aux_head = InterfaceHead(dim=dim, dropout=dropout)

    def forward(self, X, ids_topk, q0, M, atom_counts, residue_counts, residue_graph_caches=None):
        logits = []
        aux_logits = []
        atom_offset = 0
        residue_offset = 0
        for sample_idx in range(len(atom_counts)):
            atom_count = int(atom_counts[sample_idx].item())
            residue_count = int(residue_counts[sample_idx].item())
            x = X[atom_offset : atom_offset + atom_count]
            q = q0[atom_offset : atom_offset + atom_count]
            m = M[atom_offset : atom_offset + atom_count, residue_offset : residue_offset + residue_count]
            graph_cache = None
            if residue_graph_caches is not None:
                graph_cache = residue_graph_caches[sample_idx]
            h_local, h_aux = self.encoder(x, q, m, graph_cache=graph_cache)
            mask = pt.ones((1, residue_count), dtype=h_local.dtype, device=h_local.device)
            h_final = self.global_feedback(h_local, mask)
            logits.append(self.primary_head(h_final).squeeze(0))
            aux_logits.append(self.aux_head(h_aux).squeeze(0))
            atom_offset += atom_count
            residue_offset += residue_count
        return {
            "logits": pt.cat(logits, dim=0),
            "aux_logits": pt.cat(aux_logits, dim=0),
        }
