import math

import torch as pt

from src.model_operations import StatePoolLayer, StateUpdateLayer, unpack_state_features


def masked_mean(x, mask, dim=1, eps=1e-6):
    mask = mask.to(dtype=x.dtype, device=x.device)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


class FeedForward(pt.nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.Linear(dim, dim * mult),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim * mult, dim),
            pt.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ResidualSelfAttention(pt.nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.norm_attn = pt.nn.LayerNorm(dim)
        self.attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = pt.nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, mask=None):
        key_padding_mask = None if mask is None else mask.to(device=x.device) < 0.5
        h = self.norm_attn(x)
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ffn(self.norm_ffn(x))
        if mask is not None:
            x = x * mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        return x


class ResidualCrossAttention(pt.nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.norm_q = pt.nn.LayerNorm(dim)
        self.norm_kv = pt.nn.LayerNorm(dim)
        self.attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = pt.nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query, key_value, key_mask=None, query_mask=None):
        key_padding_mask = None if key_mask is None else key_mask.to(device=query.device) < 0.5
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        query = query + out
        query = query + self.ffn(self.norm_ffn(query))
        if query_mask is not None:
            query = query * query_mask.to(dtype=query.dtype, device=query.device).unsqueeze(-1)
        return query


class CrossAttentionContext(pt.nn.Module):
    """Return cross-attention context without adding the query residual."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm_q = pt.nn.LayerNorm(dim)
        self.norm_kv = pt.nn.LayerNorm(dim)
        self.attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out = pt.nn.Sequential(pt.nn.Dropout(dropout), pt.nn.LayerNorm(dim))

    def forward(self, query, key_value, key_mask=None, query_mask=None):
        key_padding_mask = None if key_mask is None else key_mask.to(device=query.device) < 0.5
        out, _ = self.attn(
            self.norm_q(query),
            self.norm_kv(key_value),
            self.norm_kv(key_value),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.out(out)
        if query_mask is not None:
            out = out * query_mask.to(dtype=out.dtype, device=out.device).unsqueeze(-1)
        return out


class GatedUpdate(pt.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = pt.nn.Sequential(pt.nn.Linear(dim * 3, dim), pt.nn.Sigmoid())
        self.proj = pt.nn.Linear(dim, dim)
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, base, update, mask=None):
        gate = self.gate(pt.cat([base, update, base * update], dim=-1))
        out = self.norm(base + gate * self.proj(update))
        if mask is not None:
            out = out * mask.to(dtype=out.dtype, device=out.device).unsqueeze(-1)
        return out


class GatedFusion(pt.nn.Module):
    """Gate two same-shaped feature streams into one residual representation."""

    def __init__(self, dim):
        super().__init__()
        self.gate = pt.nn.Sequential(pt.nn.Linear(dim * 3, dim), pt.nn.Sigmoid())
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, base, extra, mask=None):
        gate = self.gate(pt.cat([base, extra, base * extra], dim=-1))
        out = self.norm(base + gate * extra)
        if mask is not None:
            out = out * mask.to(dtype=out.dtype, device=out.device).unsqueeze(-1)
        return out


ESM2_MODEL_SPECS = {
    "esm2_t6_8M_UR50D": {"layers": 6, "dim": 320, "params": "8M"},
    "esm2_t12_35M_UR50D": {"layers": 12, "dim": 480, "params": "35M"},
    "esm2_t30_150M_UR50D": {"layers": 30, "dim": 640, "params": "150M"},
    "esm2_t33_650M_UR50D": {"layers": 33, "dim": 1280, "params": "650M"},
    "esm2_t36_3B_UR50D": {"layers": 36, "dim": 2560, "params": "3B"},
    "esm2_t48_15B_UR50D": {"layers": 48, "dim": 5120, "params": "15B"},
}


class ProteinSequenceEncoder(pt.nn.Module):
    """Partner sequence encoder with local motifs and long-range context.

    The public interface deliberately stays token based. A frozen ESM/ProtT5
    adapter can later replace this module while keeping APEX interaction blocks
    unchanged.
    """

    def __init__(
        self,
        vocab_size=22,
        dim=256,
        hidden_dim=256,
        conv_layers=3,
        transformer_layers=3,
        heads=4,
        dropout=0.1,
        pad_idx=21,
    ):
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.embed = pt.nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_idx)
        self.local_blocks = pt.nn.ModuleList()
        for i in range(conv_layers):
            dilation = 2**i
            self.local_blocks.append(
                pt.nn.Sequential(
                    pt.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
                    pt.nn.GELU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    pt.nn.GELU(),
                )
            )
        layer = pt.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = pt.nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.norm = pt.nn.LayerNorm(hidden_dim)
        self.proj = pt.nn.Linear(hidden_dim, dim)

    def forward(self, seq_indices, partner_mask=None):
        mask = (seq_indices != self.pad_idx).to(dtype=pt.float32, device=seq_indices.device)
        if partner_mask is not None:
            mask = mask * partner_mask.to(dtype=mask.dtype, device=mask.device)
        x = self.embed(seq_indices)
        h = x.transpose(1, 2)
        for block in self.local_blocks:
            h = h + block(h)
        x = h.transpose(1, 2)
        key_padding_mask = mask < 0.5
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.proj(self.norm(x))
        return x * mask.unsqueeze(-1)


class ESM2PartnerEncoder(pt.nn.Module):
    """Frozen ESM2 partner encoder.

    This module is intentionally optional. If `fair-esm` is unavailable, the
    constructor raises an explicit error instead of silently falling back and
    pretending that ESM features are being used.
    """

    def __init__(
        self,
        dim=256,
        model_name="esm2_t33_650M_UR50D",
        freeze=True,
        max_length=1022,
        repr_layer=None,
    ):
        super().__init__()
        try:
            import esm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ESM2 partner encoder requested, but package `esm` is not installed. "
                "Install fair-esm on the training server or use --partner-encoder transformer."
            ) from exc
        if model_name not in ESM2_MODEL_SPECS:
            supported = ", ".join(sorted(ESM2_MODEL_SPECS))
            raise ValueError(f"Unsupported ESM2 model name: {model_name}. Supported: {supported}")
        if not hasattr(esm.pretrained, model_name):
            raise ValueError(f"fair-esm installation does not expose pretrained loader: {model_name}")
        loader = getattr(esm.pretrained, model_name)
        self.esm_model, self.alphabet = loader()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.freeze = bool(freeze)
        self.max_length = int(max_length) if max_length else 0
        if self.freeze:
            self.esm_model.eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
        self.repr_layer = int(repr_layer) if repr_layer is not None else ESM2_MODEL_SPECS[model_name]["layers"]
        esm_dim = self.esm_model.embed_dim
        self.proj = pt.nn.Sequential(pt.nn.LayerNorm(esm_dim), pt.nn.Linear(esm_dim, dim))
        self.idx_to_aa = {
            0: "A",
            1: "C",
            2: "D",
            3: "E",
            4: "F",
            5: "G",
            6: "H",
            7: "I",
            8: "K",
            9: "L",
            10: "M",
            11: "N",
            12: "P",
            13: "Q",
            14: "R",
            15: "S",
            16: "T",
            17: "V",
            18: "W",
            19: "Y",
            20: "X",
        }

    def tokens_to_strings(self, seq_indices, partner_mask):
        seqs = []
        for seq, mask in zip(seq_indices.detach().cpu().tolist(), partner_mask.detach().cpu().tolist()):
            chars = []
            for idx, m in zip(seq, mask):
                if m < 0.5:
                    continue
                chars.append(self.idx_to_aa.get(int(idx), "X"))
                if self.max_length and len(chars) >= self.max_length:
                    break
            seqs.append("".join(chars) or "X")
        return seqs

    def forward(self, seq_indices, partner_mask=None):
        if partner_mask is None:
            partner_mask = pt.ones_like(seq_indices, dtype=pt.float32)
        device = seq_indices.device
        seqs = self.tokens_to_strings(seq_indices, partner_mask)
        labels = [(f"partner_{i}", seq) for i, seq in enumerate(seqs)]
        _labels, _strs, tokens = self.batch_converter(labels)
        tokens = tokens.to(device)
        self.esm_model = self.esm_model.to(device)
        if self.freeze:
            self.esm_model.eval()
        with pt.no_grad() if self.freeze else pt.enable_grad():
            out = self.esm_model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer][:, 1:-1, :]
        max_len = seq_indices.shape[1]
        if reps.shape[1] < max_len:
            pad = reps.new_zeros(reps.shape[0], max_len - reps.shape[1], reps.shape[2])
            reps = pt.cat([reps, pad], dim=1)
        reps = reps[:, :max_len, :]
        mask = partner_mask.to(device=device, dtype=reps.dtype)
        if self.max_length and mask.shape[1] > self.max_length:
            mask = mask.clone()
            mask[:, self.max_length :] = 0.0
        return self.proj(reps) * mask.unsqueeze(-1)


class CachedESM2PartnerEncoder(pt.nn.Module):
    """Projection head for precomputed ESM2 residue embeddings.

    It intentionally does not load the ESM2 model. The dataset supplies
    per-residue ESM vectors from disk, and this module only maps them into the
    APEX hidden dimension.
    """

    def __init__(self, dim=256, model_name="esm2_t33_650M_UR50D"):
        super().__init__()
        if model_name not in ESM2_MODEL_SPECS:
            supported = ", ".join(sorted(ESM2_MODEL_SPECS))
            raise ValueError(f"Unsupported ESM2 model name: {model_name}. Supported: {supported}")
        esm_dim = int(ESM2_MODEL_SPECS[model_name]["dim"])
        self.proj = pt.nn.Sequential(pt.nn.LayerNorm(esm_dim), pt.nn.Linear(esm_dim, dim))

    def forward(self, embeddings, partner_mask=None):
        out = self.proj(embeddings)
        if partner_mask is not None:
            out = out * partner_mask.to(dtype=out.dtype, device=out.device).unsqueeze(-1)
        return out


class ResidueGeometryRefiner(pt.nn.Module):
    """Residue graph refinement using CA-like centers from PeSTo atom masks.

    This is a scalar geometry-aware GNN block. It does not claim strict GVP
    equivariance, but it gives the APEX head explicit residue-distance message
    passing on top of the existing PeSTo geometric encoder.
    """

    def __init__(self, dim=256, k=24, rbf_bins=16, layers=2, dropout=0.1):
        super().__init__()
        self.k = int(k)
        self.rbf_bins = int(rbf_bins)
        centers = pt.linspace(0.0, 30.0, rbf_bins)
        self.register_buffer("rbf_centers", centers)
        self.edge_mlp = pt.nn.Sequential(
            pt.nn.Linear(rbf_bins + 1, dim),
            pt.nn.GELU(),
            pt.nn.Linear(dim, dim),
        )
        self.layers = pt.nn.ModuleList(
            [
                pt.nn.ModuleDict(
                    {
                        "msg": pt.nn.Sequential(
                            pt.nn.LayerNorm(dim * 3),
                            pt.nn.Linear(dim * 3, dim),
                            pt.nn.GELU(),
                            pt.nn.Dropout(dropout),
                            pt.nn.Linear(dim, dim),
                        ),
                        "gate": GatedUpdate(dim),
                        "self": ResidualSelfAttention(dim, heads=4, dropout=dropout),
                    }
                )
                for _ in range(layers)
            ]
        )

    def centers_from_atom_mask(self, X, M):
        weights = M.to(dtype=X.dtype, device=X.device)
        denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return weights.transpose(0, 1).matmul(X) / denom

    def rbf(self, dist):
        widths = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        return pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(dist.device)) / widths) ** 2)

    def forward(self, h, X, M):
        n = h.shape[1]
        if n <= 1:
            return h
        centers = self.centers_from_atom_mask(X, M)
        dist = pt.cdist(centers, centers).clamp_min(0.0)
        k = min(self.k + 1, n)
        nn_dist, nn_idx = pt.topk(dist, k=k, dim=-1, largest=False)
        nn_dist = nn_dist[:, 1:]
        nn_idx = nn_idx[:, 1:]
        rbf = self.rbf(nn_dist)
        rel_pos = (nn_idx - pt.arange(n, device=h.device).unsqueeze(-1)).abs().float().unsqueeze(-1)
        rel_pos = pt.log1p(rel_pos) / math.log(max(n, 2))
        edge = self.edge_mlp(pt.cat([rbf, rel_pos], dim=-1))

        x = h.squeeze(0)
        for layer in self.layers:
            neigh = x[nn_idx]
            base = x.unsqueeze(1).expand_as(neigh)
            msg_in = pt.cat([base, neigh, edge], dim=-1)
            msg = layer["msg"](msg_in).mean(dim=1)
            x = layer["gate"](x, msg)
            x = layer["self"](x.unsqueeze(0)).squeeze(0)
        return x.unsqueeze(0)


class GVPLayer(pt.nn.Module):
    """Small GVP-style scalar/vector update layer.

    It keeps scalar and vector channels separate, injects vector norms into
    scalar updates, and gates vector outputs by scalar context.
    """

    def __init__(self, scalar_dim, vector_dim):
        super().__init__()
        self.scalar = pt.nn.Sequential(
            pt.nn.LayerNorm(scalar_dim + vector_dim),
            pt.nn.Linear(scalar_dim + vector_dim, scalar_dim),
            pt.nn.SiLU(),
            pt.nn.Linear(scalar_dim, scalar_dim),
        )
        self.vector_mix = pt.nn.Linear(vector_dim, vector_dim, bias=False)
        self.vector_gate = pt.nn.Sequential(pt.nn.Linear(scalar_dim, vector_dim), pt.nn.Sigmoid())
        self.scalar_norm = pt.nn.LayerNorm(scalar_dim)

    def forward(self, s, v):
        v_mixed = self.vector_mix(v.transpose(-1, -2)).transpose(-1, -2)
        v_norm = pt.linalg.norm(v_mixed, dim=-1)
        ds = self.scalar(pt.cat([s, v_norm], dim=-1))
        s = self.scalar_norm(s + ds)
        gate = self.vector_gate(s).unsqueeze(-1)
        v = v + gate * v_mixed
        return s, v


class GVPStructureEncoder(pt.nn.Module):
    """GVP-style residue graph encoder built from receptor coordinates."""

    def __init__(self, dim=256, vector_dim=8, k=24, layers=3, rbf_bins=16, dropout=0.1):
        super().__init__()
        self.k = int(k)
        self.vector_dim = int(vector_dim)
        self.rbf_bins = int(rbf_bins)
        centers = pt.linspace(0.0, 30.0, rbf_bins)
        self.register_buffer("rbf_centers", centers)
        self.edge_scalar = pt.nn.Sequential(
            pt.nn.Linear(rbf_bins + 1, dim),
            pt.nn.SiLU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.node_init = pt.nn.Linear(dim, dim)
        self.layers = pt.nn.ModuleList([GVPLayer(dim, vector_dim) for _ in range(layers)])
        self.msg_scalar = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.LayerNorm(dim * 3),
                    pt.nn.Linear(dim * 3, dim),
                    pt.nn.SiLU(),
                    pt.nn.Dropout(dropout),
                    pt.nn.Linear(dim, dim),
                )
                for _ in range(layers)
            ]
        )
        self.out = pt.nn.Sequential(pt.nn.LayerNorm(dim + vector_dim), pt.nn.Linear(dim + vector_dim, dim))

    def centers_from_atom_mask(self, X, M):
        weights = M.to(dtype=X.dtype, device=X.device)
        denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return weights.transpose(0, 1).matmul(X) / denom

    def rbf(self, dist):
        widths = (self.rbf_centers[1] - self.rbf_centers[0]).abs().clamp_min(1e-3)
        return pt.exp(-((dist.unsqueeze(-1) - self.rbf_centers.to(dist.device)) / widths) ** 2)

    def initial_vectors(self, centers, nn_idx):
        n = centers.shape[0]
        neigh = centers[nn_idx]
        vec = neigh - centers.unsqueeze(1)
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if vec.shape[1] < self.vector_dim:
            reps = math.ceil(self.vector_dim / max(vec.shape[1], 1))
            vec = vec.repeat(1, reps, 1)
        return vec[:, : self.vector_dim, :]

    def forward(self, h, X, M):
        n = h.shape[1]
        if n <= 1:
            return h
        centers = self.centers_from_atom_mask(X, M)
        dist = pt.cdist(centers, centers).clamp_min(0.0)
        k = min(self.k + 1, n)
        nn_dist, nn_idx = pt.topk(dist, k=k, dim=-1, largest=False)
        nn_dist = nn_dist[:, 1:]
        nn_idx = nn_idx[:, 1:]
        rbf = self.rbf(nn_dist)
        rel_pos = (nn_idx - pt.arange(n, device=h.device).unsqueeze(-1)).abs().float().unsqueeze(-1)
        rel_pos = pt.log1p(rel_pos) / math.log(max(n, 2))
        edge = self.edge_scalar(pt.cat([rbf, rel_pos], dim=-1))

        s = self.node_init(h.squeeze(0))
        v = self.initial_vectors(centers, nn_idx)
        for layer, msg_scalar in zip(self.layers, self.msg_scalar):
            neigh_s = s[nn_idx]
            base_s = s.unsqueeze(1).expand_as(neigh_s)
            scalar_msg = msg_scalar(pt.cat([base_s, neigh_s, edge], dim=-1)).mean(dim=1)
            neigh_v = v[nn_idx].mean(dim=1)
            v_msg = neigh_v[:, : self.vector_dim, :]
            s, v = layer(s + scalar_msg, v + v_msg)
        v_norm = pt.linalg.norm(v, dim=-1)
        return self.out(pt.cat([s, v_norm], dim=-1)).unsqueeze(0)


class GVPTupleLayerNorm(pt.nn.Module):
    def __init__(self, scalar_dim, vector_dim, eps=1e-6):
        super().__init__()
        self.scalar_norm = pt.nn.LayerNorm(scalar_dim)
        self.vector_scale = pt.nn.Parameter(pt.ones(vector_dim))
        self.eps = float(eps)

    def forward(self, s, v):
        s = self.scalar_norm(s)
        if v.numel() == 0:
            return s, v
        rms = pt.sqrt(pt.mean(pt.sum(v * v, dim=-1), dim=-1, keepdim=True).clamp_min(self.eps))
        v = v / rms.unsqueeze(-1)
        v = v * self.vector_scale.view(*([1] * (v.dim() - 2)), -1, 1)
        return s, v


class GVP(pt.nn.Module):
    """Geometric Vector Perceptron block with scalar/vector channels."""

    def __init__(self, in_dims, out_dims, activations=(pt.nn.SiLU(), None), vector_gate=True, eps=1e-8):
        super().__init__()
        si, vi = in_dims
        so, vo = out_dims
        self.si, self.vi, self.so, self.vo = int(si), int(vi), int(so), int(vo)
        self.scalar_act, self.vector_act = activations
        self.vector_gate = bool(vector_gate)
        self.eps = float(eps)
        hidden_v = max(self.vi, self.vo, 1)
        self.wh = pt.nn.Linear(self.vi, hidden_v, bias=False) if self.vi > 0 else None
        self.wv = pt.nn.Linear(hidden_v, self.vo, bias=False) if self.vo > 0 else None
        scalar_in = self.si + hidden_v
        self.ws = pt.nn.Linear(scalar_in, self.so)
        self.wg = pt.nn.Linear(self.so, self.vo) if self.vector_gate and self.vo > 0 else None

    def forward(self, s, v):
        if self.vi > 0 and self.vo > 0:
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)
            vn = pt.linalg.norm(vh, dim=-1)
            s_out = self.ws(pt.cat([s, vn], dim=-1))
            v_out = self.wv(vh.transpose(-1, -2)).transpose(-1, -2)
        elif self.vi > 0:
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)
            vn = pt.linalg.norm(vh, dim=-1)
            s_out = self.ws(pt.cat([s, vn], dim=-1))
            v_out = v.new_zeros(*s.shape[:-1], self.vo, 3)
        else:
            extra = s.new_zeros(*s.shape[:-1], 1)
            s_out = self.ws(pt.cat([s, extra], dim=-1))
            v_out = s.new_zeros(*s.shape[:-1], self.vo, 3)
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
        if self.vo > 0:
            if self.vector_gate:
                gate = pt.sigmoid(self.wg(s_out)).unsqueeze(-1)
                v_out = v_out * gate
            elif self.vector_act is not None:
                v_out = self.vector_act(v_out)
        return s_out, v_out


class GVPDropout(pt.nn.Module):
    def __init__(self, scalar_p=0.1, vector_p=0.1):
        super().__init__()
        self.scalar_dropout = pt.nn.Dropout(scalar_p)
        self.vector_p = float(vector_p)

    def forward(self, s, v):
        s = self.scalar_dropout(s)
        if self.training and self.vector_p > 0 and v.numel() > 0:
            keep = (pt.rand(*v.shape[:-1], 1, device=v.device) > self.vector_p).to(dtype=v.dtype)
            v = v * keep / max(1.0 - self.vector_p, 1e-6)
        return s, v


class FullGVPMessageLayer(pt.nn.Module):
    """Full residue-level GVP message passing layer with edge scalar/vector features."""

    def __init__(self, node_dims, edge_dims, dropout=0.1):
        super().__init__()
        ns, nv = node_dims
        es, ev = edge_dims
        msg_in = (ns * 2 + es, nv * 2 + ev)
        self.msg = pt.nn.Sequential()
        self.msg_gvp1 = GVP(msg_in, (ns, nv))
        self.msg_norm1 = GVPTupleLayerNorm(ns, nv)
        self.msg_gvp2 = GVP((ns, nv), (ns, nv))
        self.msg_norm2 = GVPTupleLayerNorm(ns, nv)
        self.dropout = GVPDropout(dropout, dropout)
        self.node_norm = GVPTupleLayerNorm(ns, nv)
        self.ff_gvp1 = GVP((ns, nv), (ns * 2, nv * 2))
        self.ff_gvp2 = GVP((ns * 2, nv * 2), (ns, nv), activations=(None, None))
        self.ff_norm = GVPTupleLayerNorm(ns, nv)

    def forward(self, s, v, edge_s, edge_v, nn_idx):
        neigh_s = s[nn_idx]
        neigh_v = v[nn_idx]
        base_s = s.unsqueeze(1).expand_as(neigh_s)
        base_v = v.unsqueeze(1).expand_as(neigh_v)
        msg_s_in = pt.cat([base_s, neigh_s, edge_s], dim=-1)
        msg_v_in = pt.cat([base_v, neigh_v, edge_v], dim=-2)
        msg_s, msg_v = self.msg_gvp1(msg_s_in, msg_v_in)
        msg_s, msg_v = self.msg_norm1(msg_s, msg_v)
        msg_s, msg_v = self.msg_gvp2(msg_s, msg_v)
        msg_s = msg_s.mean(dim=1)
        msg_v = msg_v.mean(dim=1)
        msg_s, msg_v = self.dropout(msg_s, msg_v)
        s, v = self.node_norm(s + msg_s, v + msg_v)
        ff_s, ff_v = self.ff_gvp1(s, v)
        ff_s, ff_v = self.ff_gvp2(ff_s, ff_v)
        ff_s, ff_v = self.dropout(ff_s, ff_v)
        return self.ff_norm(s + ff_s, v + ff_v)


class FullResidueGVPEncoder(pt.nn.Module):
    """Residue-level full GVP-GNN receptor encoder.

    Node scalar: residue/atom composition and backbone presence.
    Node vector: local backbone/body directions when available.
    Edge scalar: RBF distance and relative sequence offset.
    Edge vector: unit direction from residue i to residue j.
    """

    def __init__(
        self,
        atom_feature_dim,
        dim=256,
        vector_dim=16,
        k=24,
        layers=4,
        rbf_bins=16,
        dropout=0.1,
    ):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.vector_dim = int(vector_dim)
        self.k = int(k)
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
            [FullGVPMessageLayer((dim, vector_dim), (dim, vector_dim), dropout=dropout) for _ in range(layers)]
        )
        self.out = pt.nn.Sequential(pt.nn.LayerNorm(dim + vector_dim), pt.nn.Linear(dim + vector_dim, dim))

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

    def cached_knn(self, n, device, graph_cache=None):
        if not graph_cache:
            return None, None
        idx = graph_cache.get(f"knn_idx_k{self.k}")
        dist = graph_cache.get(f"knn_dist_k{self.k}")
        if idx is None or dist is None:
            return None, None
        idx = idx.to(device=device, dtype=pt.long)
        dist = dist.to(device=device)
        if idx.shape[0] != n or dist.shape[0] != n:
            return None, None
        max_k = min(self.k, max(n - 1, 0), idx.shape[1], dist.shape[1])
        if max_k <= 0:
            return idx[:, :0], dist[:, :0]
        return idx[:, :max_k].contiguous(), dist[:, :max_k].contiguous()

    def forward(self, X, q_atom, M, graph_cache=None):
        n = M.shape[1]
        scalar_node = self.aggregate_atom_features(q_atom, M)
        s = self.node_scalar_proj(scalar_node)
        base_vec, centers = self.backbone_vectors(X, M, q_atom)
        v = self.node_vector_proj(base_vec.transpose(-1, -2)).transpose(-1, -2)
        s, v = self.input_norm(s, v)
        if n <= 1:
            return self.out(pt.cat([s, pt.linalg.norm(v, dim=-1)], dim=-1)).unsqueeze(0)
        nn_idx, nn_dist = self.cached_knn(n, X.device, graph_cache=graph_cache)
        if nn_idx is None or nn_dist is None:
            dist = pt.cdist(centers, centers).clamp_min(0.0)
            k = min(self.k + 1, n)
            nn_dist, nn_idx = pt.topk(dist, k=k, dim=-1, largest=False)
            nn_dist = nn_dist[:, 1:]
            nn_idx = nn_idx[:, 1:]
        nn_dist = nn_dist.to(dtype=centers.dtype)
        direction = centers[nn_idx] - centers.unsqueeze(1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        edge_v = self.edge_vector_proj(direction.unsqueeze(-2).transpose(-1, -2)).transpose(-1, -2)
        rel = (nn_idx - pt.arange(n, device=X.device).unsqueeze(-1)).float()
        rel_abs = pt.log1p(rel.abs()).unsqueeze(-1) / math.log(max(n, 2))
        rel_sign = rel.sign().unsqueeze(-1)
        edge_s = self.edge_scalar_proj(pt.cat([self.rbf(nn_dist), rel_abs, rel_sign], dim=-1))
        for layer in self.layers:
            s, v = layer(s, v, edge_s, edge_v, nn_idx)
        return self.out(pt.cat([s, pt.linalg.norm(v, dim=-1)], dim=-1)).unsqueeze(0)


class PureResidueGVPEncoder(pt.nn.Module):
    """Pure residue-level structure encoder for the main APEX architecture.

    This branch does not use PeSTo StateUpdateLayer/StatePoolLayer. It builds
    residue nodes from atom masks and atom-level one-hot features, then refines
    them with GVP-style residue graph message passing.
    """

    def __init__(self, atom_feature_dim, dim=256, vector_dim=8, k=24, layers=4, dropout=0.1):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.center_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(atom_feature_dim + 4),
            pt.nn.Linear(atom_feature_dim + 4, dim),
            pt.nn.SiLU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.LayerNorm(dim),
        )
        self.gvp = GVPStructureEncoder(
            dim=dim,
            vector_dim=vector_dim,
            k=k,
            layers=layers,
            dropout=dropout,
        )
        self.post = ResidualSelfAttention(dim, heads=4, dropout=dropout)

    def residue_atom_summary(self, q_atom, M):
        weights = M.to(dtype=q_atom.dtype, device=q_atom.device)
        denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        mean_feat = weights.transpose(0, 1).matmul(q_atom) / denom
        atom_count = weights.sum(dim=0).log1p().unsqueeze(-1)
        has_atoms = (weights.sum(dim=0) > 0).to(dtype=q_atom.dtype).unsqueeze(-1)

        # Backbone atom presence is a compact quality/geometry cue, not a PeSTo
        # encoder. q_atom contains element, residue name and atom-name one-hots.
        atom_name_offset = self.atom_feature_dim - 64
        if atom_name_offset >= 0 and q_atom.shape[1] >= atom_name_offset + 4:
            ca = q_atom[:, atom_name_offset + 0 : atom_name_offset + 1]
            n = q_atom[:, atom_name_offset + 1 : atom_name_offset + 2]
            c = q_atom[:, atom_name_offset + 2 : atom_name_offset + 3]
            o = q_atom[:, atom_name_offset + 3 : atom_name_offset + 4]
            bb = weights.transpose(0, 1).matmul(pt.cat([ca, n, c, o], dim=-1)).clamp_max(1.0)
            bb_quality = bb.mean(dim=-1, keepdim=True)
        else:
            bb_quality = has_atoms
        return pt.cat([mean_feat, atom_count, has_atoms, bb_quality, has_atoms * bb_quality], dim=-1)

    def forward(self, X, q_atom, M):
        node = self.residue_atom_summary(q_atom, M)
        h = self.center_proj(node).unsqueeze(0)
        h = self.gvp(h, X, M)
        residue_mask = pt.ones((1, h.shape[1]), device=h.device, dtype=h.dtype)
        return self.post(h, residue_mask)


class InteractionQueryGenerator(pt.nn.Module):
    """Generate partner-conditioned query tokens from whole-receptor context.

    The queries are deliberately not tied to a learned surface selector. They are
    conditioned by receptor and partner global summaries, then anchored to evenly
    sampled receptor residues so the query bank has a structural reference.
    """

    def __init__(self, dim=256, num_queries=4):
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.query_bank = pt.nn.Parameter(pt.randn(num_queries, dim) * 0.02)
        self.modulator = pt.nn.Sequential(
            pt.nn.Linear(dim * 4, dim * 2),
            pt.nn.GELU(),
            pt.nn.Linear(dim * 2, num_queries * dim * 2),
        )
        self.anchor_proj = pt.nn.Linear(dim, dim)
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h_all, partner, partner_mask, residue_mask=None):
        receptor_summary = h_all.mean(dim=1)
        partner_summary = masked_mean(partner, partner_mask, dim=1)
        context = pt.cat(
            [
                receptor_summary,
                partner_summary,
                receptor_summary * partner_summary,
                (receptor_summary - partner_summary).abs(),
            ],
            dim=-1,
        )
        gamma_beta = self.modulator(context).view(h_all.shape[0], self.num_queries, self.dim, 2)
        gamma = gamma_beta[..., 0]
        beta = gamma_beta[..., 1]
        q = self.query_bank.unsqueeze(0).expand(h_all.shape[0], -1, -1)
        q = q * (1.0 + gamma) + beta

        if h_all.shape[1] >= self.num_queries:
            anchor_idx = pt.linspace(0, h_all.shape[1] - 1, self.num_queries, device=h_all.device).long()
            anchors = h_all[:, anchor_idx, :]
        else:
            reps = math.ceil(self.num_queries / max(h_all.shape[1], 1))
            anchors = h_all.repeat(1, reps, 1)[:, : self.num_queries, :]
        return self.norm(q + self.anchor_proj(anchors))


class ApexInteractionBlock(pt.nn.Module):
    """One cycle of full-receptor and partner reciprocal interaction."""

    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.partner_self = ResidualSelfAttention(dim, heads, dropout)
        self.receptor_self_pre = ResidualSelfAttention(dim, heads, dropout)

        self.q_reads_partner = ResidualCrossAttention(dim, heads, dropout)
        self.q_reads_receptor = ResidualCrossAttention(dim, heads, dropout)
        self.q_self = ResidualSelfAttention(dim, heads, dropout)

        self.all_reads_q = ResidualCrossAttention(dim, heads, dropout)
        self.all_reads_partner = ResidualCrossAttention(dim, heads, dropout)
        self.all_self = ResidualSelfAttention(dim, heads, dropout)
        self.global_self = ResidualSelfAttention(dim, heads, dropout)
        self.all_reads_global = ResidualCrossAttention(dim, heads, dropout)

        self.all_gate = GatedUpdate(dim)
        self.all_partner_gate = GatedUpdate(dim)
        self.partner_gate = GatedUpdate(dim)
        self.global_gate = GatedUpdate(dim)

    def forward(self, h_all, partner, q_tokens, global_tokens, partner_mask, residue_mask):
        global_mask = pt.ones(
            (global_tokens.shape[0], global_tokens.shape[1]),
            device=global_tokens.device,
            dtype=global_tokens.dtype,
        )

        partner_refined = self.partner_self(partner, partner_mask)
        partner = self.partner_gate(partner, partner_refined, partner_mask)

        h_all = self.receptor_self_pre(h_all, residue_mask)

        receptor_summary = masked_mean(h_all, residue_mask, dim=1)
        partner_summary = masked_mean(partner, partner_mask, dim=1)
        observed_globals = pt.stack([receptor_summary, partner_summary], dim=1)
        global_update = self.global_self(global_tokens + observed_globals, global_mask)
        global_tokens = self.global_gate(global_tokens, global_update, global_mask)

        q_tokens = self.q_reads_receptor(q_tokens, h_all, residue_mask)
        q_tokens = self.q_reads_partner(q_tokens, partner, partner_mask)
        q_tokens = self.q_reads_receptor(q_tokens, global_tokens, global_mask)
        q_tokens = self.q_self(q_tokens)

        all_update = self.all_reads_q(h_all, q_tokens, query_mask=residue_mask)
        h_all = self.all_gate(h_all, all_update, residue_mask)
        all_update = self.all_reads_partner(h_all, partner, partner_mask, residue_mask)
        h_all = self.all_partner_gate(h_all, all_update, residue_mask)
        all_update = self.all_reads_global(h_all, global_tokens, global_mask, residue_mask)
        h_all = self.all_gate(h_all, all_update, residue_mask)
        h_all = self.all_self(h_all, residue_mask)
        return h_all, partner, q_tokens, global_tokens


class ResiduePartnerFusionHead(pt.nn.Module):
    """Fuse residue, partner and global contexts with attention before scoring."""

    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.prior_proj = pt.nn.Linear(1, dim)
        self.norm = pt.nn.LayerNorm(dim)
        self.attn = pt.nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(dim, mult=2, dropout=dropout)
        self.out = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )

    def forward(self, h_res, partner_context, query_context, receptor_global, partner_global, prior):
        bsz, n_res, dim = h_res.shape
        receptor_global = receptor_global.unsqueeze(1).expand(-1, n_res, -1)
        partner_global = partner_global.unsqueeze(1).expand(-1, n_res, -1)
        prior_token = self.prior_proj(prior.view(bsz, n_res, 1).detach())
        tokens = pt.stack(
            [
                h_res,
                partner_context,
                query_context,
                receptor_global,
                partner_global,
                prior_token,
                h_res * partner_context,
                (h_res - partner_context).abs(),
            ],
            dim=2,
        ).reshape(bsz * n_res, 8, dim)
        fused, _ = self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens), need_weights=False)
        residue_token = tokens[:, 0, :] + fused[:, 0, :]
        residue_token = residue_token + self.ffn(self.norm(residue_token))
        return self.out(residue_token).view(bsz, n_res)


class ApexNet(pt.nn.Module):
    """APEX-Net: partner-conditioned apo-to-holo interface predictor.

    Inputs:
      receptor structure tensors from PeSTo encoding
      partner amino-acid token sequence

    Outputs:
      logit_prior: receptor-only interface propensity
      logit_conditioned: partner-conditioned interface probability logits
    """

    def __init__(
        self,
        config,
        dim=256,
        num_queries=4,
        num_blocks=4,
        heads=4,
        surface_fraction=0.5,
        max_surface=512,
        min_surface=128,
        dropout=0.1,
        residue_graph_k=24,
        residue_graph_layers=2,
        structure_encoder="full_gvp",
        gvp_layers=3,
        gvp_vector_dim=8,
        partner_encoder="transformer",
        esm_model_name="esm2_t33_650M_UR50D",
        freeze_esm=True,
        esm_max_length=1022,
        esm_repr_layer=None,
        receptor_sequence_encoder="esm2",
        partner_dropout=0.0,
        use_cached_partner_esm=False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.surface_fraction = float(surface_fraction)
        self.max_surface = int(max_surface)
        self.min_surface = int(min_surface)
        self.partner_dropout = float(partner_dropout)
        self.structure_encoder_name = str(structure_encoder)
        self.partner_encoder_name = str(partner_encoder)
        self.receptor_sequence_encoder_name = str(receptor_sequence_encoder)
        self.use_cached_partner_esm = bool(use_cached_partner_esm)

        self.use_pesto_backbone = self.structure_encoder_name in {"pesto", "pesto_graph", "pesto_gvp", "pesto_graph_gvp"}
        self.use_pure_gvp_backbone = self.structure_encoder_name == "pure_gvp"
        self.use_full_gvp_backbone = self.structure_encoder_name == "full_gvp"

        self.em = pt.nn.Sequential(
            pt.nn.Linear(config["em"]["N0"], config["em"]["N1"]),
            pt.nn.ELU(),
            pt.nn.Linear(config["em"]["N1"], config["em"]["N1"]),
            pt.nn.ELU(),
            pt.nn.Linear(config["em"]["N1"], config["em"]["N1"]),
        )
        self.sum = pt.nn.Sequential(*[StateUpdateLayer(layer_params) for layer_params in config["sum"]])
        self.spl = StatePoolLayer(config["spl"]["N0"], config["spl"]["N1"], config["spl"]["Nh"])

        pesto_dim = 2 * config["dm"]["N0"]
        self.receptor_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(pesto_dim),
            pt.nn.Linear(pesto_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.LayerNorm(dim),
        )
        self.pure_gvp_backbone = None
        if self.use_pure_gvp_backbone:
            self.pure_gvp_backbone = PureResidueGVPEncoder(
                atom_feature_dim=config["em"]["N0"],
                dim=dim,
                vector_dim=gvp_vector_dim,
                k=residue_graph_k,
                layers=gvp_layers,
                dropout=dropout,
            )
        self.full_gvp_backbone = None
        if self.use_full_gvp_backbone:
            self.full_gvp_backbone = FullResidueGVPEncoder(
                atom_feature_dim=123,
                dim=dim,
                vector_dim=gvp_vector_dim,
                k=residue_graph_k,
                layers=gvp_layers,
                dropout=dropout,
            )
        valid_structure_encoders = {"full_gvp", "pure_gvp", "pesto", "pesto_graph", "pesto_gvp", "pesto_graph_gvp"}
        if self.structure_encoder_name not in valid_structure_encoders:
            raise ValueError(
                f"Unsupported structure_encoder={self.structure_encoder_name}. "
                f"Choose one of {sorted(valid_structure_encoders)}."
            )
        self.geometry_refiner = None
        if self.structure_encoder_name in {"pesto_graph", "pesto_graph_gvp"}:
            self.geometry_refiner = ResidueGeometryRefiner(
                dim=dim,
                k=residue_graph_k,
                layers=residue_graph_layers,
                dropout=dropout,
            )
        self.gvp_encoder = None
        self.gvp_fusion = None
        if self.structure_encoder_name in {"pesto_gvp", "pesto_graph_gvp"}:
            self.gvp_encoder = GVPStructureEncoder(
                dim=dim,
                vector_dim=gvp_vector_dim,
                k=residue_graph_k,
                layers=gvp_layers,
                dropout=dropout,
            )
            self.gvp_fusion = GatedFusion(dim)

        valid_partner_encoders = {"transformer", "esm2"}
        if self.partner_encoder_name not in valid_partner_encoders:
            raise ValueError(
                f"Unsupported partner_encoder={self.partner_encoder_name}. "
                f"Choose one of {sorted(valid_partner_encoders)}."
            )
        if self.partner_encoder_name == "esm2":
            if self.use_cached_partner_esm:
                self.partner_encoder = CachedESM2PartnerEncoder(dim=dim, model_name=esm_model_name)
            else:
                self.partner_encoder = ESM2PartnerEncoder(
                    dim=dim,
                    model_name=esm_model_name,
                    freeze=freeze_esm,
                    max_length=esm_max_length,
                    repr_layer=esm_repr_layer,
                )
        else:
            self.partner_encoder = ProteinSequenceEncoder(dim=dim, heads=heads, dropout=dropout)
        valid_receptor_sequence_encoders = {"none", "shared_partner", "esm2", "transformer"}
        if self.receptor_sequence_encoder_name not in valid_receptor_sequence_encoders:
            raise ValueError(
                f"Unsupported receptor_sequence_encoder={self.receptor_sequence_encoder_name}. "
                f"Choose one of {sorted(valid_receptor_sequence_encoders)}."
            )
        self.receptor_sequence_encoder = None
        if self.receptor_sequence_encoder_name == "shared_partner":
            self.receptor_sequence_encoder = self.partner_encoder
        elif self.receptor_sequence_encoder_name == "esm2":
            self.receptor_sequence_encoder = ESM2PartnerEncoder(
                dim=dim,
                model_name=esm_model_name,
                freeze=freeze_esm,
                max_length=esm_max_length,
                repr_layer=esm_repr_layer,
            )
        elif self.receptor_sequence_encoder_name == "transformer":
            self.receptor_sequence_encoder = ProteinSequenceEncoder(dim=dim, heads=heads, dropout=dropout)
        self.receptor_seq_fusion = GatedFusion(dim) if self.receptor_sequence_encoder is not None else None
        self.empty_partner = pt.nn.Parameter(pt.zeros(1, dim))
        self.global_token_bank = pt.nn.Parameter(pt.randn(2, dim) * 0.02)

        self.prior_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )
        self.probe_generator = InteractionQueryGenerator(dim=dim, num_queries=num_queries)
        self.blocks = pt.nn.ModuleList([ApexInteractionBlock(dim, heads, dropout) for _ in range(num_blocks)])
        self.residue_reads_partner_final = CrossAttentionContext(dim, heads, dropout)
        self.residue_reads_query_final = CrossAttentionContext(dim, heads, dropout)
        self.conditioned_head = ResiduePartnerFusionHead(dim=dim, heads=heads, dropout=dropout)
        self.pair_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 6 + 3),
            pt.nn.Linear(dim * 6 + 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Linear(dim // 2, 1),
        )

    def encode_receptor(self, X, ids_topk, q0, M, residue_graph_cache=None):
        if self.use_full_gvp_backbone:
            return self.full_gvp_backbone(X, q0, M, graph_cache=residue_graph_cache).squeeze(0)
        if self.use_pure_gvp_backbone:
            return self.pure_gvp_backbone(X, q0, M).squeeze(0)
        q = self.em.forward(q0)
        p0 = pt.zeros((q.shape[0] + 1, X.shape[1], q.shape[1]), device=X.device, dtype=q.dtype)
        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)
        qa, pa, _, _, _ = self.sum.forward((q, p0, ids_topk, D_nn, R_nn))
        qr, pr = self.spl.forward(qa[1:], pa[1:], M)
        return pt.cat([qr, pt.norm(pr, dim=1)], dim=1)

    def _maybe_drop_partner(self, partner, partner_mask):
        if not self.training or self.partner_dropout <= 0:
            return partner, partner_mask
        if pt.rand((), device=partner.device) >= self.partner_dropout:
            return partner, partner_mask
        partner = self.empty_partner.view(1, 1, -1).expand(partner.shape[0], 1, -1)
        partner_mask = pt.ones((partner.shape[0], 1), device=partner.device, dtype=partner.dtype)
        return partner, partner_mask

    def _surface_indices(self, h_res, prior):
        n = h_res.shape[1]
        indices = pt.arange(n, device=h_res.device)
        return indices, prior.detach()

    def _partner_to_context(self, h_res, partner, partner_mask):
        summary = masked_mean(partner, partner_mask, dim=1)
        return summary.unsqueeze(1).expand(-1, h_res.shape[1], -1)

    def refine_receptor_geometry(self, h_res, X_i, M_i, residue_mask):
        if self.geometry_refiner is not None:
            h_res = self.geometry_refiner(h_res, X_i, M_i)
        if self.gvp_encoder is not None:
            h_gvp = self.gvp_encoder(h_res, X_i, M_i)
            h_res = self.gvp_fusion(h_res, h_gvp, residue_mask)
        return h_res

    def forward(
        self,
        X,
        ids_topk,
        q0,
        M,
        atom_counts,
        residue_counts,
        partner_seq_indices,
        partner_mask,
        receptor_seq_indices=None,
        receptor_mask=None,
        partner_esm_embedding=None,
        partner_esm_mask=None,
        residue_graph_caches=None,
    ):
        outputs_prior = []
        outputs_cond = []
        outputs_surface_score = []
        outputs_pair = []
        outputs_delta = []
        surface_index_list = []
        atom_offset = 0
        residue_offset = 0
        for sample_idx in range(len(atom_counts)):
            atom_count = int(atom_counts[sample_idx].item())
            residue_count = int(residue_counts[sample_idx].item())
            X_i = X[atom_offset : atom_offset + atom_count]
            ids_i = ids_topk[atom_offset : atom_offset + atom_count].clone()
            valid_ids = ids_i > 0
            ids_i[valid_ids] -= atom_offset
            q_i = q0[atom_offset : atom_offset + atom_count]
            M_i = M[atom_offset : atom_offset + atom_count, residue_offset : residue_offset + residue_count]

            graph_cache_i = None
            if residue_graph_caches is not None:
                graph_cache_i = residue_graph_caches[sample_idx]
            zr = self.encode_receptor(X_i, ids_i, q_i, M_i, residue_graph_cache=graph_cache_i)
            if self.use_full_gvp_backbone or self.use_pure_gvp_backbone:
                h_res = zr.unsqueeze(0)
            else:
                h_res = self.receptor_proj(zr).unsqueeze(0)
            residue_mask = pt.ones((1, residue_count), device=h_res.device, dtype=h_res.dtype)
            if not (self.use_full_gvp_backbone or self.use_pure_gvp_backbone):
                h_res = self.refine_receptor_geometry(h_res, X_i, M_i, residue_mask)
            if self.receptor_sequence_encoder is not None and receptor_seq_indices is not None:
                receptor_i = receptor_seq_indices[sample_idx : sample_idx + 1, :residue_count].to(device=h_res.device)
                if receptor_mask is None:
                    receptor_mask_i = pt.ones_like(receptor_i, dtype=h_res.dtype, device=h_res.device)
                else:
                    receptor_mask_i = receptor_mask[sample_idx : sample_idx + 1, :residue_count].to(
                        device=h_res.device, dtype=h_res.dtype
                    )
                receptor_seq_emb = self.receptor_sequence_encoder(receptor_i, receptor_mask_i)
                if receptor_seq_emb.shape[1] < residue_count:
                    pad = receptor_seq_emb.new_zeros(1, residue_count - receptor_seq_emb.shape[1], receptor_seq_emb.shape[2])
                    receptor_seq_emb = pt.cat([receptor_seq_emb, pad], dim=1)
                receptor_seq_emb = receptor_seq_emb[:, :residue_count, :]
                h_res = self.receptor_seq_fusion(h_res, receptor_seq_emb, residue_mask)

            partner_i = partner_seq_indices[sample_idx : sample_idx + 1].to(device=h_res.device)
            partner_mask_i = partner_mask[sample_idx : sample_idx + 1].to(device=h_res.device, dtype=h_res.dtype)
            if partner_esm_embedding is not None:
                if self.partner_encoder_name != "esm2":
                    raise ValueError("partner_esm_embedding cache requires partner_encoder='esm2'.")
                raw_partner = partner_esm_embedding[sample_idx : sample_idx + 1].to(
                    device=h_res.device,
                    dtype=h_res.dtype,
                )
                if partner_esm_mask is None:
                    partner_mask_i = pt.ones(
                        raw_partner.shape[:2],
                        device=h_res.device,
                        dtype=h_res.dtype,
                    )
                else:
                    partner_mask_i = partner_esm_mask[sample_idx : sample_idx + 1].to(
                        device=h_res.device,
                        dtype=h_res.dtype,
                    )
                partner = self.partner_encoder.proj(raw_partner) * partner_mask_i.unsqueeze(-1)
            else:
                partner = self.partner_encoder(partner_i, partner_mask_i)
            partner, partner_mask_i = self._maybe_drop_partner(partner, partner_mask_i)

            prior = self.prior_head(h_res).squeeze(0).squeeze(-1)
            surface_indices, candidate_scores = self._surface_indices(h_res, prior)
            q_tokens = self.probe_generator(h_res, partner, partner_mask_i, residue_mask)
            global_tokens = self.global_token_bank.unsqueeze(0).expand(h_res.shape[0], -1, -1)

            for block in self.blocks:
                h_res, partner, q_tokens, global_tokens = block(
                    h_res,
                    partner,
                    q_tokens,
                    global_tokens,
                    partner_mask_i,
                    residue_mask,
                )

            partner_context = self.residue_reads_partner_final(
                h_res,
                partner,
                partner_mask_i,
                residue_mask,
            )
            query_context = self.residue_reads_query_final(
                h_res,
                q_tokens,
                query_mask=residue_mask,
            )
            receptor_summary = masked_mean(h_res, residue_mask, dim=1)
            partner_summary = masked_mean(partner, partner_mask_i, dim=1)
            conditioned_delta = self.conditioned_head(
                h_res,
                partner_context,
                query_context,
                global_tokens[:, 0, :],
                global_tokens[:, 1, :],
                prior,
            ).squeeze(0)
            interface_weights = pt.sigmoid(prior.detach()).softmax(dim=0)
            interface_summary = (h_res.squeeze(0) * interface_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            abs_delta_mean = conditioned_delta.abs().mean().view(1, 1)
            signed_delta_mean = conditioned_delta.mean().view(1, 1)
            prior_mean = prior.mean().view(1, 1)
            pair_features = pt.cat(
                [
                    receptor_summary,
                    interface_summary,
                    partner_summary,
                    receptor_summary * partner_summary,
                    interface_summary * partner_summary,
                    global_tokens[:, 0, :] * global_tokens[:, 1, :],
                    abs_delta_mean,
                    signed_delta_mean,
                    prior_mean,
                ],
                dim=-1,
            )
            pair_logit = self.pair_head(pair_features).view(-1)

            outputs_prior.append(prior)
            outputs_cond.append(conditioned_delta)
            outputs_surface_score.append(candidate_scores)
            outputs_pair.append(pair_logit)
            outputs_delta.append(conditioned_delta)
            surface_index_list.append(surface_indices)

            atom_offset += atom_count
            residue_offset += residue_count

        return {
            "logit_prior": pt.cat(outputs_prior, dim=0),
            "logit_conditioned": pt.cat(outputs_cond, dim=0),
            "logit_delta": pt.cat(outputs_delta, dim=0),
            "pair_logit": pt.cat(outputs_pair, dim=0),
            "candidate_scores": pt.cat(outputs_surface_score, dim=0),
            "surface_indices": surface_index_list,
        }
