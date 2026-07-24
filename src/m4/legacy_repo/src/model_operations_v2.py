import torch as pt

from src.feature_encoding_v2 import EDGE_FEATURE_NAMES


def unpack_state_features_v2(X, ids_topk, q):
    mask = ids_topk > 0
    safe_ids = (ids_topk - 1).clamp(min=0, max=max(X.shape[0] - 1, 0))
    R_nn = X[safe_ids] - X.unsqueeze(1)
    D_nn = pt.norm(R_nn, dim=2)
    R_nn = R_nn / D_nn.clamp_min(1e-6).unsqueeze(2)
    D_nn = D_nn * mask.float()
    R_nn = R_nn * mask.unsqueeze(2).float()

    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device, dtype=q.dtype), q], dim=0)
    ids_topk = pt.cat(
        [
            pt.zeros((1, ids_topk.shape[1]), dtype=pt.long, device=ids_topk.device),
            ids_topk.long(),
        ],
        dim=0,
    )
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1]), device=D_nn.device, dtype=D_nn.dtype), D_nn], dim=0)
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], 3), device=R_nn.device, dtype=R_nn.dtype), R_nn], dim=0)
    return q, ids_topk, D_nn, R_nn


def _gather_meta(meta, ids_nn, fill_value=-1):
    meta = pt.cat(
        [pt.full((1,), fill_value, dtype=meta.dtype, device=meta.device), meta],
        dim=0,
    )
    return meta[ids_nn]


def compute_edge_features(
    D_nn,
    ids_nn,
    atom_res_index,
    atom_chain_index,
    atom_element_index=None,
    atom_name_index=None,
    rbf_count=16,
    rbf_max=20.0,
):
    device = D_nn.device
    dtype = D_nn.dtype
    N, n = D_nn.shape
    valid = ids_nn > 0

    centers = pt.linspace(0.0, rbf_max, rbf_count, device=device, dtype=dtype)
    gamma = 1.0 / ((rbf_max / max(rbf_count - 1, 1)) ** 2)
    rbf = pt.exp(-gamma * (D_nn.unsqueeze(2) - centers.view(1, 1, -1)) ** 2)
    distance_norm = (D_nn / rbf_max).clamp(0.0, 1.0).unsqueeze(2)

    res_i = pt.cat(
        [pt.full((1,), -1, dtype=atom_res_index.dtype, device=device), atom_res_index.to(device)],
        dim=0,
    ).view(-1, 1)
    chain_i = pt.cat(
        [pt.full((1,), -1, dtype=atom_chain_index.dtype, device=device), atom_chain_index.to(device)],
        dim=0,
    ).view(-1, 1)
    row_res = res_i.expand(N, n)
    row_chain = chain_i.expand(N, n)
    nbr_res = _gather_meta(atom_res_index.to(device), ids_nn)
    nbr_chain = _gather_meta(atom_chain_index.to(device), ids_nn)

    same_residue = (row_res == nbr_res) & valid
    same_chain = (row_chain == nbr_chain) & valid
    inter_chain = (row_chain != nbr_chain) & valid
    residue_gap = pt.abs(row_res - nbr_res)
    residue_gap_1 = (residue_gap == 1) & same_chain
    residue_gap_2_4 = (residue_gap >= 2) & (residue_gap <= 4) & same_chain
    residue_gap_ge5 = (residue_gap >= 5) & same_chain

    sequential_backbone = pt.zeros((N, n), dtype=pt.bool, device=device)
    covalent_candidate = pt.zeros((N, n), dtype=pt.bool, device=device)
    if atom_name_index is not None:
        name_i = pt.cat(
            [pt.full((1,), -1, dtype=atom_name_index.dtype, device=device), atom_name_index.to(device)],
            dim=0,
        ).view(-1, 1)
        name_j = _gather_meta(atom_name_index.to(device), ids_nn)
        # std_names indices: N=1, C=2, CA=0.
        n_idx, c_idx, ca_idx = 1, 2, 0
        c_to_n = ((name_i == c_idx) & (name_j == n_idx)) | ((name_i == n_idx) & (name_j == c_idx))
        has_ca = (name_i == ca_idx) | (name_j == ca_idx)
        sequential_backbone = residue_gap_1 & same_chain & (c_to_n | has_ca)
        covalent_candidate = same_chain & (D_nn < 1.95) & (same_residue | residue_gap_1)
        covalent_candidate = covalent_candidate | (sequential_backbone & (D_nn < 2.2))
    else:
        covalent_candidate = same_chain & (D_nn < 1.95) & (same_residue | residue_gap_1)

    binary = pt.stack(
        [
            same_residue,
            same_chain,
            inter_chain,
            residue_gap_1,
            residue_gap_2_4,
            residue_gap_ge5,
            sequential_backbone,
            covalent_candidate,
        ],
        dim=2,
    ).to(dtype)

    edge_attr = pt.cat([rbf, distance_norm, binary], dim=2)
    edge_attr = edge_attr * valid.unsqueeze(2).to(dtype)
    return edge_attr


class ScalarVectorCoAttentionLayer(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk, nn, edge_dim=len(EDGE_FEATURE_NAMES)):
        super().__init__()
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk
        self.nn = nn
        self.edge_dim = edge_dim
        self.m_nn = pt.nn.Parameter(pt.arange(nn, dtype=pt.int64), requires_grad=False)

        scalar_in = 7 * Ns + edge_dim
        self.query_s = pt.nn.Sequential(
            pt.nn.Linear(2 * Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nh * Nk),
        )
        self.key_s = pt.nn.Sequential(
            pt.nn.Linear(scalar_in, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )
        self.value_s = pt.nn.Sequential(
            pt.nn.Linear(scalar_in, 2 * Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2 * Ns, Nh * Ns),
        )
        self.query_v = pt.nn.Sequential(
            pt.nn.Linear(2 * Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nh * Nk),
        )
        self.key_v = pt.nn.Sequential(
            pt.nn.Linear(scalar_in, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )
        self.gate_v = pt.nn.Sequential(
            pt.nn.Linear(scalar_in, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nh * Ns),
            pt.nn.Sigmoid(),
        )
        self.value_v = pt.nn.Sequential(
            pt.nn.Linear(scalar_in, 2 * Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2 * Ns, Nh * Ns),
        )
        self.proj_s = pt.nn.Sequential(
            pt.nn.Linear(Nh * Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )
        self.proj_v = pt.nn.Linear(Nh * Ns, Ns, bias=False)
        self.scale = pt.nn.Parameter(pt.sqrt(pt.tensor(float(Nk))), requires_grad=False)

    def forward(self, Z):
        (
            q,
            p,
            ids_topk,
            D_topk,
            R_topk,
            edge_attr,
        ) = Z
        k = min(self.nn, ids_topk.shape[1])
        ids_nn = ids_topk[:, self.m_nn[:k]]
        d_nn = D_topk[:, self.m_nn[:k]]
        r_nn = R_topk[:, self.m_nn[:k]]
        e_nn = edge_attr[:, self.m_nn[:k]]
        valid = ids_nn > 0

        q_nn = q[ids_nn]
        p_nn = p[ids_nn]
        q_center = q.unsqueeze(1).expand(-1, k, -1)
        p_center = p.unsqueeze(1).expand(-1, k, -1, -1)

        p_norm_i = pt.norm(p, dim=1).unsqueeze(1).expand(-1, k, -1)
        p_norm_j = pt.norm(p_nn, dim=2)
        dot_pij = pt.sum(p_center * p_nn, dim=2)
        proj_pi_r = pt.sum(p_center * r_nn.unsqueeze(3), dim=2)
        proj_pj_r = pt.sum(p_nn * r_nn.unsqueeze(3), dim=2)
        vector_invariants = pt.cat([p_norm_i, p_norm_j, dot_pij, proj_pi_r, proj_pj_r], dim=2)

        scalar_input = pt.cat([q_center, q_nn, e_nn, vector_invariants], dim=2)
        node_input = pt.cat([q, pt.norm(p, dim=1)], dim=1)

        Qs = self.query_s(node_input).view(q.shape[0], self.Nh, self.Nk)
        Ks = self.key_s(scalar_input).view(q.shape[0], k, self.Nk).transpose(1, 2)
        Vs = self.value_s(scalar_input).view(q.shape[0], k, self.Nh, self.Ns).transpose(1, 2)
        logits_s = pt.matmul(Qs, Ks) / self.scale
        logits_s = logits_s.masked_fill(~valid.unsqueeze(1), -1e4)
        alpha_s = pt.nn.functional.softmax(logits_s, dim=2) * valid.unsqueeze(1).float()
        z_s = pt.sum(alpha_s.unsqueeze(3) * Vs, dim=2).reshape(q.shape[0], self.Nh * self.Ns)
        dq = self.proj_s(z_s)

        Qv = self.query_v(node_input).view(q.shape[0], self.Nh, self.Nk)
        Kv = self.key_v(scalar_input).view(q.shape[0], k, self.Nk).transpose(1, 2)
        logits_v = pt.matmul(Qv, Kv) / self.scale
        logits_v = logits_v.masked_fill(~valid.unsqueeze(1), -1e4)
        alpha_v = pt.nn.functional.softmax(logits_v, dim=2) * valid.unsqueeze(1).float()
        gate_v = self.gate_v(scalar_input).view(q.shape[0], k, self.Nh, self.Ns).transpose(1, 2)
        value_v = self.value_v(scalar_input).view(q.shape[0], k, self.Nh, self.Ns).transpose(1, 2)
        coeff_v = alpha_v.unsqueeze(3) * gate_v * value_v
        r_basis = r_nn.permute(0, 2, 1).unsqueeze(1).unsqueeze(4)
        z_v = pt.sum(coeff_v.unsqueeze(2) * r_basis, dim=3)
        z_v = z_v.permute(0, 2, 1, 3).reshape(q.shape[0], 3, self.Nh * self.Ns)
        dp = self.proj_v(z_v)

        row_mask = (pt.arange(q.shape[0], device=q.device) > 0).float().view(-1, 1)
        q = (q + dq) * row_mask
        p = (p + dp) * row_mask.view(-1, 1, 1)
        return q, p, ids_topk, D_topk, R_topk, edge_attr


def residue_mean_pool(q, p, M):
    denom = M.sum(dim=0).clamp_min(1.0)
    q_res = pt.matmul(M.transpose(0, 1), q) / denom.unsqueeze(1)
    p_res = pt.einsum("nr,ncs->rcs", M, p) / denom.view(-1, 1, 1)
    return q_res, p_res
