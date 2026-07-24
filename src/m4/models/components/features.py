"""Deterministic residue-level chemistry and geometry features."""

import torch as pt


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_PHYS_TABLE = {
    # hydrophobicity(KD scaled), volume, charge, polar, aromatic, sulfur, proline, glycine
    "A": (0.62, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "C": (0.29, 0.39, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0),
    "D": (-0.90, 0.42, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "E": (-0.74, 0.50, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "F": (1.00, 0.67, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "G": (0.48, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "H": (-0.40, 0.52, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0),
    "I": (0.95, 0.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "K": (-1.00, 0.64, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "L": (0.84, 0.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "M": (0.64, 0.62, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "N": (-0.78, 0.42, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "P": (0.12, 0.43, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "Q": (-0.85, 0.54, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "R": (-1.00, 0.70, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "S": (-0.18, 0.33, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "T": (-0.05, 0.43, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "V": (0.74, 0.47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "W": (0.37, 0.85, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
    "Y": (-0.31, 0.72, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    "X": (0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


class AugFeatureBuilder(pt.nn.Module):
    """Build deterministic residue-level chemistry, pseudo-SS and exposure features.

    The features intentionally use only current cached atom tensors and residue graph
    geometry, so no DSSP/SASA preprocessing is required.
    """

    def __init__(self, atom_feature_dim=123, residue_feature_dim=43):
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.residue_feature_dim = int(residue_feature_dim)

    @staticmethod
    def _scatter_sum(values, atom_to_residue_index, num_residues):
        out = values.new_zeros((int(num_residues),) + tuple(values.shape[1:]))
        out.index_add_(0, atom_to_residue_index.to(device=values.device, dtype=pt.long), values)
        return out

    @staticmethod
    def _assignment(M, atom_to_residue_index, num_residues, device):
        if atom_to_residue_index is not None:
            if num_residues is None:
                raise ValueError("num_residues is required with atom_to_residue_index.")
            return atom_to_residue_index.to(device=device, dtype=pt.long), int(num_residues)
        if M is None:
            raise ValueError("Either M or atom_to_residue_index must be provided.")
        return M.to(device=device).argmax(dim=1).long(), int(M.shape[1])

    def centers_from_atom_mask(self, X, M=None, atom_to_residue_index=None, num_residues=None):
        if M is not None and atom_to_residue_index is None:
            weights = M.to(dtype=X.dtype, device=X.device)
            denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            return weights.transpose(0, 1).matmul(X) / denom
        atom_index, residue_count = self._assignment(M, atom_to_residue_index, num_residues, X.device)
        sums = self._scatter_sum(X, atom_index, residue_count)
        counts = self._scatter_sum(X.new_ones((X.shape[0], 1)), atom_index, residue_count).clamp_min(1.0)
        return sums / counts

    def aggregate_atom_features(self, q_atom, M=None, atom_to_residue_index=None, num_residues=None):
        if M is not None and atom_to_residue_index is None:
            weights = M.to(dtype=q_atom.dtype, device=q_atom.device)
            denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            return weights.transpose(0, 1).matmul(q_atom) / denom
        atom_index, residue_count = self._assignment(M, atom_to_residue_index, num_residues, q_atom.device)
        sums = self._scatter_sum(q_atom, atom_index, residue_count)
        counts = self._scatter_sum(q_atom.new_ones((q_atom.shape[0], 1)), atom_index, residue_count).clamp_min(1.0)
        return sums / counts

    def aa_features(self, q_res):
        n = q_res.shape[0]
        device = q_res.device
        dtype = q_res.dtype
        atom_offset = max(self.atom_feature_dim - 64, 0)
        res_start = 30
        res_end = min(res_start + 29, q_res.shape[1], atom_offset)
        res_onehot = q_res[:, res_start:res_end]
        if res_onehot.shape[1] < 20:
            pad = q_res.new_zeros(n, 20 - res_onehot.shape[1])
            aa20 = pt.cat([res_onehot, pad], dim=-1)
        else:
            aa20 = res_onehot[:, :20]
        table = q_res.new_tensor([AA_PHYS_TABLE[aa] for aa in AA_ORDER])
        known = aa20.sum(dim=-1, keepdim=True).clamp_max(1.0)
        phys = aa20.matmul(table)
        unknown = 1.0 - known
        phys = phys + unknown * q_res.new_tensor(AA_PHYS_TABLE["X"]).unsqueeze(0)
        return aa20.to(device=device, dtype=dtype), phys.to(device=device, dtype=dtype)

    def graph_features(self, centers, graph_cache=None):
        n = centers.shape[0]
        if n <= 1:
            return centers.new_zeros(n, 9)
        idx = None
        dist = None
        if graph_cache:
            for k in (16, 24, 32):
                idx = graph_cache.get(f"knn_idx_k{k}")
                dist = graph_cache.get(f"knn_dist_k{k}")
                if idx is not None and dist is not None:
                    idx = idx.to(device=centers.device, dtype=pt.long)
                    dist = dist.to(device=centers.device, dtype=centers.dtype)
                    break
        if idx is None or dist is None or idx.shape[0] != n:
            full = pt.cdist(centers, centers)
            kk = min(17, n)
            dist, idx = pt.topk(full, k=kk, dim=-1, largest=False)
            dist = dist[:, 1:]
            idx = idx[:, 1:]
        k = min(idx.shape[1], dist.shape[1], max(n - 1, 1))
        idx = idx[:, :k]
        dist = dist[:, :k]
        density = pt.exp(-dist / 10.0).mean(dim=-1, keepdim=True)
        mean_dist = dist.mean(dim=-1, keepdim=True) / 30.0
        min_dist = dist.min(dim=-1, keepdim=True).values / 10.0
        max_dist = dist.max(dim=-1, keepdim=True).values / 30.0
        radial = centers - centers.mean(dim=0, keepdim=True)
        radial_norm = radial.norm(dim=-1, keepdim=True)
        radial_z = (radial_norm - radial_norm.mean()) / radial_norm.std().clamp_min(1e-3)
        neigh = centers[idx]
        centered = centers.unsqueeze(1)
        local_vec = neigh - centered
        cov_trace = (local_vec.square().sum(dim=-1).mean(dim=-1, keepdim=True)).sqrt() / 30.0
        seq_rel = idx - pt.arange(n, device=centers.device).unsqueeze(-1)
        close_seq_frac = (seq_rel.abs() <= 4).to(centers.dtype).mean(dim=-1, keepdim=True)
        far_seq_frac = (seq_rel.abs() > 24).to(centers.dtype).mean(dim=-1, keepdim=True)
        return pt.cat([density, mean_dist, min_dist, max_dist, radial_z, cov_trace, close_seq_frac, far_seq_frac, radial_norm / 50.0], dim=-1)

    def pseudo_secondary_features(self, centers):
        n = centers.shape[0]
        if n <= 2:
            return centers.new_zeros(n, 6)
        prev = pt.cat([centers[:1], centers[:-1]], dim=0)
        nxt = pt.cat([centers[1:], centers[-1:]], dim=0)
        tangent_prev = centers - prev
        tangent_next = nxt - centers
        step_prev = tangent_prev.norm(dim=-1, keepdim=True) / 4.0
        step_next = tangent_next.norm(dim=-1, keepdim=True) / 4.0
        u_prev = tangent_prev / tangent_prev.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        u_next = tangent_next / tangent_next.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        bend_cos = (u_prev * u_next).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        bend = 0.5 * (1.0 - bend_cos)
        pseudo_helix = pt.exp(-((bend_cos - 0.5) / 0.35) ** 2)
        pseudo_beta = pt.exp(-((bend_cos - 0.95) / 0.15) ** 2)
        pseudo_loop = (1.0 - pt.maximum(pseudo_helix, pseudo_beta)).clamp(0.0, 1.0)
        return pt.cat([step_prev, step_next, bend, pseudo_helix, pseudo_beta, pseudo_loop], dim=-1)

    def forward(self, X, q_atom, M=None, graph_cache=None, atom_to_residue_index=None, num_residues=None, centers_override=None):
        dense_replay = M is not None and atom_to_residue_index is None
        if dense_replay:
            residue_count = int(M.shape[1])
            centers = (
                centers_override.to(device=X.device, dtype=X.dtype)
                if centers_override is not None else self.centers_from_atom_mask(X, M)
            )
            q_res = self.aggregate_atom_features(q_atom, M)
        else:
            atom_index, residue_count = self._assignment(M, atom_to_residue_index, num_residues, X.device)
            centers = (
                centers_override.to(device=X.device, dtype=X.dtype)
                if centers_override is not None
                else self.centers_from_atom_mask(X, atom_to_residue_index=atom_index, num_residues=residue_count)
            )
            q_res = self.aggregate_atom_features(
                q_atom, atom_to_residue_index=atom_index, num_residues=residue_count
            )
        aa20, phys = self.aa_features(q_res)
        graph = self.graph_features(centers, graph_cache=graph_cache)
        pseudo_ss = self.pseudo_secondary_features(centers)
        if dense_replay:
            atom_count = M.to(dtype=X.dtype, device=X.device).sum(dim=0).log1p().unsqueeze(-1) / 5.0
        else:
            atom_count = self._scatter_sum(
                X.new_ones((X.shape[0], 1)), atom_index, residue_count
            ).log1p() / 5.0
        feats = pt.cat([aa20, phys, graph, pseudo_ss, atom_count], dim=-1)
        if feats.shape[1] < self.residue_feature_dim:
            feats = pt.cat([feats, feats.new_zeros(feats.shape[0], self.residue_feature_dim - feats.shape[1])], dim=-1)
        elif feats.shape[1] > self.residue_feature_dim:
            feats = feats[:, : self.residue_feature_dim]
        return feats



