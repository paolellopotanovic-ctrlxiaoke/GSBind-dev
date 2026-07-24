import math

import torch as pt

from apex_model import (
    CachedESM2PartnerEncoder,
    FullResidueGVPEncoder,
    ResidualCrossAttention,
    ResidualSelfAttention,
    masked_mean,
)


class SimplePartnerInteractionBlock(pt.nn.Module):
    """One plain reciprocal receptor-partner interaction layer."""

    def __init__(self, dim=256, heads=4, dropout=0.1, partner_updates=True):
        super().__init__()
        self.partner_updates = bool(partner_updates)
        self.receptor_self = ResidualSelfAttention(dim, heads=heads, dropout=dropout)
        self.partner_self = ResidualSelfAttention(dim, heads=heads, dropout=dropout)
        self.receptor_reads_partner = ResidualCrossAttention(dim, heads=heads, dropout=dropout)
        self.partner_reads_receptor = ResidualCrossAttention(dim, heads=heads, dropout=dropout)

    def forward(self, receptor, partner, receptor_mask, partner_mask):
        receptor = self.receptor_self(receptor, receptor_mask)
        partner = self.partner_self(partner, partner_mask)
        receptor = self.receptor_reads_partner(receptor, partner, key_mask=partner_mask, query_mask=receptor_mask)
        if self.partner_updates:
            partner = self.partner_reads_receptor(partner, receptor, key_mask=receptor_mask, query_mask=partner_mask)
        return receptor, partner


class SimplePartnerGVPESMNet(pt.nn.Module):
    """Simple partner-conditioned residue interface predictor.

    Input: receptor apo/AF2 structure tensors plus cached partner ESM embeddings.
    Output: one receptor residue logit per valid receptor residue.
    """

    def __init__(
        self,
        dim=256,
        heads=4,
        num_blocks=4,
        gvp_layers=4,
        gvp_vector_dim=16,
        residue_graph_k=24,
        esm_model_name="esm2_t33_650M_UR50D",
        dropout=0.1,
        partner_updates=True,
    ):
        super().__init__()
        self.receptor_encoder = FullResidueGVPEncoder(
            atom_feature_dim=123,
            dim=dim,
            vector_dim=gvp_vector_dim,
            k=residue_graph_k,
            layers=gvp_layers,
            dropout=dropout,
        )
        self.partner_encoder = CachedESM2PartnerEncoder(dim=dim, model_name=esm_model_name)
        self.blocks = pt.nn.ModuleList(
            [
                SimplePartnerInteractionBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    partner_updates=partner_updates,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
        )
        self.pair_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 4),
            pt.nn.Linear(dim * 4, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )

    def encode_partner(self, partner_esm_embedding, partner_esm_mask):
        partner = self.partner_encoder(partner_esm_embedding, partner_esm_mask)
        return partner

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
        partner_esm_embedding=None,
        partner_esm_mask=None,
        residue_graph_caches=None,
    ):
        if partner_esm_embedding is None:
            raise ValueError("SimplePartnerGVPESMNet requires cached partner_esm_embedding.")
        outputs = []
        pair_logits = []
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
            graph_cache_i = residue_graph_caches[sample_idx] if residue_graph_caches is not None else None

            receptor = self.receptor_encoder(X_i, q_i, M_i, graph_cache=graph_cache_i)
            receptor_mask = pt.ones((1, residue_count), dtype=receptor.dtype, device=receptor.device)

            partner_raw = partner_esm_embedding[sample_idx : sample_idx + 1].to(device=receptor.device, dtype=receptor.dtype)
            if partner_esm_mask is None:
                partner_mask_i = pt.ones(partner_raw.shape[:2], dtype=receptor.dtype, device=receptor.device)
            else:
                partner_mask_i = partner_esm_mask[sample_idx : sample_idx + 1].to(device=receptor.device, dtype=receptor.dtype)
            partner = self.encode_partner(partner_raw, partner_mask_i)

            for block in self.blocks:
                receptor, partner = block(receptor, partner, receptor_mask, partner_mask_i)

            logits = self.output_head(receptor).squeeze(0).squeeze(-1)
            receptor_global = masked_mean(receptor, receptor_mask, dim=1)
            partner_global = masked_mean(partner, partner_mask_i, dim=1)
            pair_feature = pt.cat(
                [
                    receptor_global,
                    partner_global,
                    receptor_global * partner_global,
                    (receptor_global - partner_global).abs(),
                ],
                dim=-1,
            )
            pair_logit = self.pair_head(pair_feature).view(-1)
            outputs.append(logits)
            pair_logits.append(pair_logit)
            atom_offset += atom_count
            residue_offset += residue_count
        return {"logits": pt.cat(outputs, dim=0), "pair_logit": pt.cat(pair_logits, dim=0)}
