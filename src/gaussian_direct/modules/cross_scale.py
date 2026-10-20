"""Bidirectional residue/surface fusion and structural patch modules."""

import math

import torch as pt

from .gaussian_projection import inverse_softplus


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
