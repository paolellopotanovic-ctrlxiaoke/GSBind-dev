"""Residue features, view fusion, and stochastic output transforms."""

import torch as pt


class StochasticRuntimeMixin:
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

