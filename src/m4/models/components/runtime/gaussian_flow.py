"""Multiview and chunked pair-Gaussian runtime operations."""

import torch as pt
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class GaussianFlowMixin:
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

