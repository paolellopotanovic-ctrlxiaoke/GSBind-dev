"""Complete paper-facing dataflow for one M4 forward pass.

The scientific helper stages live in ``components.runtime``. This file keeps
the end-to-end operation order visible in one place while the mixins separate
feature preparation, surface processing, and Gaussian-field execution.
"""

import torch as pt

from .runtime import GaussianFlowMixin, StochasticRuntimeMixin, SurfaceFlowMixin


class M4ExecutionMixin(
    StochasticRuntimeMixin,
    SurfaceFlowMixin,
    GaussianFlowMixin,
):

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
