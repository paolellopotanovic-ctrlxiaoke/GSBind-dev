"""Residue/surface graph preparation and message-passing runtime."""

import torch as pt

from ..surface_cache import build_s1_surface_proxy_cache


class SurfaceFlowMixin:
    def _surface_bus_pre_global_views(self, view_stack, mask):
        if view_stack is None or view_stack.numel() == 0:
            return view_stack, None
        updated = []
        deltas = []
        for view_idx in range(int(view_stack.shape[0])):
            before = view_stack[view_idx].unsqueeze(0)
            after = self.global_feedback(before, mask).squeeze(0)
            updated.append(after)
            deltas.append((after - view_stack[view_idx]).norm(dim=-1).mean())
        return pt.stack(updated, dim=0), pt.stack(deltas).mean() if deltas else None

    def _apply_surface_bus_static(self, h_surf_base, view_stack, surface_pos, surface_normal, surface_geom, centers, mask=None):
        if self.surface_bus is None or view_stack is None or view_stack.numel() == 0:
            return h_surf_base
        bus_views = view_stack
        pre_global_delta = None
        if self.surface_bus_mode == "static_pre_global":
            bus_views, pre_global_delta = self._surface_bus_pre_global_views(view_stack, mask)
        bus = self.surface_bus(
            h_surf_base,
            bus_views,
            surface_pos,
            surface_normal,
            surface_geom,
            centers,
        )
        if pre_global_delta is not None:
            self.surface_bus.last_diagnostics["surface_bus_pre_global_update_norm"] = pre_global_delta.detach()
        return self.surface_bus_static_norm(h_surf_base + self.surface_bus_beta * (bus - h_surf_base))

    def _advance_surface_bus_view_states(self, view_stack, surf_context, residue_vectors, centers, graph_cache=None, true_feat=None, chain_ids=None):
        encoder = self.encoder
        if not hasattr(encoder, "views"):
            raise ValueError("recurrent surface_bus_mode requires multiview encoder views.")
        if getattr(encoder, "cross_view_edge_mode", "none") != "none":
            raise ValueError("recurrent surface_bus_mode currently requires cross_view_edge_mode=none.")
        if getattr(encoder, "oagnn_mode", "none") == "official":
            raise ValueError("recurrent surface_bus_mode currently does not support oagnn_mode=official.")
        centers = centers.to(device=view_stack.device, dtype=view_stack.dtype)
        if residue_vectors is None:
            v0 = view_stack.new_zeros((view_stack.shape[1], self.vector_dim, 3))
        else:
            v0 = residue_vectors.to(device=view_stack.device, dtype=view_stack.dtype)
        updated_views = []
        update_norms = []
        gate_means = []
        for view_idx, name in enumerate(encoder.graph_views[: int(view_stack.shape[0])]):
            view_state = view_stack[view_idx]
            view_input = pt.cat([view_state, surf_context], dim=-1)
            view_update = self.surface_bus_view_update(view_input)
            view_gate = self.surface_bus_view_gate(view_input)
            view_state = self.surface_bus_view_norm(view_state + self.surface_bus_beta * view_gate * view_update)
            update_norms.append(view_update.detach().norm(dim=-1).mean())
            gate_means.append(view_gate.detach().mean())
            nn_idx, nn_dist = encoder._view_edges(name, centers, graph_cache, true_feature_cache=true_feat, chain_ids=chain_ids)
            nn_idx = nn_idx.to(device=view_state.device, dtype=pt.long)
            nn_dist = nn_dist.to(device=view_state.device, dtype=view_state.dtype)
            edge_s, edge_v = encoder.feature_builder.edge_features(centers, nn_idx, nn_dist)
            rel_type = encoder._edge_relation_types(name, centers, nn_idx, nn_dist)
            updated_views.append(
                encoder.views[name](
                    view_state,
                    v0,
                    nn_idx,
                    edge_s,
                    edge_v,
                    rel_type=rel_type,
                    centers=centers,
                )
            )
        next_stack = pt.stack(updated_views, dim=0)
        diag = {
            "surface_bus_recurrent_rounds": view_stack.new_tensor(1.0).detach(),
            "surface_bus_view_update_norm": pt.stack(update_norms).mean().detach() if update_norms else view_stack.new_tensor(0.0).detach(),
            "surface_bus_view_gate_mean": pt.stack(gate_means).mean().detach() if gate_means else view_stack.new_tensor(0.0).detach(),
        }
        return next_stack, diag

    def _interleaved_residue_graph_refine(self, h_res, residue_vectors, centers, graph_cache=None, true_feat=None, chain_ids=None):
        encoder = self.encoder
        if not hasattr(encoder, "views"):
            raise ValueError("global-feedback interleaved residue graph currently requires multiview residue_encoder_mode.")
        h = h_res.squeeze(0)
        centers = centers.to(device=h.device, dtype=h.dtype)
        if residue_vectors is None:
            v0 = h.new_zeros((h.shape[0], self.vector_dim, 3))
        else:
            v0 = residue_vectors.to(device=h.device, dtype=h.dtype)
        view_inputs = {}
        for name in encoder.graph_views:
            nn_idx, nn_dist = encoder._view_edges(name, centers, graph_cache, true_feature_cache=true_feat, chain_ids=chain_ids)
            nn_idx = nn_idx.to(device=h.device, dtype=pt.long)
            nn_dist = nn_dist.to(device=h.device, dtype=h.dtype)
            edge_s, edge_v = encoder.feature_builder.edge_features(centers, nn_idx, nn_dist)
            rel_type = encoder._edge_relation_types(name, centers, nn_idx, nn_dist)
            view_inputs[name] = {
                "nn_idx": nn_idx,
                "edge_s": edge_s,
                "edge_v": edge_v,
                "rel_type": rel_type,
            }
        if getattr(encoder, "cross_view_edge_mode", "none") == "none":
            view_outputs = []
            for name, values in view_inputs.items():
                view_outputs.append(
                    encoder.views[name](
                        h,
                        v0,
                        values["nn_idx"],
                        values["edge_s"],
                        values["edge_v"],
                        rel_type=values["rel_type"],
                        centers=centers,
                    )
                )
        else:
            view_outputs = encoder._forward_cross_view(h, v0, view_inputs, centers=centers)
        stacked = pt.stack(view_outputs, dim=0)
        if encoder.router_mode == "global":
            weights = pt.softmax(encoder.view_logits, dim=0).to(dtype=stacked.dtype, device=stacked.device)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            router_entropy = -(weights * weights.clamp_min(encoder.router_entropy_eps).log()).sum()
        elif encoder.router_mode == "per_residue":
            per_residue = pt.cat(view_outputs, dim=-1)
            router_map = pt.softmax(encoder.view_router(per_residue), dim=-1)
            fused = (router_map.unsqueeze(-1) * stacked.permute(1, 0, 2)).sum(dim=1)
            router_entropy = -(router_map * router_map.clamp_min(encoder.router_entropy_eps).log()).sum(dim=-1).mean()
        else:
            view_tokens = stacked.permute(1, 0, 2).contiguous()
            query = encoder.residue_query(h).unsqueeze(1)
            router_map = None
            fused = None
            for attn, update, norm in zip(encoder.view_attn_rounds, encoder.residue_updates, encoder.attn_round_norms):
                attn_out, attn_weights = attn(query, view_tokens, view_tokens, need_weights=True, average_attn_weights=True)
                query = norm(query + update(pt.cat([query, attn_out], dim=-1)))
                fused = query.squeeze(1)
                router_map = attn_weights.squeeze(1)
            router_entropy = -(router_map * router_map.clamp_min(encoder.router_entropy_eps).log()).sum(dim=-1).mean()
        return encoder.fusion_norm(fused).unsqueeze(0), router_entropy

    def _surface_cache(self, cache, X, q, m):
        if cache is not None:
            return cache
        if m is None:
            raise ValueError("A precomputed S1 surface cache is required by the dense-M-free clean path.")
        item = {
            "X": X.detach().float().cpu(),
            "q": q.detach().float().cpu(),
            "M": m.detach().float().cpu(),
        }
        return build_s1_surface_proxy_cache(item, num_surface_points=self.surface_points, graph_k=self.surface_graph_k)

    def _surface_graph_for_encoder(self, cache, surface_pos):
        if self.surface_extra_graph_ks:
            return self._surface_graph_for_k(cache, surface_pos, self.surface_graph_k)
        if self.surface_extra_graph_k <= 0:
            return cache.get("surface_graph_idx"), cache.get("surface_graph_dist")
        return self._surface_graph_for_k(cache, surface_pos, self.surface_extra_graph_k)

    def _surface_graph_for_k(self, cache, surface_pos, graph_k):
        graph_k = int(graph_k)
        if graph_k <= 0:
            return cache.get("surface_graph_idx"), cache.get("surface_graph_dist")
        if graph_k == self.surface_graph_k:
            graph_idx = cache.get("surface_graph_idx")
            graph_dist = cache.get("surface_graph_dist")
            if graph_idx is not None and graph_idx.numel() > 0:
                return graph_idx, graph_dist
        p = int(surface_pos.shape[0])
        if p <= 1:
            return surface_pos.new_empty((p, 0), dtype=pt.long), surface_pos.new_empty((p, 0))
        k = min(graph_k + 1, p)
        dist = pt.cdist(surface_pos, surface_pos).clamp_min(0.0)
        graph_dist, graph_idx = pt.topk(dist, k=k, dim=-1, largest=False)
        return graph_idx[:, 1:].contiguous(), graph_dist[:, 1:].contiguous()

    def _encode_surface_multigraph(
        self,
        h_surf_state,
        surface_geom,
        surface_pos,
        surface_normal,
        cache,
        surface_initial_vectors=None,
        graph_bundle=None,
    ):
        encoders = [self.surface_encoder, *list(self.surface_extra_encoders)]
        graph_ks = [self.surface_graph_k, *list(self.surface_extra_graph_ks)]
        if self.surface_graph_flow_mode == "k16_then_k32":
            h = h_surf_state
            vectors = []
            used_ks = []
            for encoder, graph_k in zip(encoders, graph_ks):
                graph_idx, graph_dist = (
                    graph_bundle[int(graph_k)] if graph_bundle is not None and int(graph_k) in graph_bundle
                    else self._surface_graph_for_k(cache, surface_pos, graph_k)
                )
                if self.surface_encoder_mode == "gvp":
                    h = encoder(
                        h,
                        surface_geom,
                        surface_pos,
                        surface_normal,
                        graph_idx,
                        graph_dist=graph_dist,
                        initial_vectors=surface_initial_vectors,
                    )
                    vec = getattr(encoder, "last_surface_vectors", None)
                    if vec is not None:
                        vectors.append(vec)
                        surface_initial_vectors = vec
                else:
                    h = encoder(
                        h,
                        surface_geom,
                        surface_pos,
                        surface_normal,
                        graph_idx,
                        graph_dist=graph_dist,
                    )
                used_ks.append(int(graph_k))
            if self.surface_encoder_mode == "gvp" and vectors:
                self.surface_encoder.last_surface_vectors = vectors[-1]
            diagnostics = {
                "surface_graph_view_count": h_surf_state.new_tensor(float(len(used_ks))).detach(),
                "surface_graph_flow_serial": h_surf_state.new_tensor(1.0).detach(),
            }
            for i, graph_k in enumerate(used_ks):
                diagnostics[f"surface_graph_serial_step{i + 1}_k"] = h_surf_state.new_tensor(float(graph_k)).detach()
            self.last_surface_graph_diagnostics = diagnostics
            return h
        outputs = []
        vectors = []
        used_ks = []
        for encoder, graph_k in zip(encoders, graph_ks):
            graph_idx, graph_dist = (
                graph_bundle[int(graph_k)] if graph_bundle is not None and int(graph_k) in graph_bundle
                else self._surface_graph_for_k(cache, surface_pos, graph_k)
            )
            if self.surface_encoder_mode == "gvp":
                out = encoder(
                    h_surf_state,
                    surface_geom,
                    surface_pos,
                    surface_normal,
                    graph_idx,
                    graph_dist=graph_dist,
                    initial_vectors=surface_initial_vectors,
                )
                vec = getattr(encoder, "last_surface_vectors", None)
                if vec is not None:
                    vectors.append(vec)
            else:
                out = encoder(
                    h_surf_state,
                    surface_geom,
                    surface_pos,
                    surface_normal,
                    graph_idx,
                    graph_dist=graph_dist,
                )
            outputs.append(out)
            used_ks.append(int(graph_k))
        if len(outputs) == 1:
            self.last_surface_graph_diagnostics = {
                "surface_graph_view_count": h_surf_state.new_tensor(1.0).detach(),
            }
            return outputs[0]
        weights = pt.softmax(self.surface_graph_logits[: len(outputs)], dim=0).to(device=h_surf_state.device, dtype=h_surf_state.dtype)
        stacked = pt.stack(outputs, dim=0)
        fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
        fused = self.surface_graph_fusion_norm(fused)
        if self.surface_encoder_mode == "gvp" and len(vectors) == len(outputs):
            vec_stack = pt.stack(vectors, dim=0)
            fused_vec = (weights.view(-1, 1, 1, 1) * vec_stack).sum(dim=0)
            self.surface_encoder.last_surface_vectors = fused_vec
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum()
        diagnostics = {
            "surface_graph_view_count": h_surf_state.new_tensor(float(len(outputs))).detach(),
            "surface_graph_router_entropy": entropy.detach(),
        }
        for graph_k, weight in zip(used_ks, weights):
            diagnostics[f"surface_graph_weight_k{graph_k}"] = weight.detach()
        self.last_surface_graph_diagnostics = diagnostics
        return fused
