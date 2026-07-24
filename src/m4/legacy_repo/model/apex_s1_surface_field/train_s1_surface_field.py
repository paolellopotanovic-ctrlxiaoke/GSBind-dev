import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch as pt

try:
    from scipy.optimize import linear_sum_assignment
except ModuleNotFoundError:
    linear_sum_assignment = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_dataset import read_manifest
from apex_s1_surface_field.dataset import S1SurfaceFieldDataset, collate_s1_surface_field_batch
from apex_s1_surface_field.model import S1SurfaceFieldPredictor
from apex_s1_surface_field.soft_labels import masked_soft_bce, residue_centers_from_atom_mask, surface_soft_labels
from apex_generic_gvp_multigraph.metrics import aggregate_metrics, case_metrics, metric_scalar, read_csv, write_csv
from apex_teacher_prior.modules import teacher_logit_distillation_loss
from pinder_manifest_dataset import write_csv as write_manifest_csv
from train_apex import estimate_pos_weight, split_rows_grouped
from train_apex_simple import masked_bce, move_batch_to_device


def dataloader_kwargs(args, shuffle):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "collate_fn": collate_s1_surface_field_batch,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def build_loader(rows, args, shuffle):
    return pt.utils.data.DataLoader(
        S1SurfaceFieldDataset(
            rows,
            strict_residue_alignment=not args.allow_residue_mismatch,
            skip_bad_examples=True,
            tensor_cache_dir=args.tensor_cache_dir,
            require_tensor_cache=args.require_tensor_cache,
            esm_cache_dir=None,
            require_esm_cache=False,
            residue_graph_cache_dir=args.residue_graph_cache_dir,
            require_residue_graph_cache=args.require_residue_graph_cache,
            residue_graph_ks=args.residue_graph_cache_ks,
            surface_cache_dir="",
            require_surface_cache=False,
            build_surface_on_the_fly=False,
            true_feature_cache_dir=args.true_feature_cache_dir,
            require_true_features=args.require_true_features,
            true_feature_version=args.true_feature_version,
            s1_surface_cache_dir=args.s1_surface_cache_dir,
            require_s1_surface_cache=args.require_s1_surface_cache,
            build_s1_surface_on_the_fly=not args.disable_s1_surface_on_the_fly,
            s1_surface_points=args.s1_surface_points,
            s1_surface_graph_k=args.s1_surface_graph_k,
            s1_surface_lappe_dim=args.s1_surface_lappe_dim,
            s1_surface_cache_version=args.s1_surface_cache_version,
            teacher_cache_dir=args.teacher_cache_dir,
            teacher_name=args.teacher_name,
            teacher_cache_version=args.teacher_cache_version,
            require_teacher_cache=args.require_teacher_cache,
            build_soft_residue_labels=args.use_soft_residue_labels,
            soft_label_sigma=args.soft_label_sigma,
            soft_label_max_value=args.soft_label_max_value,
            soft_label_cutoff=args.soft_label_cutoff,
            s1_static_geometry_cache_dir=args.s1_static_geometry_cache_dir,
            s1_static_geometry_cache_version=args.s1_static_geometry_cache_version,
            require_s1_static_geometry_cache=args.require_s1_static_geometry_cache,
        ),
        **dataloader_kwargs(args, shuffle=shuffle),
    )


def model_forward(model, batch):
    if model.training:
        atom_index = batch["atom_to_residue_index"].to(device=batch["q"].device, dtype=pt.long)
        num_residues = int(batch["residue_counts"].sum().item())
        dense_assignment = batch["q"].new_zeros((atom_index.shape[0], num_residues))
        dense_assignment.scatter_(1, atom_index.unsqueeze(1), 1.0)
        return model(
            batch["X"],
            None,
            batch["q"],
            dense_assignment,
            batch["atom_counts"],
            batch["residue_counts"],
            residue_graph_caches=batch.get("residue_graph_caches"),
            s1_surface_caches=batch.get("s1_surface_caches"),
            true_residue_features=batch.get("true_residue_features"),
            residue_chain_ids=batch.get("residue_chain_ids"),
            teacher_hidden=batch.get("teacher_hidden"),
            atom_to_residue_index=None,
            s1_static_geometry_caches=None,
        )
    return model(
        batch["X"],
        None,
        batch["q"],
        None,
        batch["atom_counts"],
        batch["residue_counts"],
        residue_graph_caches=batch.get("residue_graph_caches"),
        s1_surface_caches=batch.get("s1_surface_caches"),
        true_residue_features=batch.get("true_residue_features"),
        residue_chain_ids=batch.get("residue_chain_ids"),
        teacher_hidden=batch.get("teacher_hidden"),
        atom_to_residue_index=batch["atom_to_residue_index"],
        s1_static_geometry_caches=batch.get("s1_static_geometry_caches"),
    )


def build_model(args):
    return S1SurfaceFieldPredictor(
        atom_feature_dim=args.atom_feature_dim,
        dim=args.dim,
        vector_dim=args.gvp_vector_dim,
        graph_views=tuple(x.strip() for x in args.graph_views.split(",") if x.strip()),
        view_layers=args.view_layers,
        rbf_bins=args.rbf_bins,
        dropout=args.dropout,
        router_mode=args.router_mode,
        use_augmented_features=args.use_augmented_features,
        residue_feature_dim=args.residue_feature_dim,
        use_true_features=args.use_true_features,
        true_feature_dim=args.true_feature_dim,
        surface_points=args.s1_surface_points,
        surface_graph_k=args.s1_surface_graph_k,
        surface_extra_graph_k=args.surface_extra_graph_k,
        surface_extra_graph_ks=args.surface_extra_graph_ks,
        surface_graph_flow_mode=args.surface_graph_flow_mode,
        surface_layers=args.s1_surface_layers,
        surface_residue_topk=args.s1_surface_residue_topk,
        residue_surface_topk=args.s1_residue_surface_topk,
        sigma_tangent=args.s1_sigma_tangent,
        sigma_normal=args.s1_sigma_normal,
        projection_mode=args.s1_projection_mode,
        multiscale_sigmas=args.s1_multiscale_sigmas,
        surface_attention_prior_weight=args.s1_surface_attention_prior_weight,
        surface_attention_sigma_tangent=args.s1_surface_attention_sigma_tangent,
        surface_attention_sigma_normal=args.s1_surface_attention_sigma_normal,
        surface_to_residue_mode=args.surface_to_residue_mode,
        phys_feature_dim=args.phys_feature_dim,
        phys_attention_bias_weight=args.phys_attention_bias_weight,
        phys_context_weight=args.phys_context_weight,
        phys_message_weight=args.phys_message_weight,
        surface_patch_mode=args.surface_patch_mode,
        surface_patch_count=args.surface_patch_count,
        surface_patch_beta=args.surface_patch_beta,
        surface_patch_attention_heads=args.surface_patch_attention_heads,
        surface_lappe_dim=args.s1_surface_lappe_dim,
        surface_lappe_mode=args.surface_lappe_mode,
        surface_spectral_filter_dim=args.surface_spectral_filter_dim,
        internal_operator_mode=args.internal_operator_mode,
        internal_operator_k=args.internal_operator_k,
        internal_operator_layers=args.internal_operator_layers,
        internal_operator_beta=args.internal_operator_beta,
        surface_encoder_mode=args.surface_encoder_mode,
        surface_vector_bridge_mode=args.surface_vector_bridge_mode,
        surface_co_update_mode=args.surface_co_update_mode,
        surface_co_update_layers=args.surface_co_update_layers,
        surface_vector_state_mode=args.surface_vector_state_mode,
        patch_state_mode=args.patch_state_mode,
        surface_serial_stage_mode=args.surface_serial_stage_mode,
        surface_round_end_r2s_mode=args.surface_round_end_r2s_mode,
        post_final_residue_global_feedback_rounds=args.post_final_residue_global_feedback_rounds,
        final_residue_skip_mode=args.final_residue_skip_mode,
        final_residue_skip_source=args.final_residue_skip_source,
        final_residue_skip_beta=args.final_residue_skip_beta,
        final_residue_skip_init_zero=args.final_residue_skip_init_zero,
        pair_gaussian_mode=args.pair_gaussian_mode,
        pair_gaussian_topk=args.pair_gaussian_topk,
        pair_gaussian_beta=args.pair_gaussian_beta,
        pair_gaussian_sigma_tangent=args.pair_gaussian_sigma_tangent,
        pair_gaussian_sigma_normal=args.pair_gaussian_sigma_normal,
        pair_gaussian_sigma_mode=args.pair_gaussian_sigma_mode,
        pair_gaussian_sigma_min=args.pair_gaussian_sigma_min,
        pair_gaussian_beta_mode=args.pair_gaussian_beta_mode,
        pair_gaussian_beta_min=args.pair_gaussian_beta_min,
        pair_gaussian_beta_max=args.pair_gaussian_beta_max,
        pair_gaussian_beta_init=args.pair_gaussian_beta_init,
        pair_gaussian_content_mode=args.pair_gaussian_content_mode,
        pair_gaussian_content_weight=args.pair_gaussian_content_weight,
        pair_gaussian_rho_condition_mode=args.pair_gaussian_rho_condition_mode,
        pair_gaussian_vector_condition_mode=args.pair_gaussian_vector_condition_mode,
        pair_gaussian_single_compete_mode=args.pair_gaussian_single_compete_mode,
        pair_gaussian_single_rescue_beta=args.pair_gaussian_single_rescue_beta,
        pair_gaussian_chunk_size=args.pair_gaussian_chunk_size,
        pair_gaussian_chunk_checkpoint=not args.no_pair_gaussian_chunk_checkpoint,
        pair_gaussian_chunk_diagnostics=args.pair_gaussian_chunk_diagnostics,
        pair_gaussian_only=args.pair_gaussian_only,
        pair_gaussian_fusion_mode=args.pair_gaussian_fusion_mode,
        pair_gaussian_joint_mode=args.pair_gaussian_joint_mode,
        pair_gaussian_multiview_mode=args.pair_gaussian_multiview_mode,
        pair_gaussian_multiview_beta=args.pair_gaussian_multiview_beta,
        pair_gaussian_multiview_chunk_size=args.pair_gaussian_multiview_chunk_size,
        triplet_gaussian_mode=args.triplet_gaussian_mode,
        triplet_gaussian_topk=args.triplet_gaussian_topk,
        triplet_gaussian_beta=args.triplet_gaussian_beta,
        triplet_gaussian_beta_mode=args.triplet_gaussian_beta_mode,
        triplet_gaussian_beta_min=args.triplet_gaussian_beta_min,
        triplet_gaussian_beta_max=args.triplet_gaussian_beta_max,
        triplet_gaussian_beta_init=args.triplet_gaussian_beta_init,
        triplet_gaussian_sigma_tangent=args.triplet_gaussian_sigma_tangent,
        triplet_gaussian_sigma_normal=args.triplet_gaussian_sigma_normal,
        triplet_gaussian_fusion_mode=args.triplet_gaussian_fusion_mode,
        triplet_gaussian_score_temperature=args.triplet_gaussian_score_temperature,
        triplet_gaussian_score_clamp=args.triplet_gaussian_score_clamp,
        triplet_gaussian_preselect_topm=args.triplet_gaussian_preselect_topm,
        residue_amplitude_mode=args.residue_amplitude_mode,
        residue_amplitude_min=args.residue_amplitude_min,
        residue_amplitude_max=args.residue_amplitude_max,
        surface_fusion_mode=args.surface_fusion_mode,
        global_feedback_rounds=args.global_feedback_rounds,
        global_feedback_interleave_residue_graph=args.global_feedback_interleave_residue_graph,
        surface_global_feedback_before_residue=args.surface_global_feedback_before_residue,
        surface_global_feedback_rounds=args.surface_global_feedback_rounds,
        core_cycles=args.core_cycles,
        surface_delta_scale=args.s1_surface_delta_scale,
        residue_vector_surface_mode=args.residue_vector_surface_mode,
        residue_vector_surface_topk=args.residue_vector_surface_topk,
        teacher_hidden_dim=args.teacher_hidden_dim,
        gear_mode=args.gear_mode,
        gear_relation_k=args.gear_relation_k,
        gear_layers=args.gear_layers,
        gear_view_mode=args.gear_view_mode,
        oagnn_mode=args.oagnn_mode,
        oagnn_relation_k=args.oagnn_relation_k,
        oagnn_layers=args.oagnn_layers,
        token_encoder_mode=args.token_encoder_mode,
        token_encoder_layers=args.token_encoder_layers,
        token_encoder_heads=args.token_encoder_heads,
        edge_pair_mode=args.edge_pair_mode,
        cross_view_edge_mode=args.cross_view_edge_mode,
        cross_view_edge_beta=args.cross_view_edge_beta,
        cross_view_edge_heads=args.cross_view_edge_heads,
        residue_encoder_mode=args.residue_encoder_mode,
        residue_graph_flow_mode=args.residue_graph_flow_mode,
        sequence_pos_enc_mode=args.sequence_pos_enc_mode,
        sequence_pos_enc_dim=args.sequence_pos_enc_dim,
        atom_residue_encoder_mode=args.atom_residue_encoder_mode,
        atom_graph_k=args.atom_graph_k,
        atom_contact_k=args.atom_contact_k,
        atom_contact_radius=args.atom_contact_radius,
        atom_pre_encoder_mode=args.atom_pre_encoder_mode,
        atom_pre_dim=args.atom_pre_dim,
        atom_pre_layers=args.atom_pre_layers,
        atom_pre_topk=args.atom_pre_topk,
        atom_pre_cutoff=args.atom_pre_cutoff,
        atom_pre_dropout=args.atom_pre_dropout,
        post_final_fourview_mode=args.post_final_fourview_mode,
        post_final_fourview_beta=args.post_final_fourview_beta,
        surface_bus_mode=args.surface_bus_mode,
        surface_bus_beta=args.surface_bus_beta,
        surface_bus_topk=args.surface_bus_topk,
        struct_community_mode=args.struct_community_mode,
        struct_community_count=args.struct_community_count,
        struct_community_geom_bias=args.struct_community_geom_bias,
        stochastic_head_mode=args.stochastic_head_mode,
        ctmc_diffusion_steps=args.ctmc_diffusion_steps,
        ctmc_diffusion_gamma=args.ctmc_diffusion_gamma,
        ctmc_diffusion_topk=args.ctmc_diffusion_topk,
        poisson_lambda_max=args.poisson_lambda_max,
        pocket_mass_topk=args.pocket_mass_topk,
        motif_patch_mode=args.motif_patch_mode,
        motif_patch_count=args.motif_patch_count,
        motif_patch_beta=args.motif_patch_beta,
        motif_patch_attention_heads=args.motif_patch_attention_heads,
        motif_patch_geom_bias=args.motif_patch_geom_bias,
        semantic_patch_mode=args.semantic_patch_mode,
        semantic_patch_query_count=args.semantic_patch_query_count,
        semantic_patch_beta=args.semantic_patch_beta,
        semantic_patch_radius_min=args.semantic_patch_radius_min,
        semantic_patch_radius_max=args.semantic_patch_radius_max,
        semantic_patch_adjacency_weight=args.semantic_patch_adjacency_weight,
        semantic_patch_active_topk=args.semantic_patch_active_topk,
        semantic_patch_graph_hops=args.semantic_patch_graph_hops,
        semantic_patch_geodesic_max_steps=args.semantic_patch_geodesic_max_steps,
        semantic_patch_geodesic_normal_penalty=args.semantic_patch_geodesic_normal_penalty,
        semantic_patch_geodesic_tangent_penalty=args.semantic_patch_geodesic_tangent_penalty,
        semantic_patch_detach_backbone=args.semantic_patch_detach_backbone,
        semantic_patch_boundary_mix=args.semantic_patch_boundary_mix,
        semantic_patch_boundary_temperature=args.semantic_patch_boundary_temperature,
    )


def patch_diagnostics(out, batch):
    if "patch_assignment_entropy" not in out:
        return {}
    metrics = {
        "patch_assignment_entropy": metric_scalar(out["patch_assignment_entropy"].detach().float().mean()),
        "patch_usage_active_mean": metric_scalar(out["patch_usage_active"].detach().float().mean()),
    }
    for key in (
        "patch_usage_entropy", "patch_spatial_compactness", "patch_normal_consistency",
        "patch_token_norm_mean", "patch_token_diversity", "patch_residue_attention_entropy",
        "patch_residue_attention_max", "patch_residue_effective_count", "patch_context_update_norm",
        "patch_residue_gate_mean", "patch_residue_gate_std", "connected_patch_components_mean",
        "connected_patch_components_max", "connected_patch_size_min", "connected_patch_size_max",
        "connected_patch_size_cv", "connected_patch_boundary_fraction", "connected_patch_degree_mean",
        "connected_patch_hop_radius_mean", "connected_patch_hop_radius_max",
    ):
        if key in out:
            metrics[key] = metric_scalar(out[key].detach().float().mean())
    if "patch_surface_gate" not in out or "patch_gate" not in out or "patch_primary_residue_idx" not in out:
        return metrics
    gate = out["patch_surface_gate"].detach().float()
    idx = out["patch_primary_residue_idx"].detach().long().clamp(0, max(int(batch["y"].numel()) - 1, 0))
    labels = batch["y"].detach().float()[idx]
    valid = batch["mask"].detach().float()[idx] > 0.5
    metrics.update({
        "patch_surface_gate_mean": metric_scalar(gate.mean()),
        "patch_surface_gate_std": metric_scalar(gate.std(unbiased=False)),
        "patch_gate_mean": metric_scalar(out["patch_gate"].detach().float().mean()),
        "patch_gate_std": metric_scalar(out["patch_gate"].detach().float().std(unbiased=False)),
    })
    if pt.any(valid):
        g = gate[valid]
        y = labels[valid]
        metrics["patch_nearest_positive_rate"] = metric_scalar(y.mean())
        if g.numel() > 2 and float(g.std(unbiased=False).item()) > 1e-8 and float(y.std(unbiased=False).item()) > 1e-8:
            corr = pt.corrcoef(pt.stack([g, y]))[0, 1].clamp(-1.0, 1.0)
            metrics["patch_gate_label_corr"] = metric_scalar(corr)
        else:
            metrics["patch_gate_label_corr"] = "nan"
        cutoff = pt.quantile(g, 0.8) if g.numel() >= 5 else g.max()
        high = g >= cutoff
        metrics["patch_high_gate_positive_rate"] = metric_scalar(y[high].mean() if pt.any(high) else y.mean())
        metrics["patch_high_gate_enrichment"] = metric_scalar((y[high].mean() / y.mean().clamp_min(1e-6)) if pt.any(high) else pt.tensor(1.0, device=y.device))
    return metrics


SIGMA_DIAGNOSTIC_KEYS = (
    "sigma_global_weight_entropy",
    "sigma_global_weight_max",
    "sigma_tangent_mean",
    "sigma_tangent_min",
    "sigma_tangent_max",
    "sigma_tangent_std",
    "sigma_normal_mean",
    "sigma_normal_min",
    "sigma_normal_max",
    "sigma_normal_std",
    "surface_to_residue_sigma_tangent_mean",
    "surface_to_residue_sigma_tangent_std",
    "surface_to_residue_sigma_normal_mean",
    "surface_to_residue_sigma_normal_std",
    "surface_to_residue_geoparam_sigma_tangent_mean",
    "surface_to_residue_geoparam_sigma_tangent_min",
    "surface_to_residue_geoparam_sigma_tangent_max",
    "surface_to_residue_geoparam_sigma_tangent_std",
    "surface_to_residue_geoparam_sigma_normal_mean",
    "surface_to_residue_geoparam_sigma_normal_min",
    "surface_to_residue_geoparam_sigma_normal_max",
    "surface_to_residue_geoparam_sigma_normal_std",
    "surface_to_residue_geoparam_gamma_mean",
    "surface_to_residue_geoparam_gamma_std",
    "surface_to_residue_geoparam_prior_mean",
    "surface_to_residue_geoparam_prior_std",
    "phys_field_abs_mean",
    "phys_field_std",
    "mixture_ball_entropy",
    "mixture_ball_max_weight",
    "local3d_attention_entropy",
    "local3d_attention_max",
    "atom_pointcloud_attention_entropy",
    "atom_pointcloud_attention_max",
    "atom_pointcloud_gate_mean",
    "atom_residue_encoder_enabled",
    "atom_residue_attention_entropy",
    "atom_residue_attention_max",
    "atom_residue_gate_mean",
    "atom_residue_update_norm",
    "atom_residue_graph_update_norm",
    "atom_residue_contact_update_norm",
    "pointcloud_patch_assignment_entropy",
    "pointcloud_patch_usage_active",
    "pointcloud_patch_gate_mean",
    "pointcloud_patch_gate_std",
    "gear_edge_count_mean",
    "gear_relation_entropy",
    "gear_attention_entropy",
    "gear_update_norm",
    "gear_view_edge_update_norm",
    "gear_view_relation_entropy",
    "gear_view_edge_count_mean",
    "oagnn_edge_count_mean",
    "oagnn_relation_entropy",
    "oagnn_attention_entropy",
    "oagnn_update_norm",
    "oagnn_direction_weight_norm",
    "oagnn_cross_weight_norm",
    "oagnn_local_dir_std",
    "official_oagnn_edge_count_mean",
    "official_oagnn_message_norm",
    "official_oagnn_vector_norm",
    "official_oagnn_frame_valid_mean",
    "surface_graph_view_count",
    "surface_graph_router_entropy",
    "surface_graph_flow_serial",
    "surface_graph_serial_step1_k",
    "surface_graph_serial_step2_k",
    "surface_graph_weight_k16",
    "surface_graph_weight_k8",
    "surface_graph_weight_k32",
    "surface_global_feedback_update_norm",
    "token_encoder_gate_mean",
    "token_encoder_gate_std",
    "token_encoder_update_norm",
    "edge_pair_update_norm",
    "edge_pair_gate_mean",
    "edge_pair_count_mean",
    "pair_gaussian_entropy",
    "pair_gaussian_weight_max",
    "pair_gaussian_beta_eff",
    "pair_gaussian_beta_init",
    "pair_gaussian_beta_deep",
    "pair_gaussian_beta_round0",
    "pair_gaussian_beta_round1",
    "pair_gaussian_beta_round2",
    "pair_gaussian_beta_round3",
    "pair_gaussian_beta_round4",
    "pair_gaussian_joint_rho_abs",
    "pair_gaussian_rho_dot_i_abs",
    "pair_gaussian_rho_dot_j_abs",
    "pair_gaussian_rho_dot_pair_abs",
    "pair_gaussian_vector_condition_abs",
    "pair_gaussian_vector_condition_std",
    "pair_gaussian_vector_raw_abs",
    "pair_gaussian_vector_raw_std",
    "pair_gaussian_vector_raw_max_abs",
    "pair_gaussian_vector_raw_finite_frac",
    "pair_gaussian_vector_proj_abs",
    "pair_gaussian_vector_proj_std",
    "pair_gaussian_vector_proj_max_abs",
    "pair_gaussian_vector_proj_finite_frac",
    "pair_gaussian_vector_v_norm_mean",
    "pair_gaussian_vector_v_norm_min",
    "pair_gaussian_vector_v_norm_max",
    "pair_gaussian_vector_dot_n_abs",
    "pair_gaussian_vector_dot_n_max_abs",
    "pair_gaussian_vector_dot_r_abs",
    "pair_gaussian_vector_dot_axis_abs",
    "pair_gaussian_vector_tangent_mean",
    "pair_gaussian_vector_tangent_min",
    "pair_gaussian_vector_tangent_sq_min",
    "pair_gaussian_vector_tangent_sq_mean",
    "pair_gaussian_vector_tangent_sq_zero_frac",
    "pair_gaussian_vector_tangent_sq_lt1em12_frac",
    "pair_gaussian_vector_tangent_sq_lt1em8_frac",
    "pair_gaussian_vector_tangent_grad_risk_max",
    "pair_gaussian_coupling_abs",
    "pair_gaussian_coupling_mean",
    "pair_gaussian_coattn_pair_weight",
    "pair_gaussian_coattn_single_weight",
    "pair_gaussian_sigma_tangent_mean",
    "pair_gaussian_sigma_tangent_min",
    "pair_gaussian_sigma_tangent_max",
    "pair_gaussian_sigma_tangent_std",
    "pair_gaussian_sigma_normal_mean",
    "pair_gaussian_sigma_normal_min",
    "pair_gaussian_sigma_normal_max",
    "pair_gaussian_sigma_normal_std",
    "pair_gaussian_content_score_mean",
    "pair_gaussian_content_score_std",
    "pair_gaussian_content_score_abs",
    "pair_gaussian_content_distance_weight_mean",
    "pair_gaussian_single_rescue_gate_mean",
    "pair_gaussian_single_rescue_beta",
    "pair_gaussian_strong_gate_single_mean",
    "pair_gaussian_strong_gate_pair_mean",
    "pair_gaussian_multiview_gate_mean",
    "pair_gaussian_multiview_update_norm",
    "pair_gaussian_multiview_view_count",
    "pair_gaussian_multiview_pair_gaussian_entropy",
    "pair_gaussian_multiview_pair_gaussian_weight_max",
    "pair_gaussian_multiview_pair_gaussian_joint_rho_abs",
    "pair_gaussian_multiview_pair_gaussian_coupling_abs",
    "pair_gaussian_multiview_pair_gaussian_coupling_mean",
    "pair_gaussian_multiview_pair_gaussian_sigma_tangent_mean",
    "pair_gaussian_multiview_pair_gaussian_sigma_tangent_std",
    "pair_gaussian_multiview_pair_gaussian_sigma_normal_mean",
    "pair_gaussian_multiview_pair_gaussian_sigma_normal_std",
    "triplet_gaussian_entropy",
    "triplet_gaussian_weight_max",
    "triplet_gaussian_beta_eff",
    "triplet_gaussian_beta_init",
    "triplet_gaussian_beta_deep",
    "triplet_gaussian_beta_round0",
    "triplet_gaussian_beta_round1",
    "triplet_gaussian_beta_round2",
    "triplet_gaussian_beta_round3",
    "triplet_gaussian_beta_round4",
    "triplet_gaussian_rho_abs",
    "triplet_gaussian_coupling_abs",
    "triplet_gaussian_coupling_mean",
    "triplet_gaussian_independent_score_mean",
    "triplet_gaussian_joint_score_mean",
    "triplet_gaussian_score_for_weight_mean",
    "triplet_gaussian_score_temperature",
    "triplet_gaussian_score_clamp",
    "triplet_gaussian_raw_triplet_count",
    "triplet_gaussian_preselect_topm",
    "triplet_gaussian_gate_mean",
    "triplet_gaussian_update_norm",
    "triplet_gaussian_triplet_count",
    "residue_vector_surface_attn_entropy",
    "residue_vector_surface_attn_max",
    "residue_vector_surface_value_norm",
    "residue_vector_surface_topk",
    "struct_comm_active_mean",
    "struct_comm_surface_entropy",
    "struct_comm_surface_gate_mean",
    "struct_comm_residue_gate_mean",
    "motif_patch_motif_count",
    "motif_patch_active_mean",
    "motif_patch_motif_patch_entropy",
    "motif_patch_patch_motif_entropy",
    "motif_patch_residue_patch_entropy",
    "motif_patch_residue_gate_mean",
    "motif_patch_context_gate_mean",
    "post_final_fourview_router_entropy",
    "post_final_fourview_update_norm",
    "post_final_dual_surface_gate_mean",
    "post_final_dual_graph_gate_mean",
    "post_final_graph_logit_abs",
    "post_final_surface_logit_abs",
    "post_final_residue_global_feedback_update_norm",
    "ctmc_rate_mean",
    "ctmc_rate_std",
    "ctmc_rate_max",
    "ctmc_gate_mean",
    "ctmc_gate_std",
    "ctmc_gate_min",
    "ctmc_gate_max",
    "ctmc_update_norm",
    "ctmc_update_rel_norm",
    "ctmc_weight_entropy",
    "ctmc_weight_max",
    "ctmc_logit_delta_abs",
    "ctmc_hidden_delta_norm",
    "poisson_lambda_mean",
    "poisson_lambda_min",
    "poisson_lambda_max",
    "poisson_lambda_std",
    "event_rate_open_mean",
    "event_rate_open_min",
    "event_rate_open_max",
    "event_rate_open_std",
    "event_rate_contact_mean",
    "event_rate_contact_min",
    "event_rate_contact_max",
    "event_rate_contact_std",
    "event_rate_rigid_mean",
    "event_rate_rigid_min",
    "event_rate_rigid_max",
    "event_rate_rigid_std",
    "event_rate_collapse_mean",
    "event_rate_collapse_min",
    "event_rate_collapse_max",
    "event_rate_collapse_std",
    "surface_bus_view_entropy",
    "surface_bus_view_max",
    "surface_bus_update_norm",
    "surface_bus_view_count",
    "surface_bus_residue_entropy",
    "surface_bus_pre_global_update_norm",
    "surface_bus_recurrent_rounds",
    "surface_bus_view_update_norm",
    "surface_bus_view_gate_mean",
    "surface_bus_view_router_entropy",
    "fullcov_sigma_t1_mean",
    "fullcov_sigma_t1_min",
    "fullcov_sigma_t1_max",
    "fullcov_sigma_t2_mean",
    "fullcov_sigma_t2_min",
    "fullcov_sigma_t2_max",
    "fullcov_sigma_normal_mean",
    "fullcov_sigma_normal_min",
    "fullcov_sigma_normal_max",
    "fullcov_offdiag_10_mean",
    "fullcov_offdiag_20_mean",
    "fullcov_offdiag_21_mean",
    "residue_amplitude_mean",
    "residue_amplitude_min",
    "residue_amplitude_max",
    "residue_amplitude_std",
    "internal_operator_diffusion_norm",
    "atom_pre_encoder_enabled",
    "atom_pre_edge_dist_mean",
    "atom_pre_update_norm",
    "atom_pre_gate_mean",
    "atom_pre_bb_present_frac",
    "atom_pre_sc_present_frac",
    "final_res_skip_gate_mean",
    "final_res_skip_gate_std",
    "final_res_skip_update_norm",
    "final_res_skip_cos_to_final",
    "final_res_skip_delta_logit_abs",
    "final_res_skip_delta_logit_pos_mean",
    "final_res_skip_delta_logit_neg_mean",
)


def sigma_diagnostics(out):
    metrics = {}
    for key in SIGMA_DIAGNOSTIC_KEYS:
        value = out.get(key)
        if value is not None:
            metrics[key] = metric_scalar(value.detach().float())
    return metrics


def pocket_mass_loss_from_batch(out, batch, pos_weight, args):
    if str(getattr(args, "stochastic_head_mode", "none")) != "poisson_pocket" or float(getattr(args, "pocket_mass_loss_weight", 0.0)) <= 0:
        return out["logits"].sum() * 0.0
    emb = out.get("residue_embeddings")
    if emb is None:
        return out["logits"].sum() * 0.0
    logits = out["logits"]
    y = batch["y"].to(device=logits.device, dtype=logits.dtype)
    mask = batch["mask"].to(device=logits.device, dtype=logits.dtype)
    counts = batch["residue_counts"].detach().cpu().tolist()
    losses = []
    offset = 0
    topk = max(1, int(getattr(args, "pocket_mass_topk", 16)))
    for count in counts:
        n = int(count)
        if n <= 1:
            offset += n
            continue
        sl = slice(offset, offset + n)
        h = emb[sl].to(device=logits.device, dtype=logits.dtype)
        local_logits = logits[sl]
        local_y = y[sl]
        local_mask = mask[sl]
        k = min(topk, n)
        dist = pt.cdist(h, h)
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        prob = pt.sigmoid(local_logits).clamp(1e-6, 1.0 - 1e-6)
        lam = -pt.log1p(-prob).clamp_max(float(getattr(args, "poisson_lambda_max", 20.0)))
        region_lam = lam[idx].sum(dim=-1).clamp_max(float(getattr(args, "poisson_lambda_max", 20.0)))
        region_prob = (1.0 - pt.exp(-region_lam)).clamp(1e-6, 1.0 - 1e-6)
        region_logit = pt.logit(region_prob)
        region_y = local_y[idx].max(dim=-1).values
        if pt.any(local_mask > 0.5):
            losses.append(masked_bce(region_logit, region_y, local_mask, pos_weight))
        offset += n
    if not losses:
        return out["logits"].sum() * 0.0
    return pt.stack(losses).mean()

def surface_soft_bce_from_batch(out, batch, pos_weight, args):
    if float(args.surface_soft_loss_weight) <= 0:
        return out["logits"].sum() * 0.0
    losses = []
    atom_offset = 0
    residue_offset = 0
    surface_offset = 0
    for sample_idx, residue_count_t in enumerate(batch["residue_counts"]):
        atom_count = int(batch["atom_counts"][sample_idx].item())
        residue_count = int(residue_count_t.item())
        x = batch["X"][atom_offset : atom_offset + atom_count]
        atom_index = batch["atom_to_residue_index"][atom_offset : atom_offset + atom_count] - residue_offset
        labels = batch["y"][residue_offset : residue_offset + residue_count]
        mask = batch["mask"][residue_offset : residue_offset + residue_count]
        cache = batch.get("s1_surface_caches", [None] * len(batch["residue_counts"]))[sample_idx]
        if cache is None:
            atom_offset += atom_count
            residue_offset += residue_count
            continue
        surface_pos = cache["surface_pos"].to(device=out["surface_logits"].device, dtype=out["surface_logits"].dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is not None:
            surface_normal = surface_normal.to(device=surface_pos.device, dtype=surface_pos.dtype)
        surface_count = int(surface_pos.shape[0])
        x_float = x.float()
        centers = x_float.new_zeros((residue_count, 3))
        centers.index_add_(0, atom_index.long(), x_float)
        counts = x_float.new_zeros((residue_count, 1))
        counts.index_add_(0, atom_index.long(), x_float.new_ones((atom_count, 1)))
        centers = (centers / counts.clamp_min(1.0)).to(device=surface_pos.device, dtype=surface_pos.dtype)
        target = surface_soft_labels(
            labels.to(device=surface_pos.device),
            mask.to(device=surface_pos.device),
            centers,
            surface_pos,
            surface_normal=surface_normal,
            sigma_tangent=args.surface_soft_sigma_tangent,
            sigma_normal=args.surface_soft_sigma_normal,
            max_value=args.surface_soft_max_value,
            cutoff=args.surface_soft_cutoff,
            mode=args.surface_soft_label_mode,
            topk=args.surface_soft_label_topk,
        )
        logits = out["surface_logits"][surface_offset : surface_offset + surface_count]
        losses.append(masked_soft_bce(logits, target, mask=None, pos_weight=pos_weight))
        atom_offset += atom_count
        residue_offset += residue_count
        surface_offset += surface_count
    if not losses:
        return out["logits"].sum() * 0.0
    return pt.stack(losses).mean()


def motif_patch_aux_bce_from_batch(out, batch, pos_weight, args):
    if float(args.motif_patch_aux_loss_weight) <= 0:
        return out["logits"].sum() * 0.0
    losses = []
    y = batch["y"]
    mask = batch["mask"]
    for logits_key, idx_key in (
        ("motif_patch_motif_logits", "motif_patch_motif_primary_idx"),
        ("motif_patch_patch_logits", "motif_patch_patch_primary_idx"),
    ):
        logits = out.get(logits_key)
        idx = out.get(idx_key)
        if logits is None or idx is None or logits.numel() == 0 or idx.numel() == 0:
            continue
        idx = idx.to(device=y.device, dtype=pt.long).clamp(0, max(int(y.numel()) - 1, 0))
        target = y[idx].to(device=logits.device, dtype=logits.dtype)
        valid = mask[idx].to(device=logits.device, dtype=logits.dtype)
        losses.append(masked_bce(logits, target, valid, pos_weight.to(device=logits.device, dtype=logits.dtype)))
    if not losses:
        return out["logits"].sum() * 0.0
    return pt.stack(losses).mean()


def _positive_components(labels, valid, centers, graph_k, max_components):
    residue_count = int(labels.numel())
    positive = (labels > 0.5) & (valid > 0.5)
    positive_idx = pt.nonzero(positive, as_tuple=False).flatten()
    if positive_idx.numel() == 0:
        return []
    if residue_count < 2:
        return [positive.to(dtype=labels.dtype)]
    k = min(max(int(graph_k), 0), residue_count - 1)
    if k == 0:
        return [positive.to(dtype=labels.dtype)]
    distance = pt.cdist(centers.float(), centers.float())
    distance.fill_diagonal_(float("inf"))
    neighbors = pt.topk(distance, k=k, dim=-1, largest=False).indices
    adjacency = pt.zeros((residue_count, residue_count), device=centers.device, dtype=pt.bool)
    rows = pt.arange(residue_count, device=centers.device).unsqueeze(1).expand_as(neighbors)
    adjacency[rows, neighbors] = True
    adjacency = adjacency | adjacency.transpose(0, 1)
    positive_cpu = positive.detach().cpu()
    adjacency_cpu = adjacency.detach().cpu()
    remaining = set(int(index) for index in positive_idx.detach().cpu().tolist())
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        members = [start]
        while stack:
            current = stack.pop()
            for neighbor in pt.nonzero(adjacency_cpu[current], as_tuple=False).flatten().tolist():
                neighbor = int(neighbor)
                if bool(positive_cpu[neighbor]) and neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    members.append(neighbor)
        component = labels.new_zeros((residue_count,))
        component[members] = 1.0
        components.append(component)
    components.sort(key=lambda item: float(item.sum().item()), reverse=True)
    max_components = max(1, int(max_components))
    if len(components) <= max_components:
        return components
    kept = components[:max_components]
    for extra in components[max_components:]:
        extra_idx = pt.nonzero(extra > 0.5, as_tuple=False).flatten()
        extra_center = centers[extra_idx].float().mean(dim=0, keepdim=True)
        distances = []
        for component in kept:
            component_idx = pt.nonzero(component > 0.5, as_tuple=False).flatten()
            component_center = centers[component_idx].float().mean(dim=0, keepdim=True)
            distances.append(pt.linalg.norm(extra_center - component_center).item())
        kept[int(min(range(len(distances)), key=distances.__getitem__))] = pt.maximum(
            kept[int(min(range(len(distances)), key=distances.__getitem__))], extra
        )
    return kept


def _match_patch_masks(predicted, targets, valid):
    if not targets:
        return []
    target = pt.stack(targets, dim=0).to(device=predicted.device, dtype=predicted.dtype)
    valid = valid.to(device=predicted.device, dtype=predicted.dtype).unsqueeze(0)
    pred = predicted.unsqueeze(1)
    truth = target.unsqueeze(0)
    intersection = (pred * truth * valid).sum(dim=-1)
    denominator = ((pred + truth) * valid).sum(dim=-1).clamp_min(1e-6)
    dice_cost = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    cost = dice_cost.detach().cpu()
    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(cost.numpy())
        return list(zip(rows.tolist(), columns.tolist()))
    matches = []
    used_rows = set()
    used_columns = set()
    flat_order = pt.argsort(cost.flatten()).tolist()
    target_count = int(target.shape[0])
    for flat_index in flat_order:
        row = int(flat_index // target_count)
        column = int(flat_index % target_count)
        if row in used_rows or column in used_columns:
            continue
        matches.append((row, column))
        used_rows.add(row)
        used_columns.add(column)
        if len(used_columns) == target_count:
            break
    return matches


def semantic_patch_losses(out, batch, args):
    outputs = out.get("semantic_patch_outputs")
    zero = out["logits"].sum() * 0.0
    result = {
        "amplitude": zero,
        "set": zero,
        "existence": zero,
        "connectivity": zero,
    }
    if not outputs:
        return result
    amplitude_losses = []
    set_losses = []
    existence_losses = []
    connectivity_losses = []
    residue_offset = 0
    for semantic_output in outputs:
        residue_count = int(semantic_output["region_to_residue"].shape[-1])
        labels = batch["y"][residue_offset : residue_offset + residue_count]
        valid = batch["mask"][residue_offset : residue_offset + residue_count]
        region_weights = semantic_output["region_to_residue"]
        valid_region_weights = region_weights * valid.to(region_weights.dtype).unsqueeze(0)
        region_target = (
            valid_region_weights * labels.to(region_weights.dtype).unsqueeze(0)
        ).sum(dim=-1) / valid_region_weights.sum(dim=-1).clamp_min(1e-8)
        amplitude_losses.append(
            pt.nn.functional.binary_cross_entropy_with_logits(
                semantic_output["region_amplitude_logits"], region_target
            )
        )
        region_existence_logits = semantic_output.get("region_existence_logits")
        if region_existence_logits is not None:
            existence_threshold = max(float(args.semantic_patch_region_existence_threshold), 1e-6)
            region_existence_target = (region_target / existence_threshold).clamp(0.0, 1.0)
            existence_losses.append(
                pt.nn.functional.binary_cross_entropy_with_logits(
                    region_existence_logits, region_existence_target
                )
            )
        predicted_masks = semantic_output.get("set_residue_masks")
        existence_logits = semantic_output.get("set_existence_logits")
        if predicted_masks is not None and existence_logits is not None:
            targets = _positive_components(
                labels,
                valid,
                semantic_output["residue_centers"],
                graph_k=args.semantic_patch_component_k,
                max_components=int(predicted_masks.shape[0]),
            )
            matches = _match_patch_masks(predicted_masks, targets, valid)
            existence_target = pt.zeros_like(existence_logits)
            if matches:
                target_stack = pt.stack(targets, dim=0).to(
                    device=predicted_masks.device, dtype=predicted_masks.dtype
                )
                matched_losses = []
                valid_float = valid.to(device=predicted_masks.device, dtype=predicted_masks.dtype)
                for query_index, target_index in matches:
                    existence_target[query_index] = 1.0
                    prediction = predicted_masks[query_index]
                    target = target_stack[target_index]
                    with pt.cuda.amp.autocast(enabled=False):
                        prediction_float = prediction.float()
                        target_float = target.float()
                        valid_float32 = valid_float.float()
                        bce = pt.nn.functional.binary_cross_entropy(
                            prediction_float,
                            target_float,
                            reduction="none",
                        )
                        bce = (bce * valid_float32).sum() / valid_float32.sum().clamp_min(1.0)
                        intersection = (prediction_float * target_float * valid_float32).sum()
                        denominator = (
                            (prediction_float + target_float) * valid_float32
                        ).sum().clamp_min(1e-6)
                        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
                    matched_losses.append(bce + dice)
                set_losses.append(pt.stack(matched_losses).mean())
            existence_losses.append(
                pt.nn.functional.binary_cross_entropy_with_logits(
                    existence_logits, existence_target
                )
            )
            query_region_weights = semantic_output["set_region_weights"]
            adjacency = semantic_output["adjacency"].to(
                device=query_region_weights.device, dtype=pt.bool
            )
            non_adjacent = (~adjacency).to(query_region_weights.dtype)
            pair_mass = query_region_weights.unsqueeze(-1) * query_region_weights.unsqueeze(-2)
            connectivity_losses.append((pair_mass * non_adjacent.unsqueeze(0)).sum(dim=(-1, -2)).mean())
        boundary_tv = semantic_output.get("semantic_boundary_tv")
        if boundary_tv is not None:
            connectivity_losses.append(boundary_tv)
        residue_offset += residue_count
    if amplitude_losses:
        result["amplitude"] = pt.stack(amplitude_losses).mean()
    if set_losses:
        result["set"] = pt.stack(set_losses).mean()
    if existence_losses:
        result["existence"] = pt.stack(existence_losses).mean()
    if connectivity_losses:
        result["connectivity"] = pt.stack(connectivity_losses).mean()
    return result


def build_scheduler(args, optimizer, num_batches):
    scheduler_epochs = args.scheduler_total_epochs if args.scheduler_total_epochs and args.scheduler_total_epochs > 0 else args.epochs
    total_steps = max(num_batches * max(scheduler_epochs, 1), 1)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0:
        warmup_steps = int(round(total_steps * args.warmup_ratio))
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    min_ratio = float(args.min_learning_rate) / max(float(args.learning_rate), 1e-12)
    if args.scheduler == "none":
        return None, {"scheduler": "none", "total_steps": total_steps, "warmup_steps": warmup_steps}

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        value = 1.0 - progress if args.scheduler == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * value

    return pt.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), {
        "scheduler": args.scheduler,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "min_learning_rate": args.min_learning_rate,
    }


def build_optimizer(args, model):
    gear_params = []
    oagnn_params = []
    base_params = []
    integrated_oagnn = getattr(model, "oagnn_mode", "none") != "none"
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if integrated_oagnn and name.startswith("encoder.views."):
            base_params.append(param)
        elif name.startswith("oagnn_refiner."):
            oagnn_params.append(param)
        elif name.startswith("gear_refiner.") or ".edge_updates." in name or ".edge_norms." in name or ".relation_embedding." in name:
            gear_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if base_params:
        groups.append(
            {
                "params": base_params,
                "lr": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "name": "base",
            }
        )
    if gear_params:
        groups.append(
            {
                "params": gear_params,
                "lr": float(args.learning_rate) * float(args.gear_lr_multiplier),
                "weight_decay": float(args.weight_decay),
                "name": "gear_refiner",
            }
        )
    if oagnn_params:
        groups.append(
            {
                "params": oagnn_params,
                "lr": float(args.learning_rate) * float(args.oagnn_lr_multiplier),
                "weight_decay": float(args.weight_decay),
                "name": "oagnn_refiner",
            }
        )
    return pt.optim.AdamW(groups, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))


def optimizer_group_report(optimizer):
    return [
        {
            "name": group.get("name", f"group_{idx}"),
            "lr": float(group.get("lr", 0.0)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "num_tensors": len(group.get("params", [])),
        }
        for idx, group in enumerate(optimizer.param_groups)
    ]


def named_grad_norms(model, prefixes):
    sums = {prefix: 0.0 for prefix in prefixes}
    counts = {prefix: 0 for prefix in prefixes}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for prefix in prefixes:
            if name.startswith(prefix):
                grad = param.grad.detach().float()
                sums[prefix] += float((grad * grad).sum().cpu())
                counts[prefix] += int(grad.numel())
                break
    metrics = {}
    for prefix in prefixes:
        key = prefix.replace(".", "_").rstrip("_")
        metrics[f"{key}_grad_norm"] = math.sqrt(sums[prefix]) if sums[prefix] > 0.0 else 0.0
        metrics[f"{key}_grad_elems"] = counts[prefix]
    return metrics


def clone_state_for_checkpoint(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def init_ema_state(model):
    return clone_state_for_checkpoint(model.state_dict())


def update_ema_state(ema_state, model, decay):
    with pt.no_grad():
        for key, value in model.state_dict().items():
            value_cpu = value.detach().cpu()
            if key not in ema_state:
                ema_state[key] = value_cpu.clone()
            elif pt.is_floating_point(value_cpu):
                ema_state[key].mul_(decay).add_(value_cpu, alpha=1.0 - decay)
            else:
                ema_state[key].copy_(value_cpu)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, args, ema_state=None):
    raw_model_state = clone_state_for_checkpoint(model.state_dict())
    model_state = clone_state_for_checkpoint(ema_state) if ema_state is not None else raw_model_state
    pt.save(
        {
            "model_state_dict": model_state,
            "raw_model_state_dict": raw_model_state if ema_state is not None else None,
            "ema_model_state_dict": clone_state_for_checkpoint(ema_state) if ema_state is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "args": vars(args),
        },
        path,
    )


def _append_scores(batch, out, all_scores, all_labels, case_rows):
    offset = 0
    for i, count in enumerate(batch["residue_counts"].detach().cpu().tolist()):
        count = int(count)
        sl = slice(offset, offset + count)
        valid = batch["mask"][sl] > 0.5
        if pt.any(valid):
            labels = batch["y"][sl][valid].detach().float().cpu()
            scores = pt.sigmoid(out["logits"][sl][valid]).detach().float().cpu()
            all_labels.append(labels)
            all_scores.append(scores)
            case_id = batch["rows"][i].get("apex_case_id", batch["rows"][i].get("system_id", str(i)))
            row = case_metrics(scores, labels, case_id)
            row["system_id"] = batch["rows"][i].get("system_id", "")
            row["input_state"] = batch["rows"][i].get("input_state", "")
            row["receptor_uniprot"] = batch["rows"][i].get("receptor_uniprot", "")
            case_rows.append(row)
        offset += count


def run_eval(model, loader, device, pos_weight, prefix, args):
    model.eval()
    losses = []
    all_scores = []
    all_labels = []
    case_rows = []
    skipped_examples = 0
    skipped_batches = 0
    base_losses = []
    final_losses = []
    atom_pre_aux_losses = []
    distill_losses = []
    gate_means = []
    gate_stds = []
    delta_stds = []
    patch_diag_values = {}
    with pt.no_grad():
        for batch in tqdm(loader, desc=f"eval_s1_{prefix}"):
            skipped_examples += int(batch.get("skipped_count", 0) or 0)
            if batch.get("skip_batch"):
                skipped_batches += 1
                continue
            batch = move_batch_to_device(batch, device)
            out = model_forward(model, batch)
            loss_final = masked_bce(out["logits"], batch["y"], batch["mask"], pos_weight)
            loss_base = masked_bce(out["base_logits"], batch["y"], batch["mask"], pos_weight)
            if out.get("atom_pre_logits") is not None and float(args.atom_pre_aux_loss_weight) > 0:
                loss_atom_pre_aux = masked_bce(out["atom_pre_logits"], batch["y"], batch["mask"], pos_weight)
            else:
                loss_atom_pre_aux = out["logits"].sum() * 0.0
            loss_distill = teacher_logit_distillation_loss(
                out["logits"],
                batch.get("teacher_logits"),
                batch["mask"],
                weight=args.teacher_logit_weight,
                mode=args.teacher_logit_loss,
                temperature=args.teacher_temperature,
            )
            loss = (
                loss_final
                + float(args.base_aux_loss_weight) * loss_base
                + float(args.atom_pre_aux_loss_weight) * loss_atom_pre_aux
                + loss_distill
            )
            losses.append(float(loss.cpu()))
            final_losses.append(float(loss_final.cpu()))
            base_losses.append(float(loss_base.cpu()))
            atom_pre_aux_losses.append(float(loss_atom_pre_aux.cpu()))
            distill_losses.append(float(loss_distill.cpu()))
            gate_means.append(float(out["surface_gate"].detach().float().mean().cpu()))
            gate_stds.append(float(out["surface_gate"].detach().float().std(unbiased=False).cpu()))
            delta_stds.append(float(out["surface_delta"].detach().float().std(unbiased=False).cpu()))
            for key, value in patch_diagnostics(out, batch).items():
                try:
                    patch_diag_values.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass
            for key, value in sigma_diagnostics(out).items():
                try:
                    patch_diag_values.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass
            _append_scores(batch, out, all_scores, all_labels, case_rows)
    labels = pt.cat(all_labels) if all_labels else pt.empty(0)
    scores = pt.cat(all_scores) if all_scores else pt.empty(0)
    metrics = aggregate_metrics(scores, labels, losses, case_rows, prefix)
    metrics[f"{prefix}_loss_final"] = sum(final_losses) / max(len(final_losses), 1)
    metrics[f"{prefix}_loss_base"] = sum(base_losses) / max(len(base_losses), 1)
    metrics[f"{prefix}_loss_atom_pre_aux"] = sum(atom_pre_aux_losses) / max(len(atom_pre_aux_losses), 1)
    metrics[f"{prefix}_loss_teacher_distill"] = sum(distill_losses) / max(len(distill_losses), 1)
    metrics["surface_gate_mean"] = sum(gate_means) / max(len(gate_means), 1)
    metrics["surface_gate_std"] = sum(gate_stds) / max(len(gate_stds), 1)
    metrics["surface_delta_std"] = sum(delta_stds) / max(len(delta_stds), 1)
    for key, values in patch_diag_values.items():
        metrics[key] = sum(values) / max(len(values), 1)
    metrics["skipped_examples"] = skipped_examples
    metrics["skipped_batches"] = skipped_batches
    return metrics, case_rows


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.matmul_precision:
        pt.set_float32_matmul_precision(args.matmul_precision)
    states = {x.strip() for x in args.states.split(",") if x.strip()}
    if args.fixed_train_manifest:
        train_rows = read_manifest(args.fixed_train_manifest)
        train_rows = [row for row in train_rows if row.get("input_state") in states]
        val_rows = read_manifest(args.fixed_val_manifest) if args.fixed_val_manifest else []
        val_rows = [row for row in val_rows if row.get("input_state") in states]
        rows = train_rows + val_rows
        if not train_rows:
            raise RuntimeError("No rows left in fixed train manifest after state filtering.")
    else:
        rows = read_manifest(args.manifest)
        rows = [row for row in rows if row.get("input_state") in states]
        if args.max_cases:
            rows = rows[: args.max_cases]
        if not rows:
            raise RuntimeError("No rows left after manifest/state filtering.")
        train_rows, val_rows = split_rows_grouped(rows, val_fraction=args.val_fraction, seed=args.seed, group_key=args.val_split_group)
    write_manifest_csv(output_dir / "train_manifest.csv", train_rows)
    write_manifest_csv(output_dir / "val_manifest.csv", val_rows)
    pos_weight_value, label_stats = estimate_pos_weight(train_rows, factor=args.pos_weight_factor)
    pos_weight = pt.tensor(float(pos_weight_value), dtype=pt.float)
    train_loader = build_loader(train_rows, args, shuffle=True)
    val_loader = build_loader(val_rows, args, shuffle=False) if val_rows and args.eval_every_epoch else None

    device = pt.device(args.device if args.device == "cpu" or pt.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    optimizer = build_optimizer(args, model)
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    optimizer_steps_per_epoch = max(int(math.ceil(len(train_loader) / grad_accum_steps)), 1)
    scheduler, scheduler_info = build_scheduler(args, optimizer, optimizer_steps_per_epoch)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = pt.cuda.amp.GradScaler(enabled=use_amp)

    run_report = {
        "args": vars(args),
        "model": "S1SurfaceFieldPredictor",
        "task": "frozen_E89_strict_Gaussian_surface_field_residue_prior",
        "heads": "diagnostic shared readout at pre-surface prior, surface-conditioned residue state, and trained final fusion",
        "loss": "BCE(final residue logits) + base_aux_weight * BCE(surface-conditioned logits) + atom_pre_aux_weight * BCE(AtomGNN residue logits) + optional teacher logit distillation; prior logits are diagnostic-only",
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "label_stats": label_stats,
        "pos_weight": pos_weight_value,
        "scheduler": scheduler_info,
        "optimizer_param_groups": optimizer_group_report(optimizer),
        "ema_decay": float(args.ema_decay),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows_log = read_csv(output_dir / "train_step_log.csv") if args.resume_checkpoint else []
    eval_log = read_csv(output_dir / "eval_metrics.csv") if args.resume_checkpoint else []
    global_step = 0
    start_epoch = 0
    if args.resume_checkpoint:
        checkpoint = pt.load(args.resume_checkpoint, map_location=device)
        if float(args.ema_decay) > 0.0 and checkpoint.get("raw_model_state_dict") is not None:
            model_state = checkpoint["raw_model_state_dict"]
        else:
            model_state = checkpoint["model_state_dict"]
        if args.resume_non_strict:
            current_state = model.state_dict()
            filtered_state = {}
            skipped_shape = {}
            for key, value in model_state.items():
                current_value = current_state.get(key)
                if current_value is not None and tuple(current_value.shape) != tuple(value.shape):
                    skipped_shape[key] = {
                        "checkpoint": list(value.shape),
                        "current": list(current_value.shape),
                    }
                    continue
                filtered_state[key] = value
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            (output_dir / "resume_load_report.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.resume_checkpoint),
                        "missing": list(missing),
                        "unexpected": list(unexpected),
                        "skipped_shape_mismatch": skipped_shape,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        else:
            model.load_state_dict(model_state, strict=True)
        if not args.resume_model_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if args.resume_reset_progress:
            global_step = 0
            start_epoch = 0
        else:
            global_step = int(checkpoint.get("global_step", 0))
            start_epoch = int(checkpoint.get("epoch", 0))
    if args.init_a4_checkpoint:
        checkpoint = pt.load(args.init_a4_checkpoint, map_location=device)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        remapped = dict(model_state)
        for key, value in list(model_state.items()):
            if key.startswith("primary_head."):
                remapped["base_head." + key[len("primary_head.") :]] = value
        current_state = model.state_dict()
        skipped_shape = []
        compatible_state = {}
        for key, value in remapped.items():
            if key in current_state and tuple(current_state[key].shape) != tuple(value.shape):
                skipped_shape.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(current_state[key].shape),
                    }
                )
                continue
            compatible_state[key] = value
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        (output_dir / "init_a4_load_report.json").write_text(
            json.dumps(
                {"missing": list(missing), "unexpected": list(unexpected), "skipped_shape": skipped_shape},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    ema_state = None
    if float(args.ema_decay) > 0.0:
        if not (0.0 < float(args.ema_decay) < 1.0):
            raise ValueError("--ema-decay must be in (0, 1) when enabled.")
        ema_state = init_ema_state(model)
        if args.resume_checkpoint:
            checkpoint_ema = checkpoint.get("ema_model_state_dict") if isinstance(checkpoint, dict) else None
            if checkpoint_ema is not None:
                ema_state = clone_state_for_checkpoint(checkpoint_ema)

    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        epoch_seed = int(args.seed) + int(epoch)
        random.seed(epoch_seed)
        pt.manual_seed(epoch_seed)
        if pt.cuda.is_available():
            pt.cuda.manual_seed_all(epoch_seed)
        semantic_head = getattr(model, "semantic_patch_head", None)
        if semantic_head is not None and hasattr(semantic_head, "set_boundary_refinement_scale"):
            boundary_active = (epoch + 1) >= int(args.semantic_patch_boundary_start_epoch)
            semantic_head.set_boundary_refinement_scale(1.0 if boundary_active else 0.0)
        model.train()
        accum_counter = 0
        accum_metric_sums = {}
        step_start = time.time()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train_s1_e{epoch + 1}")):
            if batch.get("skip_batch"):
                continue
            if accum_counter == 0:
                step_start = time.time()
            batch = move_batch_to_device(batch, device)
            with pt.cuda.amp.autocast(enabled=use_amp):
                out = model_forward(model, batch)
                loss_final = masked_bce(out["logits"], batch["y"], batch["mask"], pos_weight)
                loss_base = masked_bce(out["base_logits"], batch["y"], batch["mask"], pos_weight)
                loss_prior_diagnostic = masked_bce(out["prior_logits"], batch["y"], batch["mask"], pos_weight)
                soft_target = batch.get("soft_y")
                if soft_target is not None and float(args.soft_residue_loss_weight) > 0:
                    loss_soft_residue = masked_soft_bce(out["logits"], soft_target, batch["mask"], pos_weight)
                else:
                    loss_soft_residue = out["logits"].sum() * 0.0
                loss_surface_soft = surface_soft_bce_from_batch(out, batch, pos_weight, args)
                if out.get("atom_pre_logits") is not None and float(args.atom_pre_aux_loss_weight) > 0:
                    loss_atom_pre_aux = masked_bce(out["atom_pre_logits"], batch["y"], batch["mask"], pos_weight)
                else:
                    loss_atom_pre_aux = out["logits"].sum() * 0.0
                loss_motif_patch_aux = motif_patch_aux_bce_from_batch(out, batch, pos_weight, args)
                semantic_losses = semantic_patch_losses(out, batch, args)
                loss_pocket_mass = pocket_mass_loss_from_batch(out, batch, pos_weight, args)
                loss_distill = teacher_logit_distillation_loss(
                    out["logits"],
                    batch.get("teacher_logits"),
                    batch["mask"],
                    weight=args.teacher_logit_weight,
                    mode=args.teacher_logit_loss,
                    temperature=args.teacher_temperature,
                )
                loss = (
                    loss_final
                    + float(args.base_aux_loss_weight) * loss_base
                    + float(args.soft_residue_loss_weight) * loss_soft_residue
                    + float(args.surface_soft_loss_weight) * loss_surface_soft
                    + float(args.atom_pre_aux_loss_weight) * loss_atom_pre_aux
                    + float(args.motif_patch_aux_loss_weight) * loss_motif_patch_aux
                    + float(args.semantic_patch_amplitude_loss_weight) * semantic_losses["amplitude"]
                    + float(args.semantic_patch_set_loss_weight) * semantic_losses["set"]
                    + float(args.semantic_patch_existence_loss_weight) * semantic_losses["existence"]
                    + float(args.semantic_patch_connectivity_loss_weight) * semantic_losses["connectivity"]
                    + float(args.pocket_mass_loss_weight) * loss_pocket_mass
                    + loss_distill
                )
            accum_counter += 1
            for key, value in {
                "loss": loss,
                "loss_final": loss_final,
                "loss_base": loss_base,
                "loss_prior_diagnostic": loss_prior_diagnostic,
                "loss_soft_residue": loss_soft_residue,
                "loss_surface_soft": loss_surface_soft,
                "loss_atom_pre_aux": loss_atom_pre_aux,
                "loss_motif_patch_aux": loss_motif_patch_aux,
                "loss_semantic_patch_amplitude": semantic_losses["amplitude"],
                "loss_semantic_patch_set": semantic_losses["set"],
                "loss_semantic_patch_existence": semantic_losses["existence"],
                "loss_semantic_patch_connectivity": semantic_losses["connectivity"],
                "loss_teacher_distill": loss_distill,
                "loss_pocket_mass": loss_pocket_mass,
            }.items():
                accum_metric_sums[key] = accum_metric_sums.get(key, 0.0) + float(metric_scalar(value))
            scaler.scale(loss / float(grad_accum_steps)).backward()
            should_step = accum_counter >= grad_accum_steps or batch_idx == len(train_loader) - 1
            if not should_step:
                continue
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = pt.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                grad_norm = pt.tensor(0.0)
            head_grad_metrics = named_grad_norms(
                model,
                ("ctmc_rate.", "ctmc_gate.", "event_competition_head.", "base_head.", "same_level_head.", "atom_pre_encoder.", "atom_pre_aux_head."),
            )
            scaler.step(optimizer)
            scaler.update()
            if ema_state is not None:
                update_ema_state(ema_state, model, float(args.ema_decay))
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accum_metrics = {key: value / max(accum_counter, 1) for key, value in accum_metric_sums.items()}

            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.time() - start
                valid = batch["mask"] > 0.5
                row = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "micro_batches": accum_counter,
                    "effective_batch_size": args.batch_size * accum_counter,
                    "loss": accum_metrics["loss"],
                    "loss_final": accum_metrics["loss_final"],
                    "loss_base": accum_metrics["loss_base"],
                    "loss_prior_diagnostic": accum_metrics["loss_prior_diagnostic"],
                    "loss_soft_residue": accum_metrics["loss_soft_residue"],
                    "loss_surface_soft": accum_metrics["loss_surface_soft"],
                    "loss_atom_pre_aux": accum_metrics["loss_atom_pre_aux"],
                    "loss_motif_patch_aux": accum_metrics["loss_motif_patch_aux"],
                    "loss_semantic_patch_amplitude": accum_metrics["loss_semantic_patch_amplitude"],
                    "loss_semantic_patch_set": accum_metrics["loss_semantic_patch_set"],
                    "loss_semantic_patch_existence": accum_metrics["loss_semantic_patch_existence"],
                    "loss_semantic_patch_connectivity": accum_metrics["loss_semantic_patch_connectivity"],
                    "loss_teacher_distill": accum_metrics["loss_teacher_distill"],
                    "base_aux_loss_weight": args.base_aux_loss_weight,
                    "soft_residue_loss_weight": args.soft_residue_loss_weight,
                    "surface_soft_loss_weight": args.surface_soft_loss_weight,
                    "atom_pre_aux_loss_weight": args.atom_pre_aux_loss_weight,
                    "motif_patch_aux_loss_weight": args.motif_patch_aux_loss_weight,
                    "semantic_patch_amplitude_loss_weight": args.semantic_patch_amplitude_loss_weight,
                    "semantic_patch_set_loss_weight": args.semantic_patch_set_loss_weight,
                    "semantic_patch_existence_loss_weight": args.semantic_patch_existence_loss_weight,
                    "semantic_patch_connectivity_loss_weight": args.semantic_patch_connectivity_loss_weight,
                    "pocket_mass_loss_weight": args.pocket_mass_loss_weight,
                    "teacher_logit_weight": args.teacher_logit_weight,
                    "ema_decay": args.ema_decay,
                    "surface_gate_mean": metric_scalar(out["surface_gate"].detach().float().mean()),
                    "surface_gate_std": metric_scalar(out["surface_gate"].detach().float().std(unbiased=False)),
                    "surface_delta_std": metric_scalar(out["surface_delta"].detach().float().std(unbiased=False)),
                    "prior_probability_mean": metric_scalar(pt.sigmoid(out["prior_logits"].detach()).float().mean()),
                    "surface_conditioned_probability_mean": metric_scalar(pt.sigmoid(out["surface_conditioned_logits"].detach()).float().mean()),
                    "final_probability_mean": metric_scalar(pt.sigmoid(out["logits"].detach()).float().mean()),
                    "grad_norm": metric_scalar(grad_norm),
                    "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
                    "learning_rate_groups": json.dumps(
                        {group.get("name", f"group_{idx}"): float(group.get("lr", 0.0)) for idx, group in enumerate(optimizer.param_groups)},
                        ensure_ascii=False,
                    ),
                    "batch_valid_residues": int(valid.sum().item()),
                    "batch_positive_residues": int(batch["y"][valid].sum().item()) if pt.any(valid) else 0,
                    "batch_positive_rate": metric_scalar(float(batch["y"][valid].mean().item()) if pt.any(valid) else 0.0),
                    "step_seconds": f"{time.time() - step_start:.4f}",
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "steps_per_hour": f"{global_step / max(elapsed, 1e-6) * 3600.0:.2f}",
                    "skipped_count": int(batch.get("skipped_count", 0) or 0),
                }
                if out.get("view_entropy") is not None:
                    row["router_entropy"] = metric_scalar(out["view_entropy"])
                row.update(patch_diagnostics(out, batch))
                row.update(sigma_diagnostics(out))
                row.update(head_grad_metrics)
                rows_log.append(row)
                write_csv(output_dir / "train_step_log.csv", rows_log)

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(output_dir / f"model_step_{global_step}.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args, ema_state)
            if args.max_steps and global_step >= args.max_steps:
                save_checkpoint(output_dir / "model_last.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args, ema_state)
                return
            accum_counter = 0
            accum_metric_sums = {}

        if args.save_every_epoch:
            save_checkpoint(output_dir / f"model_epoch_{epoch + 1}.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, args, ema_state)
        if val_loader is not None:
            metrics, case_rows = run_eval(model, val_loader, device, pos_weight, f"epoch{epoch + 1}_val", args)
            metrics["epoch"] = epoch + 1
            metrics["global_step"] = global_step
            eval_log.append(metrics)
            write_csv(output_dir / "eval_metrics.csv", eval_log)
            write_csv(output_dir / f"epoch{epoch + 1}_val_case_metrics.csv", case_rows)
            model.train()

    save_checkpoint(output_dir / "model_last.pt", model, optimizer, scheduler, scaler, args.epochs, global_step, args, ema_state)


def main():
    parser = argparse.ArgumentParser(description="Train S1 A4-conditioned surface interface field.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fixed-train-manifest", default="")
    parser.add_argument("--fixed-val-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-cache-dir", required=True)
    parser.add_argument("--residue-graph-cache-dir", required=True)
    parser.add_argument("--s1-surface-cache-dir", default="")
    parser.add_argument("--s1-static-geometry-cache-dir", default="")
    parser.add_argument("--s1-static-geometry-cache-version", default="e89_dense_geometry_v1")
    parser.add_argument("--require-s1-static-geometry-cache", action="store_true")
    parser.add_argument("--true-feature-cache-dir", default="")
    parser.add_argument("--teacher-cache-dir", default="")
    parser.add_argument("--teacher-name", default="")
    parser.add_argument("--teacher-cache-version", default="teacher_prior_v1")
    parser.add_argument("--states", default="apo,af2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine", "linear"], default="cosine")
    parser.add_argument("--scheduler-total-epochs", type=int, default=0, help="Use this epoch count for LR schedule length while --epochs still controls the stop epoch.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--pos-weight-factor", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-split-group", choices=["system_id", "cluster_id", "cluster_id_RL", "receptor_uniprot", "partner_uniprot"], default="cluster_id")
    parser.add_argument("--atom-feature-dim", type=int, default=123)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--gvp-vector-dim", type=int, default=16)
    parser.add_argument("--graph-views", default="seq,k16,k32,exposure")
    parser.add_argument("--residue-graph-cache-ks", type=lambda text: tuple(int(x) for x in str(text).split(",") if x), default=(16, 24, 32))
    parser.add_argument("--view-layers", type=int, default=2)
    parser.add_argument("--rbf-bins", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--router-mode", choices=["global", "per_residue", "full_attention"], default="global")
    parser.add_argument(
        "--atom-residue-encoder-mode",
        choices=["mean", "attention", "intra_graph", "local_contact"],
        default="mean",
        help="Atom-to-residue initialization: legacy mean pooling, atom attention, intra-residue atom graph, or local atom contact graph.",
    )
    parser.add_argument("--atom-graph-k", type=int, default=8)
    parser.add_argument("--atom-contact-k", type=int, default=16)
    parser.add_argument("--atom-contact-radius", type=float, default=5.0)
    parser.add_argument("--atom-pre-encoder-mode", choices=["none", "mpnn"], default="none")
    parser.add_argument("--atom-pre-dim", type=int, default=64)
    parser.add_argument("--atom-pre-layers", type=int, default=2)
    parser.add_argument("--atom-pre-topk", type=int, default=16)
    parser.add_argument("--atom-pre-cutoff", type=float, default=5.0)
    parser.add_argument("--atom-pre-dropout", type=float, default=0.05)
    parser.add_argument("--use-augmented-features", action="store_true", default=True)
    parser.add_argument("--no-augmented-features", dest="use_augmented_features", action="store_false")
    parser.add_argument("--residue-feature-dim", type=int, default=43)
    parser.add_argument("--use-true-features", action="store_true")
    parser.add_argument("--true-feature-dim", type=int, default=10)
    parser.add_argument("--true-feature-version", default="true_sasa_dssp_v1")
    parser.add_argument("--require-true-features", action="store_true")
    parser.add_argument("--require-tensor-cache", action="store_true", default=True)
    parser.add_argument("--require-residue-graph-cache", action="store_true", default=True)
    parser.add_argument("--require-s1-surface-cache", action="store_true")
    parser.add_argument("--disable-s1-surface-on-the-fly", action="store_true")
    parser.add_argument("--s1-surface-cache-version", default="s1_surface_proxy_v1")
    parser.add_argument("--s1-surface-points", type=int, default=1024)
    parser.add_argument("--s1-surface-graph-k", type=int, default=16)
    parser.add_argument("--s1-surface-lappe-dim", type=int, default=0)
    parser.add_argument("--s1-surface-layers", type=int, default=2)
    parser.add_argument("--s1-surface-residue-topk", type=int, default=8)
    parser.add_argument("--s1-residue-surface-topk", type=int, default=24)
    parser.add_argument("--s1-sigma-tangent", type=float, default=6.0)
    parser.add_argument("--s1-sigma-normal", type=float, default=3.0)
    parser.add_argument(
        "--s1-projection-mode",
        choices=[
            "single",
            "multiscale",
            "multiscale_global",
            "multiscale_learned",
            "anisotropic_sigma",
            "per_residue_sigma",
            "full_learnable_covariance",
            "phys_gaussian_field",
            "mixture_ball_k4",
            "local3d_frame",
            "local3d_atom_pointcloud",
            "local3d_patch_pointcloud",
        ],
        default="single",
    )
    parser.add_argument("--s1-multiscale-sigmas", default="4:2,8:4,14:7")
    parser.add_argument("--s1-surface-attention-prior-weight", type=float, default=0.0)
    parser.add_argument("--s1-surface-attention-sigma-tangent", type=float, default=8.0)
    parser.add_argument("--s1-surface-attention-sigma-normal", type=float, default=4.0)
    parser.add_argument("--surface-to-residue-mode", choices=["attention", "per_surface_gaussian", "geoparam_attention"], default="attention")
    parser.add_argument("--surface-extra-graph-k", type=int, default=0)
    parser.add_argument("--surface-extra-graph-ks", default="")
    parser.add_argument("--surface-graph-flow-mode", choices=["parallel", "k16_then_k32"], default="parallel")
    parser.add_argument("--phys-feature-dim", type=int, default=0)
    parser.add_argument("--phys-attention-bias-weight", type=float, default=0.0)
    parser.add_argument("--phys-context-weight", type=float, default=0.0)
    parser.add_argument("--phys-message-weight", type=float, default=0.0)
    parser.add_argument(
        "--surface-patch-mode",
        choices=[
            "none", "gate", "attn", "self_attn", "replace_attn", "independent_attn",
            "connected_pool", "connected_pool_adj",
        ],
        default="none",
    )
    parser.add_argument("--surface-patch-count", type=int, default=32)
    parser.add_argument("--surface-patch-beta", type=float, default=0.1)
    parser.add_argument("--surface-patch-attention-heads", type=int, default=4)
    parser.add_argument("--surface-lappe-mode", choices=["none", "encoder", "patch", "both"], default="none")
    parser.add_argument("--surface-spectral-filter-dim", type=int, default=0)
    parser.add_argument("--internal-operator-mode", choices=["none", "residue_laplacian"], default="none")
    parser.add_argument("--internal-operator-k", type=int, default=16)
    parser.add_argument("--internal-operator-layers", type=int, default=2)
    parser.add_argument("--internal-operator-beta", type=float, default=0.1)
    parser.add_argument("--surface-encoder-mode", choices=["graph", "attention", "gvp"], default="graph")
    parser.add_argument("--surface-vector-bridge-mode", choices=["none", "invariant", "equivariant"], default="none")
    parser.add_argument("--surface-co-update-mode", choices=["none", "s2r", "bidir"], default="none")
    parser.add_argument("--surface-co-update-layers", type=int, default=1)
    parser.add_argument("--surface-vector-state-mode", choices=["reinit", "persistent"], default="reinit")
    parser.add_argument("--patch-state-mode", choices=["stateless", "persistent"], default="stateless")
    parser.add_argument("--surface-serial-stage-mode", choices=["none", "pair_then_triplet", "perresidue_then_pair"], default="none")
    parser.add_argument("--surface-round-end-r2s-mode", choices=["attention", "skip"], default="attention")
    parser.add_argument("--post-final-residue-global-feedback-rounds", type=int, default=0)
    parser.add_argument("--final-residue-skip-mode", choices=["none", "gated"], default="none")
    parser.add_argument("--final-residue-skip-source", choices=["multiview"], default="multiview")
    parser.add_argument("--final-residue-skip-beta", type=float, default=0.0)
    parser.add_argument("--final-residue-skip-init-zero", action="store_true")
    parser.add_argument("--pair-gaussian-mode", choices=["none", "k_nearest", "random"], default="none")
    parser.add_argument("--pair-gaussian-topk", type=int, default=8)
    parser.add_argument("--pair-gaussian-beta", type=float, default=0.0)
    parser.add_argument("--pair-gaussian-sigma-tangent", type=float, default=8.0)
    parser.add_argument("--pair-gaussian-sigma-normal", type=float, default=4.0)
    parser.add_argument("--pair-gaussian-sigma-mode", choices=["fixed", "learned", "reuse_single"], default="fixed")
    parser.add_argument("--pair-gaussian-sigma-min", type=float, default=0.5)
    parser.add_argument("--pair-gaussian-beta-mode", choices=["fixed", "learned_deep"], default="fixed")
    parser.add_argument("--pair-gaussian-beta-min", type=float, default=0.0)
    parser.add_argument("--pair-gaussian-beta-max", type=float, default=0.3)
    parser.add_argument("--pair-gaussian-beta-init", type=float, default=0.05)
    parser.add_argument("--pair-gaussian-content-mode", choices=["none", "additive", "gaussian_distance"], default="none")
    parser.add_argument("--pair-gaussian-content-weight", type=float, default=1.0)
    parser.add_argument("--pair-gaussian-rho-condition-mode", choices=["none", "surface_dot"], default="none")
    parser.add_argument("--pair-gaussian-vector-condition-mode", choices=["none", "surface_frame"], default="none")
    parser.add_argument("--pair-gaussian-single-compete-mode", choices=["none", "conservative_rescue", "strong_gate"], default="none")
    parser.add_argument("--pair-gaussian-single-rescue-beta", type=float, default=0.25)
    parser.add_argument("--pair-gaussian-chunk-size", type=int, default=0)
    parser.add_argument("--no-pair-gaussian-chunk-checkpoint", action="store_true")
    parser.add_argument("--pair-gaussian-chunk-diagnostics", choices=["full", "none"], default="full")
    parser.add_argument("--pair-gaussian-only", action="store_true")
    parser.add_argument("--pair-gaussian-fusion-mode", choices=["init_only", "deep", "coattn"], default="init_only")
    parser.add_argument("--pair-gaussian-joint-mode", choices=["independent", "joint_residual", "joint_only"], default="independent")
    parser.add_argument("--pair-gaussian-multiview-mode", choices=["none", "init", "deep"], default="none")
    parser.add_argument("--pair-gaussian-multiview-beta", type=float, default=0.1)
    parser.add_argument("--pair-gaussian-multiview-chunk-size", type=int, default=256)
    parser.add_argument("--triplet-gaussian-mode", choices=["none", "k_nearest", "random"], default="none")
    parser.add_argument("--triplet-gaussian-topk", type=int, default=8)
    parser.add_argument("--triplet-gaussian-beta", type=float, default=0.0)
    parser.add_argument("--triplet-gaussian-beta-mode", choices=["fixed", "learned_deep"], default="fixed")
    parser.add_argument("--triplet-gaussian-beta-min", type=float, default=0.0)
    parser.add_argument("--triplet-gaussian-beta-max", type=float, default=0.2)
    parser.add_argument("--triplet-gaussian-beta-init", type=float, default=0.03)
    parser.add_argument("--triplet-gaussian-sigma-tangent", type=float, default=8.0)
    parser.add_argument("--triplet-gaussian-sigma-normal", type=float, default=4.0)
    parser.add_argument("--triplet-gaussian-fusion-mode", choices=["init_only", "deep"], default="init_only")
    parser.add_argument("--triplet-gaussian-score-temperature", type=float, default=1.0)
    parser.add_argument("--triplet-gaussian-score-clamp", type=float, default=0.0)
    parser.add_argument("--triplet-gaussian-preselect-topm", type=int, default=0)
    parser.add_argument("--residue-amplitude-mode", choices=["none", "bounded"], default="none")
    parser.add_argument("--residue-amplitude-min", type=float, default=0.25)
    parser.add_argument("--residue-amplitude-max", type=float, default=2.0)
    parser.add_argument("--surface-fusion-mode", choices=["delta", "same_level"], default="delta")
    parser.add_argument("--global-feedback-rounds", type=int, default=1)
    parser.add_argument("--global-feedback-interleave-residue-graph", action="store_true")
    parser.add_argument("--surface-global-feedback-before-residue", action="store_true")
    parser.add_argument("--surface-global-feedback-rounds", type=int, default=1)
    parser.add_argument("--core-cycles", type=int, default=1)
    parser.add_argument("--s1-surface-delta-scale", type=float, default=0.2)
    parser.add_argument("--residue-vector-surface-mode", choices=["none", "cross_attn"], default="none")
    parser.add_argument("--residue-vector-surface-topk", type=int, default=16)
    parser.add_argument("--gear-mode", choices=["none", "rel_edge", "edge_msg"], default="none")
    parser.add_argument("--gear-relation-k", type=int, default=32)
    parser.add_argument("--gear-layers", type=int, default=2)
    parser.add_argument("--gear-view-mode", choices=["none", "rel_edge_msg", "rel_edge_msg_ieconv"], default="none")
    parser.add_argument("--gear-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--oagnn-mode", choices=["none", "oa_edge", "official"], default="none")
    parser.add_argument("--oagnn-relation-k", type=int, default=32)
    parser.add_argument("--oagnn-layers", type=int, default=2)
    parser.add_argument("--oagnn-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--token-encoder-mode", choices=["none", "protein_token"], default="none")
    parser.add_argument("--token-encoder-layers", type=int, default=2)
    parser.add_argument("--token-encoder-heads", type=int, default=4)
    parser.add_argument("--edge-pair-mode", choices=["none", "lite"], default="none")
    parser.add_argument("--cross-view-edge-mode", choices=["none", "source_residue_attn"], default="none")
    parser.add_argument("--cross-view-edge-beta", type=float, default=0.1)
    parser.add_argument("--cross-view-edge-heads", type=int, default=4)
    parser.add_argument("--residue-encoder-mode", choices=["multiview", "multiplex"], default="multiview")
    parser.add_argument(
        "--residue-graph-flow-mode",
        choices=["parallel", "k16_then_k32", "k8_then_k16_then_k32_then_k64"],
        default="parallel",
    )
    parser.add_argument("--sequence-pos-enc-mode", choices=["none", "rel_bucket"], default="none")
    parser.add_argument("--sequence-pos-enc-dim", type=int, default=16)
    parser.add_argument("--post-final-fourview-mode", choices=["none", "refine_dual_env"], default="none")
    parser.add_argument("--post-final-fourview-beta", type=float, default=1.0)
    parser.add_argument("--surface-bus-mode", choices=["none", "static", "static_pre_global", "recurrent"], default="none")
    parser.add_argument("--surface-bus-beta", type=float, default=1.0)
    parser.add_argument("--surface-bus-topk", type=int, default=16)
    parser.add_argument("--struct-community-mode", choices=["none", "dynamic_attn"], default="none")
    parser.add_argument("--struct-community-count", type=int, default=16)
    parser.add_argument("--struct-community-geom-bias", type=float, default=0.25)
    parser.add_argument("--stochastic-head-mode", choices=["none", "poisson", "poisson_pocket", "ctmc_poisson", "event_competition"], default="none")
    parser.add_argument("--ctmc-diffusion-steps", type=int, default=3)
    parser.add_argument("--ctmc-diffusion-gamma", type=float, default=0.25)
    parser.add_argument("--ctmc-diffusion-topk", type=int, default=16)
    parser.add_argument("--poisson-lambda-max", type=float, default=20.0)
    parser.add_argument("--pocket-mass-topk", type=int, default=16)
    parser.add_argument("--pocket-mass-loss-weight", type=float, default=0.0)
    parser.add_argument("--motif-patch-mode", choices=["none", "feedback", "dual_cross"], default="none")
    parser.add_argument("--motif-patch-count", type=int, default=32)
    parser.add_argument("--motif-patch-beta", type=float, default=0.1)
    parser.add_argument("--motif-patch-attention-heads", type=int, default=4)
    parser.add_argument("--motif-patch-geom-bias", type=float, default=0.1)
    parser.add_argument("--motif-patch-aux-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic-patch-mode",
        choices=[
            "none", "amplitude", "gaussian", "setpatch", "full",
            "sparse_probe", "sparse_noisyor", "graph_motif", "graph_motif_refine",
        ],
        default="none",
    )
    parser.add_argument("--semantic-patch-query-count", type=int, default=8)
    parser.add_argument("--semantic-patch-beta", type=float, default=0.03)
    parser.add_argument("--semantic-patch-radius-min", type=float, default=2.0)
    parser.add_argument("--semantic-patch-radius-max", type=float, default=16.0)
    parser.add_argument("--semantic-patch-adjacency-weight", type=float, default=0.25)
    parser.add_argument("--semantic-patch-active-topk", type=int, default=4)
    parser.add_argument("--semantic-patch-graph-hops", type=int, default=2)
    parser.add_argument("--semantic-patch-geodesic-max-steps", type=int, default=96)
    parser.add_argument("--semantic-patch-geodesic-normal-penalty", type=float, default=1.0)
    parser.add_argument("--semantic-patch-geodesic-tangent-penalty", type=float, default=1.0)
    parser.add_argument(
        "--semantic-patch-detach-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--semantic-patch-boundary-mix", type=float, default=0.1)
    parser.add_argument("--semantic-patch-boundary-temperature", type=float, default=0.7)
    parser.add_argument("--semantic-patch-boundary-start-epoch", type=int, default=3)
    parser.add_argument("--semantic-patch-region-existence-threshold", type=float, default=0.05)
    parser.add_argument("--semantic-patch-component-k", type=int, default=16)
    parser.add_argument("--semantic-patch-amplitude-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-patch-set-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-patch-existence-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-patch-connectivity-loss-weight", type=float, default=0.0)
    parser.add_argument("--atom-pre-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--base-aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--use-soft-residue-labels", action="store_true")
    parser.add_argument("--soft-label-sigma", type=float, default=8.0)
    parser.add_argument("--soft-label-max-value", type=float, default=0.7)
    parser.add_argument("--soft-label-cutoff", type=float, default=24.0)
    parser.add_argument("--soft-residue-loss-weight", type=float, default=0.0)
    parser.add_argument("--surface-soft-loss-weight", type=float, default=0.0)
    parser.add_argument("--surface-soft-sigma-tangent", type=float, default=6.0)
    parser.add_argument("--surface-soft-sigma-normal", type=float, default=3.0)
    parser.add_argument("--surface-soft-max-value", type=float, default=1.0)
    parser.add_argument("--surface-soft-cutoff", type=float, default=24.0)
    parser.add_argument('--surface-soft-label-mode', choices=('global_max', 'local_topk_weighted', 'positive_topk_noisy_or'), default='global_max')
    parser.add_argument('--surface-soft-label-topk', type=int, default=16)
    parser.add_argument("--teacher-logit-weight", type=float, default=0.0)
    parser.add_argument("--teacher-logit-loss", choices=["mse", "bce"], default="mse")
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-hidden-dim", type=int, default=0)
    parser.add_argument("--require-teacher-cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--resume-reset-progress", action="store_true")
    parser.add_argument("--resume-non-strict", action="store_true")
    parser.add_argument("--init-a4-checkpoint", default="")
    parser.add_argument("--allow-residue-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pt.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
