import argparse
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch as pt

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
from apex_generic_gvp_multigraph.metrics import aggregate_metrics, case_metrics, write_csv
from train_apex import estimate_pos_weight
from train_apex_simple import masked_bce, move_batch_to_device


def load_rows(path, states, max_cases=0):
    rows = read_manifest(path)
    if states:
        allowed = {x.strip() for x in states.split(",") if x.strip()}
        rows = [row for row in rows if row.get("input_state") in allowed]
    if max_cases:
        rows = rows[: int(max_cases)]
    return rows


def build_loader(rows, args):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_s1_surface_field_batch,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return pt.utils.data.DataLoader(
        S1SurfaceFieldDataset(
            rows,
            strict_residue_alignment=not args.allow_residue_mismatch,
            skip_bad_examples=True,
            tensor_cache_dir=args.tensor_cache_dir,
            require_tensor_cache=True,
            esm_cache_dir=None,
            require_esm_cache=False,
            residue_graph_cache_dir=args.residue_graph_cache_dir,
            require_residue_graph_cache=True,
            residue_graph_ks=args.residue_graph_cache_ks,
            surface_cache_dir="",
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
            s1_static_geometry_cache_dir=args.s1_static_geometry_cache_dir,
            s1_static_geometry_cache_version=args.s1_static_geometry_cache_version,
            require_s1_static_geometry_cache=args.require_s1_static_geometry_cache,
        ),
        **kwargs,
    )


def build_model_from_ckpt_args(ckpt_args):
    def get_arg(name, default):
        value = ckpt_args.get(name, default)
        return default if value is None else value

    surface_encoder_mode = str(get_arg("surface_encoder_mode", "graph"))
    if surface_encoder_mode == "message":
        surface_encoder_mode = "graph"
    surface_fusion_mode = str(get_arg("surface_fusion_mode", "delta"))
    if surface_fusion_mode == "gated_delta":
        surface_fusion_mode = "delta"

    model_kwargs = dict(
        atom_feature_dim=int(get_arg("atom_feature_dim", 123)),
        dim=int(get_arg("dim", 128)),
        vector_dim=int(get_arg("gvp_vector_dim", 16)),
        graph_views=tuple(x.strip() for x in str(get_arg("graph_views", "seq,k16,k32,exposure")).split(",") if x.strip()),
        view_layers=int(get_arg("view_layers", 2)),
        rbf_bins=int(get_arg("rbf_bins", 16)),
        dropout=float(get_arg("dropout", 0.1)),
        router_mode=str(get_arg("router_mode", "global")),
        use_augmented_features=bool(get_arg("use_augmented_features", True)),
        residue_feature_dim=int(get_arg("residue_feature_dim", 43)),
        use_true_features=bool(get_arg("use_true_features", False)),
        true_feature_dim=int(get_arg("true_feature_dim", 10)),
        surface_points=int(get_arg("s1_surface_points", 1024)),
        surface_graph_k=int(get_arg("s1_surface_graph_k", 16)),
        surface_extra_graph_k=int(get_arg("surface_extra_graph_k", 0)),
        surface_extra_graph_ks=str(get_arg("surface_extra_graph_ks", "")),
        surface_graph_flow_mode=str(get_arg("surface_graph_flow_mode", "parallel")),
        surface_layers=int(get_arg("s1_surface_layers", 2)),
        surface_residue_topk=int(get_arg("s1_surface_residue_topk", 8)),
        residue_surface_topk=int(get_arg("s1_residue_surface_topk", 24)),
        sigma_tangent=float(get_arg("s1_sigma_tangent", 6.0)),
        sigma_normal=float(get_arg("s1_sigma_normal", 3.0)),
        projection_mode=str(get_arg("s1_projection_mode", "single")),
        multiscale_sigmas=str(get_arg("s1_multiscale_sigmas", "4:2,8:4,14:7")),
        surface_attention_prior_weight=float(get_arg("s1_surface_attention_prior_weight", 0.0)),
        surface_attention_sigma_tangent=float(get_arg("s1_surface_attention_sigma_tangent", 8.0)),
        surface_attention_sigma_normal=float(get_arg("s1_surface_attention_sigma_normal", 4.0)),
        surface_to_residue_mode=str(get_arg("surface_to_residue_mode", "attention")),
        phys_feature_dim=int(get_arg("phys_feature_dim", 0)),
        phys_attention_bias_weight=float(get_arg("phys_attention_bias_weight", 0.0)),
        phys_context_weight=float(get_arg("phys_context_weight", 0.0)),
        phys_message_weight=float(get_arg("phys_message_weight", 0.0)),
        surface_patch_mode=str(get_arg("surface_patch_mode", "none")),
        surface_patch_count=int(get_arg("surface_patch_count", 32)),
        surface_patch_beta=float(get_arg("surface_patch_beta", 0.1)),
        surface_patch_attention_heads=int(get_arg("surface_patch_attention_heads", 4)),
        semantic_patch_mode=str(get_arg("semantic_patch_mode", "none")),
        semantic_patch_query_count=int(get_arg("semantic_patch_query_count", 8)),
        semantic_patch_beta=float(get_arg("semantic_patch_beta", 0.03)),
        semantic_patch_radius_min=float(get_arg("semantic_patch_radius_min", 2.0)),
        semantic_patch_radius_max=float(get_arg("semantic_patch_radius_max", 16.0)),
        semantic_patch_adjacency_weight=float(get_arg("semantic_patch_adjacency_weight", 0.25)),
        semantic_patch_active_topk=int(get_arg("semantic_patch_active_topk", 4)),
        semantic_patch_graph_hops=int(get_arg("semantic_patch_graph_hops", 2)),
        semantic_patch_geodesic_max_steps=int(get_arg("semantic_patch_geodesic_max_steps", 96)),
        semantic_patch_geodesic_normal_penalty=float(
            get_arg("semantic_patch_geodesic_normal_penalty", 1.0)
        ),
        semantic_patch_geodesic_tangent_penalty=float(
            get_arg("semantic_patch_geodesic_tangent_penalty", 1.0)
        ),
        semantic_patch_detach_backbone=bool(get_arg("semantic_patch_detach_backbone", True)),
        semantic_patch_boundary_mix=float(get_arg("semantic_patch_boundary_mix", 0.1)),
        semantic_patch_boundary_temperature=float(
            get_arg("semantic_patch_boundary_temperature", 0.7)
        ),
        surface_lappe_dim=int(get_arg("s1_surface_lappe_dim", 0)),
        surface_lappe_mode=str(get_arg("surface_lappe_mode", "none")),
        surface_spectral_filter_dim=int(get_arg("surface_spectral_filter_dim", 0)),
        internal_operator_mode=str(get_arg("internal_operator_mode", "none")),
        internal_operator_k=int(get_arg("internal_operator_k", 16)),
        internal_operator_layers=int(get_arg("internal_operator_layers", 2)),
        internal_operator_beta=float(get_arg("internal_operator_beta", 0.1)),
        surface_encoder_mode=surface_encoder_mode,
        surface_vector_bridge_mode=str(get_arg("surface_vector_bridge_mode", "none")),
        surface_co_update_mode=str(get_arg("surface_co_update_mode", "none")),
        surface_co_update_layers=int(get_arg("surface_co_update_layers", 1)),
        surface_vector_state_mode=str(get_arg("surface_vector_state_mode", "reinit")),
        patch_state_mode=str(get_arg("patch_state_mode", "stateless")),
        surface_serial_stage_mode=str(get_arg("surface_serial_stage_mode", "none")),
        surface_round_end_r2s_mode=str(get_arg("surface_round_end_r2s_mode", "attention")),
        post_final_residue_global_feedback_rounds=int(get_arg("post_final_residue_global_feedback_rounds", 0)),
        final_residue_skip_mode=str(get_arg("final_residue_skip_mode", "none")),
        final_residue_skip_source=str(get_arg("final_residue_skip_source", "multiview")),
        final_residue_skip_beta=float(get_arg("final_residue_skip_beta", 0.0)),
        final_residue_skip_init_zero=bool(get_arg("final_residue_skip_init_zero", False)),
        pair_gaussian_mode=str(get_arg("pair_gaussian_mode", "none")),
        pair_gaussian_topk=int(get_arg("pair_gaussian_topk", 8)),
        pair_gaussian_beta=float(get_arg("pair_gaussian_beta", 0.0)),
        pair_gaussian_sigma_tangent=float(get_arg("pair_gaussian_sigma_tangent", 8.0)),
        pair_gaussian_sigma_normal=float(get_arg("pair_gaussian_sigma_normal", 4.0)),
        pair_gaussian_sigma_mode=str(get_arg("pair_gaussian_sigma_mode", "fixed")),
        pair_gaussian_sigma_min=float(get_arg("pair_gaussian_sigma_min", 0.5)),
        pair_gaussian_beta_mode=str(get_arg("pair_gaussian_beta_mode", "fixed")),
        pair_gaussian_beta_min=float(get_arg("pair_gaussian_beta_min", 0.0)),
        pair_gaussian_beta_max=float(get_arg("pair_gaussian_beta_max", 0.3)),
        pair_gaussian_beta_init=float(get_arg("pair_gaussian_beta_init", 0.05)),
        pair_gaussian_content_mode=str(get_arg("pair_gaussian_content_mode", "none")),
        pair_gaussian_content_weight=float(get_arg("pair_gaussian_content_weight", 1.0)),
        pair_gaussian_rho_condition_mode=str(get_arg("pair_gaussian_rho_condition_mode", "none")),
        pair_gaussian_vector_condition_mode=str(get_arg("pair_gaussian_vector_condition_mode", "none")),
        pair_gaussian_single_compete_mode=str(get_arg("pair_gaussian_single_compete_mode", "none")),
        pair_gaussian_single_rescue_beta=float(get_arg("pair_gaussian_single_rescue_beta", 0.25)),
        pair_gaussian_chunk_size=int(get_arg("pair_gaussian_chunk_size", 0)),
        pair_gaussian_only=bool(get_arg("pair_gaussian_only", False)),
        pair_gaussian_fusion_mode=str(get_arg("pair_gaussian_fusion_mode", "init_only")),
        pair_gaussian_joint_mode=str(get_arg("pair_gaussian_joint_mode", "independent")),
        pair_gaussian_multiview_mode=str(get_arg("pair_gaussian_multiview_mode", "none")),
        pair_gaussian_multiview_beta=float(get_arg("pair_gaussian_multiview_beta", 0.1)),
        pair_gaussian_multiview_chunk_size=int(get_arg("pair_gaussian_multiview_chunk_size", 256)),
        triplet_gaussian_mode=str(get_arg("triplet_gaussian_mode", "none")),
        triplet_gaussian_topk=int(get_arg("triplet_gaussian_topk", 8)),
        triplet_gaussian_beta=float(get_arg("triplet_gaussian_beta", 0.0)),
        triplet_gaussian_beta_mode=str(get_arg("triplet_gaussian_beta_mode", "fixed")),
        triplet_gaussian_beta_min=float(get_arg("triplet_gaussian_beta_min", 0.0)),
        triplet_gaussian_beta_max=float(get_arg("triplet_gaussian_beta_max", 0.2)),
        triplet_gaussian_beta_init=float(get_arg("triplet_gaussian_beta_init", 0.03)),
        triplet_gaussian_sigma_tangent=float(get_arg("triplet_gaussian_sigma_tangent", 8.0)),
        triplet_gaussian_sigma_normal=float(get_arg("triplet_gaussian_sigma_normal", 4.0)),
        triplet_gaussian_fusion_mode=str(get_arg("triplet_gaussian_fusion_mode", "init_only")),
        triplet_gaussian_score_temperature=float(get_arg("triplet_gaussian_score_temperature", 1.0)),
        triplet_gaussian_score_clamp=float(get_arg("triplet_gaussian_score_clamp", 0.0)),
        triplet_gaussian_preselect_topm=int(get_arg("triplet_gaussian_preselect_topm", 0)),
        residue_amplitude_mode=str(get_arg("residue_amplitude_mode", "none")),
        residue_amplitude_min=float(get_arg("residue_amplitude_min", 0.25)),
        residue_amplitude_max=float(get_arg("residue_amplitude_max", 2.0)),
        surface_fusion_mode=surface_fusion_mode,
        global_feedback_rounds=int(get_arg("global_feedback_rounds", 1)),
        global_feedback_interleave_residue_graph=bool(get_arg("global_feedback_interleave_residue_graph", False)),
        surface_global_feedback_before_residue=bool(get_arg("surface_global_feedback_before_residue", False)),
        surface_global_feedback_rounds=int(get_arg("surface_global_feedback_rounds", 1)),
        core_cycles=int(get_arg("core_cycles", 1)),
        surface_delta_scale=float(get_arg("s1_surface_delta_scale", 0.2)),
        residue_vector_surface_mode=str(get_arg("residue_vector_surface_mode", "none")),
        residue_vector_surface_topk=int(get_arg("residue_vector_surface_topk", 16)),
        teacher_hidden_dim=int(get_arg("teacher_hidden_dim", 0)),
        gear_mode=str(get_arg("gear_mode", "none")),
        gear_relation_k=int(get_arg("gear_relation_k", 32)),
        gear_layers=int(get_arg("gear_layers", 2)),
        gear_view_mode=str(get_arg("gear_view_mode", "none")),
        oagnn_mode=str(get_arg("oagnn_mode", "none")),
        oagnn_relation_k=int(get_arg("oagnn_relation_k", 32)),
        oagnn_layers=int(get_arg("oagnn_layers", 2)),
        token_encoder_mode=str(get_arg("token_encoder_mode", "none")),
        token_encoder_layers=int(get_arg("token_encoder_layers", 2)),
        token_encoder_heads=int(get_arg("token_encoder_heads", 4)),
        edge_pair_mode=str(get_arg("edge_pair_mode", "none")),
        cross_view_edge_mode=str(get_arg("cross_view_edge_mode", "none")),
        cross_view_edge_beta=float(get_arg("cross_view_edge_beta", 0.1)),
        cross_view_edge_heads=int(get_arg("cross_view_edge_heads", 4)),
        residue_encoder_mode=str(get_arg("residue_encoder_mode", "multiview")),
        residue_graph_flow_mode=str(get_arg("residue_graph_flow_mode", "parallel")),
        sequence_pos_enc_mode=str(get_arg("sequence_pos_enc_mode", "none")),
        sequence_pos_enc_dim=int(get_arg("sequence_pos_enc_dim", 16)),
        atom_residue_encoder_mode=str(get_arg("atom_residue_encoder_mode", "mean")),
        atom_graph_k=int(get_arg("atom_graph_k", 8)),
        atom_contact_k=int(get_arg("atom_contact_k", 16)),
        atom_contact_radius=float(get_arg("atom_contact_radius", 5.0)),
        atom_pre_encoder_mode=str(get_arg("atom_pre_encoder_mode", "none")),
        atom_pre_dim=int(get_arg("atom_pre_dim", 64)),
        atom_pre_layers=int(get_arg("atom_pre_layers", 2)),
        atom_pre_topk=int(get_arg("atom_pre_topk", 16)),
        atom_pre_cutoff=float(get_arg("atom_pre_cutoff", 5.0)),
        atom_pre_dropout=float(get_arg("atom_pre_dropout", 0.05)),
        post_final_fourview_mode=str(get_arg("post_final_fourview_mode", "none")),
        post_final_fourview_beta=float(get_arg("post_final_fourview_beta", 1.0)),
        surface_bus_mode=str(get_arg("surface_bus_mode", "none")),
        surface_bus_beta=float(get_arg("surface_bus_beta", 1.0)),
        surface_bus_topk=int(get_arg("surface_bus_topk", 16)),
        struct_community_mode=str(get_arg("struct_community_mode", "none")),
        struct_community_count=int(get_arg("struct_community_count", 16)),
        struct_community_geom_bias=float(get_arg("struct_community_geom_bias", 0.25)),
        stochastic_head_mode=str(get_arg("stochastic_head_mode", "none")),
        ctmc_diffusion_steps=int(get_arg("ctmc_diffusion_steps", 3)),
        ctmc_diffusion_gamma=float(get_arg("ctmc_diffusion_gamma", 0.25)),
        ctmc_diffusion_topk=int(get_arg("ctmc_diffusion_topk", 16)),
        poisson_lambda_max=float(get_arg("poisson_lambda_max", 20.0)),
        pocket_mass_topk=int(get_arg("pocket_mass_topk", 16)),
        motif_patch_mode=str(get_arg("motif_patch_mode", "none")),
        motif_patch_count=int(get_arg("motif_patch_count", 32)),
        motif_patch_beta=float(get_arg("motif_patch_beta", 0.1)),
        motif_patch_attention_heads=int(get_arg("motif_patch_attention_heads", 4)),
        motif_patch_geom_bias=float(get_arg("motif_patch_geom_bias", 0.1)),
    )
    init_params = inspect.signature(S1SurfaceFieldPredictor.__init__).parameters
    return S1SurfaceFieldPredictor(**{k: v for k, v in model_kwargs.items() if k in init_params})


def model_forward(model, batch):
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


def patch_diagnostics(out, batch):
    if "patch_assignment_entropy" not in out:
        return {}
    metrics = {
        "patch_assignment_entropy": float(out["patch_assignment_entropy"].detach().float().mean().cpu()),
        "patch_usage_active_mean": float(out["patch_usage_active"].detach().float().mean().cpu()),
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
            metrics[key] = float(out[key].detach().float().mean().cpu())
    if "patch_surface_gate" not in out or "patch_gate" not in out or "patch_primary_residue_idx" not in out:
        return metrics
    gate = out["patch_surface_gate"].detach().float()
    idx = out["patch_primary_residue_idx"].detach().long().clamp(0, max(int(batch["y"].numel()) - 1, 0))
    labels = batch["y"].detach().float()[idx]
    valid = batch["mask"].detach().float()[idx] > 0.5
    metrics.update({
        "patch_surface_gate_mean": float(gate.mean().cpu()),
        "patch_surface_gate_std": float(gate.std(unbiased=False).cpu()),
        "patch_gate_mean": float(out["patch_gate"].detach().float().mean().cpu()),
        "patch_gate_std": float(out["patch_gate"].detach().float().std(unbiased=False).cpu()),
    })
    if pt.any(valid):
        g = gate[valid]
        y = labels[valid]
        metrics["patch_nearest_positive_rate"] = float(y.mean().cpu())
        if g.numel() > 2 and float(g.std(unbiased=False).item()) > 1e-8 and float(y.std(unbiased=False).item()) > 1e-8:
            metrics["patch_gate_label_corr"] = float(pt.corrcoef(pt.stack([g, y]))[0, 1].clamp(-1.0, 1.0).cpu())
        cutoff = pt.quantile(g, 0.8) if g.numel() >= 5 else g.max()
        high = g >= cutoff
        if pt.any(high):
            metrics["patch_high_gate_positive_rate"] = float(y[high].mean().cpu())
            metrics["patch_high_gate_enrichment"] = float((y[high].mean() / y.mean().clamp_min(1e-6)).cpu())
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
    "surface_graph_view_count",
    "surface_graph_router_entropy",
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
    "triplet_gaussian_raw_triplet_count",
    "triplet_gaussian_preselect_topm",
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
    "final_res_skip_gate_mean",
    "final_res_skip_gate_std",
    "final_res_skip_update_norm",
    "final_res_skip_cos_to_final",
    "final_res_skip_delta_logit_abs",
    "patch_usage_entropy",
    "patch_spatial_compactness",
    "patch_normal_consistency",
    "patch_token_norm_mean",
    "patch_token_diversity",
    "patch_residue_attention_entropy",
    "patch_residue_attention_max",
    "patch_residue_effective_count",
    "patch_context_update_norm",
    "patch_residue_gate_mean",
    "patch_residue_gate_std",
)


def sigma_diagnostics(out):
    metrics = {}
    for key in SIGMA_DIAGNOSTIC_KEYS:
        value = out.get(key)
        if value is not None:
            metrics[key] = float(value.detach().float().cpu())
    return metrics


def calibration_metrics(scores, labels, bins=15):
    if scores.numel() == 0:
        return {"brier": float("nan"), "nll": float("nan"), "ece": float("nan")}
    scores = scores.float().clamp(1e-7, 1.0 - 1e-7)
    labels = labels.float()
    brier = (scores - labels).square().mean()
    nll = -(labels * scores.log() + (1.0 - labels) * (1.0 - scores).log()).mean()
    edges = pt.linspace(0.0, 1.0, int(bins) + 1)
    ece = scores.new_zeros(())
    for idx in range(int(bins)):
        if idx + 1 == int(bins):
            selected = (scores >= edges[idx]) & (scores <= edges[idx + 1])
        else:
            selected = (scores >= edges[idx]) & (scores < edges[idx + 1])
        if pt.any(selected):
            frac = selected.float().mean()
            ece = ece + frac * (scores[selected].mean() - labels[selected].mean()).abs()
    return {"brier": float(brier), "nll": float(nll), "ece": float(ece)}


def unique_structure_weights(structure_keys, dtype=pt.float32):
    counts = {}
    for key in structure_keys:
        counts[key] = counts.get(key, 0) + 1
    return pt.tensor([1.0 / max(counts[key], 1) for key in structure_keys], dtype=dtype)


def weighted_binary_metrics(scores, labels, weights):
    if scores.numel() == 0:
        return {"aupr": float("nan"), "auroc": float("nan")}
    scores = scores.float()
    labels = labels.float()
    weights = weights.float()
    order = pt.argsort(scores, descending=True, stable=True)
    y_desc = labels[order]
    w_desc = weights[order]
    tp = pt.cumsum(w_desc * y_desc, dim=0)
    fp = pt.cumsum(w_desc * (1.0 - y_desc), dim=0)
    precision = tp / (tp + fp).clamp_min(1e-12)
    total_pos = (weights * labels).sum()
    aupr = (precision * w_desc * y_desc).sum() / total_pos.clamp_min(1e-12)

    order_asc = pt.argsort(scores, descending=False, stable=True)
    s_asc = scores[order_asc]
    y_asc = labels[order_asc]
    w_asc = weights[order_asc]
    _, inverse = pt.unique_consecutive(s_asc, return_inverse=True)
    group_count = int(inverse[-1].item()) + 1
    positive_by_group = scores.new_zeros(group_count)
    negative_by_group = scores.new_zeros(group_count)
    positive_by_group.index_add_(0, inverse, w_asc * y_asc)
    negative_by_group.index_add_(0, inverse, w_asc * (1.0 - y_asc))
    negative_before = pt.cumsum(negative_by_group, dim=0) - negative_by_group
    concordant = (positive_by_group * (negative_before + 0.5 * negative_by_group)).sum()
    total_neg = (weights * (1.0 - labels)).sum()
    auroc = concordant / (total_pos * total_neg).clamp_min(1e-12)
    return {"aupr": float(aupr), "auroc": float(auroc)}


def run_eval(model, loader, device, pos_weight, prefix, base_aux_loss_weight=0.2):
    model.eval()
    losses = []
    head_names = ("prior", "surface_conditioned", "final")
    head_output_keys = {
        "prior": "prior_logits",
        "surface_conditioned": "surface_conditioned_logits",
        "final": "logits",
    }
    head_losses = {name: [] for name in head_names}
    head_scores_all = {name: [] for name in head_names}
    head_case_rows = {name: [] for name in head_names}
    all_labels = []
    pooled_scores_all = []
    structure_keys = []
    skipped_examples = 0
    skipped_batches = 0
    gate_vals = []
    delta_vals = []
    patch_diag_values = {}
    gaussian_top_records = []
    with pt.no_grad():
        for batch in tqdm(loader, desc=f"eval_s1_{prefix}"):
            skipped_examples += int(batch.get("skipped_count", 0) or 0)
            if batch.get("skip_batch"):
                skipped_batches += 1
                continue
            batch = move_batch_to_device(batch, device)
            out = model_forward(model, batch)
            batch_head_losses = {
                name: masked_bce(out[key], batch["y"], batch["mask"], pos_weight)
                for name, key in head_output_keys.items()
            }
            loss = batch_head_losses["final"] + float(base_aux_loss_weight) * batch_head_losses["surface_conditioned"]
            losses.append(float(loss.cpu()))
            for name in head_names:
                head_losses[name].append(float(batch_head_losses[name].cpu()))
            gate_vals.append(out["surface_gate"].detach().float().cpu())
            delta_vals.append(out["surface_delta"].detach().float().cpu())
            for key, value in patch_diagnostics(out, batch).items():
                patch_diag_values.setdefault(key, []).append(float(value))
            for key, value in sigma_diagnostics(out).items():
                patch_diag_values.setdefault(key, []).append(float(value))
            batch_top_records = out.get("gaussian_top_records", []) if isinstance(out, dict) else []
            if batch_top_records:
                for rec in batch_top_records:
                    rec = dict(rec)
                    sample_idx = int(rec.get("sample_index", -1))
                    if 0 <= sample_idx < len(batch["rows"]):
                        row_meta = batch["rows"][sample_idx]
                        rec["case_id"] = row_meta.get("apex_case_id", row_meta.get("system_id", str(sample_idx)))
                        rec["system_id"] = row_meta.get("system_id", "")
                        rec["input_state"] = row_meta.get("input_state", "")
                        rec["receptor_uniprot"] = row_meta.get("receptor_uniprot", "")
                    rec["eval_prefix"] = prefix
                    gaussian_top_records.append(rec)
            offset = 0
            for i, count in enumerate(batch["residue_counts"].detach().cpu().tolist()):
                count = int(count)
                sl = slice(offset, offset + count)
                valid = batch["mask"][sl] > 0.5
                if pt.any(valid):
                    labels = batch["y"][sl][valid].detach().float().cpu()
                    pooled_scores = pt.sigmoid(out["surface_pooled_logits"][sl][valid]).detach().float().cpu()
                    all_labels.append(labels)
                    pooled_scores_all.append(pooled_scores)
                    case_id = batch["rows"][i].get("apex_case_id", batch["rows"][i].get("system_id", str(i)))
                    structure_key = batch["rows"][i].get("receptor_pdb_path", "") or case_id
                    structure_keys.extend([structure_key] * int(labels.numel()))
                    for name, key in head_output_keys.items():
                        scores = pt.sigmoid(out[key][sl][valid]).detach().float().cpu()
                        head_scores_all[name].append(scores)
                        row = case_metrics(scores, labels, case_id)
                        row["head"] = name
                        row["system_id"] = batch["rows"][i].get("system_id", "")
                        row["input_state"] = batch["rows"][i].get("input_state", "")
                        row["receptor_uniprot"] = batch["rows"][i].get("receptor_uniprot", "")
                        row["receptor_pdb_path"] = structure_key
                        head_case_rows[name].append(row)
                offset += count
    labels = pt.cat(all_labels) if all_labels else pt.empty(0)
    final_scores = pt.cat(head_scores_all["final"]) if head_scores_all["final"] else pt.empty(0)
    metrics = aggregate_metrics(final_scores, labels, losses, head_case_rows["final"], prefix)
    weights = unique_structure_weights(structure_keys) if structure_keys else pt.empty(0)
    for name in head_names:
        scores = pt.cat(head_scores_all[name]) if head_scores_all[name] else pt.empty(0)
        aggregate = aggregate_metrics(scores, labels, head_losses[name], head_case_rows[name], f"{prefix}_{name}")
        for key, value in aggregate.items():
            if key != "prefix":
                metrics[f"{name}_{key}"] = value
        for key, value in calibration_metrics(scores, labels).items():
            metrics[f"{name}_{key}"] = value
        weighted = weighted_binary_metrics(scores, labels, weights) if weights.numel() else {"aupr": float("nan"), "auroc": float("nan")}
        metrics[f"{name}_unique_structure_weighted_aupr"] = weighted["aupr"]
        metrics[f"{name}_unique_structure_weighted_auroc"] = weighted["auroc"]
    metrics[f"{prefix}_loss_final"] = sum(head_losses["final"]) / max(len(head_losses["final"]), 1)
    metrics[f"{prefix}_loss_base"] = sum(head_losses["surface_conditioned"]) / max(len(head_losses["surface_conditioned"]), 1)
    if pooled_scores_all:
        metrics[f"{prefix}_surface_pooled_aupr"] = aggregate_metrics(pt.cat(pooled_scores_all), labels, [], [], prefix).get("aupr")
    if gate_vals:
        gates = pt.cat(gate_vals)
        metrics["surface_gate_mean"] = float(gates.mean())
        metrics["surface_gate_std"] = float(gates.std(unbiased=False))
    if delta_vals:
        deltas = pt.cat(delta_vals)
        metrics["surface_delta_std"] = float(deltas.std(unbiased=False))
    for key, values in patch_diag_values.items():
        metrics[key] = sum(values) / max(len(values), 1)
    metrics["skipped_examples"] = skipped_examples
    metrics["skipped_batches"] = skipped_batches
    metrics["gaussian_top_record_count"] = len(gaussian_top_records)
    case_rows = [row for name in head_names for row in head_case_rows[name]]
    return metrics, case_rows, gaussian_top_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate S1 A4-conditioned surface interface field.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-manifest", default="")
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
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--allow-residue-mismatch", action="store_true")
    parser.add_argument("--residue-graph-cache-ks", type=lambda text: tuple(int(x) for x in str(text).split(",") if x), default=(16, 24, 32))
    parser.add_argument("--s1-surface-points", type=int, default=1024)
    parser.add_argument("--s1-surface-graph-k", type=int, default=16)
    parser.add_argument("--surface-extra-graph-ks", default="")
    parser.add_argument("--surface-global-feedback-before-residue", action="store_true")
    parser.add_argument("--s1-surface-lappe-dim", type=int, default=0)
    parser.add_argument("--s1-surface-cache-version", default="s1_surface_proxy_v1")
    parser.add_argument("--require-s1-surface-cache", action="store_true")
    parser.add_argument("--disable-s1-surface-on-the-fly", action="store_true")
    parser.add_argument("--require-true-features", action="store_true")
    parser.add_argument("--true-feature-version", default="true_sasa_dssp_v1")
    parser.add_argument("--require-teacher-cache", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-skip-fraction", type=float, default=0.01)
    parser.add_argument("--allow-non-strict-load", action="store_true")
    args = parser.parse_args()
    if args.matmul_precision:
        pt.set_float32_matmul_precision(args.matmul_precision)
    checkpoint = pt.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    if int(args.s1_surface_lappe_dim) <= 0:
        args.s1_surface_lappe_dim = int(ckpt_args.get("s1_surface_lappe_dim", 0) or 0)
    if int(args.s1_surface_points) == 1024:
        args.s1_surface_points = int(ckpt_args.get("s1_surface_points", args.s1_surface_points) or args.s1_surface_points)
    if int(args.s1_surface_graph_k) == 16:
        args.s1_surface_graph_k = int(ckpt_args.get("s1_surface_graph_k", args.s1_surface_graph_k) or args.s1_surface_graph_k)
    rows = load_rows(args.manifest, args.states, max_cases=args.max_cases)
    if args.train_manifest:
        train_rows = load_rows(args.train_manifest, "")
        pos_weight_value, label_stats = estimate_pos_weight(train_rows, factor=float(ckpt_args.get("pos_weight_factor", 0.5)))
    else:
        pos_weight_value, label_stats = 1.0, {}
    pos_weight = pt.tensor(float(pos_weight_value), dtype=pt.float)
    loader = build_loader(rows, args)
    device = pt.device(args.device if args.device == "cpu" or pt.cuda.is_available() else "cpu")
    model = build_model_from_ckpt_args(ckpt_args).to(device)
    incompatible = model.load_state_dict(model_state, strict=not args.allow_non_strict_load)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if (missing or unexpected) and not args.allow_non_strict_load:
        raise RuntimeError(f"Strict checkpoint load failed: missing={missing}, unexpected={unexpected}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "load_state_report.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "missing": list(missing),
                "unexpected": list(unexpected),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics, case_rows, gaussian_top_records = run_eval(
        model,
        loader,
        device,
        pos_weight,
        args.prefix,
        base_aux_loss_weight=float(ckpt_args.get("base_aux_loss_weight", 0.2)),
    )
    skip_fraction = int(metrics.get("skipped_examples", 0) or 0) / max(len(rows), 1)
    if skip_fraction > args.max_skip_fraction:
        raise RuntimeError(f"Skip fraction {skip_fraction:.4f} exceeded --max-skip-fraction {args.max_skip_fraction:.4f}")
    payload = {
        "metrics": metrics,
        "label_stats_from_train_manifest": label_stats,
        "pos_weight": pos_weight_value,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "num_manifest_rows": len(rows),
    }
    (output_dir / f"{args.prefix}_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if gaussian_top_records:
        top_path = output_dir / f"{args.prefix}_gaussian_top_records.jsonl"
        top_path.write_text("".join(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n" for rec in gaussian_top_records), encoding="utf-8")
    write_csv(output_dir / f"{args.prefix}_metrics.csv", [metrics])
    write_csv(output_dir / f"{args.prefix}_case_metrics.csv", case_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
