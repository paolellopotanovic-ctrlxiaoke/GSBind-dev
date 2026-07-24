"""Complete modular M4 predictor.

This is the paper-facing implementation. The frozen legacy module remains only
as a numerical oracle for equivalence tests.
"""

import torch as pt

from apex_multigraph_plus.model import (
    GlobalContextFeedback,
    InterfaceHead,
    StaticMultiViewGVPEncoder,
    StaticResidueMultiplexGVPEncoder,
)
from apex_generic_gvp_augfeat.model import AugFeatureBuilder
from apex_s1_surface_field.semantic_patch import SemanticConnectedPatchHead
from apex_teacher_prior.modules import TeacherHiddenAdapter

from .execution import M4ExecutionMixin
from .cross_scale import (
    AttentionResidueToSurfacePool,
    AttentionSurfaceToResiduePool,
    MotifPatchCrossAttention,
    StructuralCommunityDynamicAttention,
    SurfaceBusFromViews,
)
from .gaussian_projection import (
    build_gaussian_projector,
)
from .pair_gaussian import (
    build_pair_gaussian_modules,
)
from .residue_encoder import (
    AtomPreResidueEncoder,
    GearStyleResidueGraphRefiner,
    OAGNNResidueGraphRefiner,
    SerialK16K32MultiViewGVPEncoder,
)
from .surface_encoder import (
    DynamicSurfacePatchCluster,
    ResidueInternalLaplacianOperator,
    ResidueVectorToSurfaceCrossAttention,
    SurfaceLaplacianSpectralFilter,
    SurfacePointGraphEncoder,
    SurfacePointGVPGraphEncoder,
)


def _parse_positive_ints(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace(";", ",").split(",")
    out = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        k = int(text)
        if k > 0 and k not in out:
            out.append(k)
    return tuple(out)


class S1SurfaceFieldPredictor(M4ExecutionMixin, pt.nn.Module):
    """M4 receptor-only protein-interface predictor.

    This composition root registers the residue, Gaussian surface-field,
    surface-graph, cross-scale fusion, semantic-patch, and output modules.
    Runtime operation order is implemented by :class:`M4ExecutionMixin`.
    """

    def __init__(
        self,
        atom_feature_dim=123,
        dim=128,
        vector_dim=16,
        graph_views=("seq", "k16", "k32", "exposure"),
        view_layers=2,
        rbf_bins=16,
        dropout=0.1,
        router_mode="global",
        use_augmented_features=True,
        residue_feature_dim=43,
        use_true_features=False,
        true_feature_dim=10,
        surface_points=1024,
        surface_graph_k=16,
        surface_extra_graph_k=0,
        surface_extra_graph_ks="",
        surface_graph_flow_mode="parallel",
        surface_layers=2,
        surface_residue_topk=8,
        residue_surface_topk=24,
        sigma_tangent=6.0,
        sigma_normal=3.0,
        projection_mode="single",
        multiscale_sigmas="4:2,8:4,14:7",
        surface_attention_prior_weight=0.0,
        surface_attention_sigma_tangent=8.0,
        surface_attention_sigma_normal=4.0,
        surface_to_residue_mode="attention",
        phys_feature_dim=0,
        phys_attention_bias_weight=0.0,
        phys_context_weight=0.0,
        phys_message_weight=0.0,
        surface_patch_mode="none",
        surface_patch_count=32,
        surface_patch_beta=0.1,
        surface_patch_attention_heads=4,
        surface_lappe_dim=0,
        surface_lappe_mode="none",
        surface_spectral_filter_dim=0,
        internal_operator_mode="none",
        internal_operator_k=16,
        internal_operator_layers=2,
        internal_operator_beta=0.1,
        surface_encoder_mode="graph",
        surface_fusion_mode="delta",
        global_feedback_rounds=1,
        global_feedback_interleave_residue_graph=False,
        surface_global_feedback_before_residue=False,
        surface_global_feedback_rounds=1,
        core_cycles=1,
        surface_vector_bridge_mode="none",
        surface_co_update_mode="none",
        surface_co_update_layers=1,
        surface_vector_state_mode="reinit",
        patch_state_mode="stateless",
        surface_serial_stage_mode="none",
        surface_round_end_r2s_mode="attention",
        post_final_residue_global_feedback_rounds=0,
        final_residue_skip_mode="none",
        final_residue_skip_source="multiview",
        final_residue_skip_beta=0.0,
        final_residue_skip_init_zero=False,
        pair_gaussian_mode="none",
        pair_gaussian_topk=8,
        pair_gaussian_beta=0.0,
        pair_gaussian_sigma_tangent=8.0,
        pair_gaussian_sigma_normal=4.0,
        pair_gaussian_sigma_mode="fixed",
        pair_gaussian_sigma_min=0.5,
        pair_gaussian_beta_mode="fixed",
        pair_gaussian_beta_min=0.0,
        pair_gaussian_beta_max=0.3,
        pair_gaussian_beta_init=0.05,
        pair_gaussian_content_mode="none",
        pair_gaussian_content_weight=1.0,
        pair_gaussian_rho_condition_mode="none",
        pair_gaussian_vector_condition_mode="none",
        pair_gaussian_single_compete_mode="none",
        pair_gaussian_single_rescue_beta=0.25,
        pair_gaussian_chunk_size=0,
        pair_gaussian_chunk_checkpoint=True,
        pair_gaussian_chunk_diagnostics="full",
        pair_gaussian_only=False,
        pair_gaussian_fusion_mode="init_only",
        pair_gaussian_joint_mode="independent",
        pair_gaussian_multiview_mode="none",
        pair_gaussian_multiview_beta=0.1,
        pair_gaussian_multiview_chunk_size=256,
        triplet_gaussian_mode="none",
        triplet_gaussian_topk=8,
        triplet_gaussian_beta=0.0,
        triplet_gaussian_beta_mode="fixed",
        triplet_gaussian_beta_min=0.0,
        triplet_gaussian_beta_max=0.2,
        triplet_gaussian_beta_init=0.03,
        triplet_gaussian_sigma_tangent=8.0,
        triplet_gaussian_sigma_normal=4.0,
        triplet_gaussian_fusion_mode="init_only",
        triplet_gaussian_score_temperature=1.0,
        triplet_gaussian_score_clamp=0.0,
        triplet_gaussian_preselect_topm=0,
        residue_amplitude_mode="none",
        residue_amplitude_min=0.25,
        residue_amplitude_max=2.0,
        surface_delta_scale=0.2,
        residue_vector_surface_mode="none",
        residue_vector_surface_topk=16,
        teacher_hidden_dim=0,
        gear_mode="none",
        gear_relation_k=32,
        gear_layers=2,
        gear_view_mode="none",
        oagnn_mode="none",
        oagnn_relation_k=32,
        oagnn_layers=2,
        token_encoder_mode="none",
        token_encoder_layers=2,
        token_encoder_heads=4,
        edge_pair_mode="none",
        cross_view_edge_mode="none",
        cross_view_edge_beta=0.1,
        cross_view_edge_heads=4,
        residue_encoder_mode="multiview",
        residue_graph_flow_mode="parallel",
        sequence_pos_enc_mode="none",
        sequence_pos_enc_dim=16,
        atom_residue_encoder_mode="mean",
        atom_graph_k=8,
        atom_contact_k=16,
        atom_contact_radius=5.0,
        atom_pre_encoder_mode="none",
        atom_pre_dim=64,
        atom_pre_layers=2,
        atom_pre_topk=16,
        atom_pre_cutoff=5.0,
        atom_pre_dropout=0.05,
        post_final_fourview_mode="none",
        post_final_fourview_beta=1.0,
        surface_bus_mode="none",
        surface_bus_beta=1.0,
        surface_bus_topk=16,
        struct_community_mode="none",
        struct_community_count=16,
        struct_community_geom_bias=0.25,
        stochastic_head_mode="none",
        ctmc_diffusion_steps=3,
        ctmc_diffusion_gamma=0.25,
        ctmc_diffusion_topk=16,
        poisson_lambda_max=20.0,
        pocket_mass_topk=16,
        motif_patch_mode="none",
        motif_patch_count=32,
        motif_patch_beta=0.1,
        motif_patch_attention_heads=4,
        motif_patch_geom_bias=0.1,
        semantic_patch_mode="none",
        semantic_patch_query_count=8,
        semantic_patch_beta=0.03,
        semantic_patch_radius_min=2.0,
        semantic_patch_radius_max=16.0,
        semantic_patch_adjacency_weight=0.25,
        semantic_patch_active_topk=4,
        semantic_patch_graph_hops=2,
        semantic_patch_geodesic_max_steps=96,
        semantic_patch_geodesic_normal_penalty=1.0,
        semantic_patch_geodesic_tangent_penalty=1.0,
        semantic_patch_detach_backbone=True,
        semantic_patch_boundary_mix=0.1,
        semantic_patch_boundary_temperature=0.7,
    ):
        super().__init__()
        # Stage 1: atom features and multi-view residue graph backbone.
        self.use_augmented_features = bool(use_augmented_features)
        self.residue_feature_dim = int(residue_feature_dim)
        self.use_true_features = bool(use_true_features)
        self.true_feature_dim = int(true_feature_dim)
        self.phys_feature_dim = int(phys_feature_dim)
        extra_dim = 0
        if self.use_augmented_features:
            extra_dim += self.residue_feature_dim
        if self.use_true_features:
            extra_dim += self.true_feature_dim
        self.atom_pre_encoder_mode = str(atom_pre_encoder_mode)
        if self.atom_pre_encoder_mode not in {"none", "mpnn"}:
            raise ValueError("atom_pre_encoder_mode must be one of: none, mpnn")
        self.atom_pre_encoder = None
        if self.atom_pre_encoder_mode == "mpnn":
            self.atom_pre_encoder = AtomPreResidueEncoder(
                atom_feature_dim=atom_feature_dim,
                dim=dim,
                atom_dim=atom_pre_dim,
                layers=atom_pre_layers,
                topk=atom_pre_topk,
                cutoff=atom_pre_cutoff,
                dropout=atom_pre_dropout,
            )
            extra_dim += int(dim)
        self.last_extra_diagnostics = {}
        self.vector_dim = int(vector_dim)
        self.oagnn_mode = str(oagnn_mode)
        if self.oagnn_mode not in {"none", "oa_edge", "official"}:
            raise ValueError("oagnn_mode must be one of: none, oa_edge, official")
        self.residue_encoder_mode = str(residue_encoder_mode)
        if self.residue_encoder_mode not in {"multiview", "multiplex"}:
            raise ValueError("residue_encoder_mode must be one of: multiview, multiplex")
        self.residue_graph_flow_mode = str(residue_graph_flow_mode)
        serial_graph_flow_views = {
            "k16_then_k32": ("k16", "k32"),
            "k8_then_k16_then_k32_then_k64": ("k8", "k16", "k32", "k64"),
        }
        if self.residue_graph_flow_mode not in {"parallel", *serial_graph_flow_views.keys()}:
            raise ValueError(
                "residue_graph_flow_mode must be one of: parallel, "
                + ", ".join(sorted(serial_graph_flow_views))
            )
        if self.residue_graph_flow_mode != "parallel" and self.residue_encoder_mode != "multiview":
            raise ValueError("serial residue_graph_flow_mode requires residue_encoder_mode=multiview")
        if self.residue_graph_flow_mode != "parallel" and str(cross_view_edge_mode) != "none":
            raise ValueError("serial residue_graph_flow_mode requires cross_view_edge_mode=none")
        encoder_view_layers = int(oagnn_layers) if self.oagnn_mode != "none" else int(view_layers)
        if self.residue_encoder_mode == "multiplex":
            encoder_cls = StaticResidueMultiplexGVPEncoder
        elif self.residue_graph_flow_mode in serial_graph_flow_views:
            encoder_cls = SerialK16K32MultiViewGVPEncoder
        else:
            encoder_cls = StaticMultiViewGVPEncoder
        encoder_kwargs = dict(
            atom_feature_dim=atom_feature_dim,
            dim=dim,
            vector_dim=vector_dim,
            graph_views=graph_views,
            view_layers=encoder_view_layers,
            rbf_bins=rbf_bins,
            dropout=dropout,
            extra_residue_feature_dim=extra_dim,
            router_mode=router_mode,
            gear_view_mode=gear_view_mode,
            oagnn_mode=self.oagnn_mode,
            token_encoder_mode=token_encoder_mode,
            token_encoder_layers=token_encoder_layers,
            token_encoder_heads=token_encoder_heads,
            edge_pair_mode=edge_pair_mode,
            cross_view_edge_mode=cross_view_edge_mode,
            cross_view_edge_beta=cross_view_edge_beta,
            cross_view_edge_heads=cross_view_edge_heads,
            sequence_pos_enc_mode=sequence_pos_enc_mode,
            sequence_pos_enc_dim=sequence_pos_enc_dim,
            atom_residue_encoder_mode=atom_residue_encoder_mode,
            atom_graph_k=atom_graph_k,
            atom_contact_k=atom_contact_k,
            atom_contact_radius=atom_contact_radius,
        )
        if self.residue_graph_flow_mode in serial_graph_flow_views:
            encoder_kwargs["serial_views"] = serial_graph_flow_views[self.residue_graph_flow_mode]
        self.encoder = encoder_cls(**encoder_kwargs)
        self.global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.gear_mode = str(gear_mode)
        self.gear_refiner = None
        if self.gear_mode != "none":
            if self.gear_mode not in {"rel_edge", "edge_msg"}:
                raise ValueError("gear_mode must be one of: none, rel_edge, edge_msg")
            self.gear_refiner = GearStyleResidueGraphRefiner(
                dim=dim,
                relation_k=gear_relation_k,
                layers=gear_layers,
                dropout=dropout,
                mode=self.gear_mode,
            )
        self.base_head = InterfaceHead(dim=dim, dropout=dropout)
        self.aux_head = InterfaceHead(dim=dim, dropout=dropout)
        self.atom_pre_aux_head = InterfaceHead(dim=dim, dropout=dropout) if self.atom_pre_encoder is not None else None
        self.stochastic_head_mode = str(stochastic_head_mode)
        if self.stochastic_head_mode not in {"none", "poisson", "poisson_pocket", "ctmc_poisson", "event_competition"}:
            raise ValueError("stochastic_head_mode must be one of: none, poisson, poisson_pocket, ctmc_poisson, event_competition")
        self.ctmc_diffusion_steps = max(1, int(ctmc_diffusion_steps))
        self.ctmc_diffusion_gamma = float(ctmc_diffusion_gamma)
        self.ctmc_diffusion_topk = max(1, int(ctmc_diffusion_topk))
        self.poisson_lambda_max = float(poisson_lambda_max)
        self.pocket_mass_topk = max(1, int(pocket_mass_topk))
        self.ctmc_rate = None
        self.ctmc_gate = None
        if self.stochastic_head_mode == "ctmc_poisson":
            self.ctmc_rate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2 + 1),
                pt.nn.Linear(dim * 2 + 1, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            self.ctmc_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
                pt.nn.Sigmoid(),
            )
        self.event_competition_head = None
        if self.stochastic_head_mode == "event_competition":
            self.event_competition_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 4),
            )
        self.final_delta = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.surface_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
            pt.nn.Sigmoid(),
        )
        self.post_final_fourview_mode = str(post_final_fourview_mode)
        if self.post_final_fourview_mode not in {"none", "refine_dual_env"}:
            raise ValueError("post_final_fourview_mode must be one of: none, refine_dual_env")
        self.post_final_fourview_beta = float(post_final_fourview_beta)
        self.post_final_graph_head = None
        self.post_final_dual_gate = None
        if self.post_final_fourview_mode != "none":
            self.post_final_graph_head = InterfaceHead(dim=dim, dropout=dropout)
            self.post_final_dual_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 3),
                pt.nn.Linear(dim * 3, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 2),
            )
        self.surface_bus_mode = str(surface_bus_mode)
        if self.surface_bus_mode not in {"none", "static", "static_pre_global", "recurrent"}:
            raise ValueError("surface_bus_mode must be one of: none, static, static_pre_global, recurrent")
        if self.surface_bus_mode != "none" and self.residue_encoder_mode != "multiview":
            raise ValueError("surface_bus_mode requires residue_encoder_mode=multiview.")
        self.surface_bus_beta = float(surface_bus_beta)
        self.surface_bus = None
        self.surface_bus_static_norm = None
        self.surface_bus_view_update = None
        self.surface_bus_view_gate = None
        self.surface_bus_view_norm = None
        view_count = len(tuple(graph_views))
        if self.surface_bus_mode != "none":
            self.surface_bus = SurfaceBusFromViews(
                dim=dim,
                view_count=view_count,
                residue_topk=surface_bus_topk,
                dropout=dropout,
            )
            self.surface_bus_static_norm = pt.nn.LayerNorm(dim)
            self.surface_bus_view_update = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.surface_bus_view_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 2),
                pt.nn.Linear(dim * 2, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
                pt.nn.Sigmoid(),
            )
            self.surface_bus_view_norm = pt.nn.LayerNorm(dim)

        # Stage 2: surface geometry and residue/surface interaction contract.
        self.aug_feature_builder = None
        if self.use_augmented_features:
            self.aug_feature_builder = AugFeatureBuilder(atom_feature_dim=atom_feature_dim, residue_feature_dim=self.residue_feature_dim)
        self.surface_points = int(surface_points)
        self.surface_graph_k = int(surface_graph_k)
        self.surface_extra_graph_k = max(0, int(surface_extra_graph_k))
        self.surface_extra_graph_ks = _parse_positive_ints(surface_extra_graph_ks)
        self.surface_graph_flow_mode = str(surface_graph_flow_mode)
        if self.surface_graph_flow_mode not in {"parallel", "k16_then_k32"}:
            raise ValueError("surface_graph_flow_mode must be one of: parallel, k16_then_k32")
        self.surface_delta_scale = float(surface_delta_scale)
        self.surface_lappe_dim = max(0, int(surface_lappe_dim))
        self.surface_lappe_mode = str(surface_lappe_mode)
        if self.surface_lappe_mode not in {"none", "encoder", "patch", "both"}:
            raise ValueError("surface_lappe_mode must be one of: none, encoder, patch, both")
        if self.surface_lappe_dim <= 0:
            self.surface_lappe_mode = "none"
        self.surface_lappe_proj = None
        if self.surface_lappe_mode in {"encoder", "both"}:
            self.surface_lappe_proj = pt.nn.Sequential(
                pt.nn.LayerNorm(self.surface_lappe_dim),
                pt.nn.Linear(self.surface_lappe_dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
        self.surface_spectral_filter_dim = max(0, int(surface_spectral_filter_dim))
        self.surface_spectral_filter = None
        if self.surface_spectral_filter_dim > 0:
            self.surface_spectral_filter = SurfaceLaplacianSpectralFilter(
                dim=dim,
                lappe_dim=self.surface_spectral_filter_dim,
                dropout=dropout,
            )
        self.internal_operator_mode = str(internal_operator_mode)
        if self.internal_operator_mode not in {"none", "residue_laplacian"}:
            raise ValueError("internal_operator_mode must be one of: none, residue_laplacian")
        self.internal_operator = None
        if self.internal_operator_mode == "residue_laplacian":
            self.internal_operator = ResidueInternalLaplacianOperator(
                dim=dim,
                relation_k=internal_operator_k,
                layers=internal_operator_layers,
                beta=internal_operator_beta,
                dropout=dropout,
            )
        self.surface_encoder_mode = str(surface_encoder_mode)
        if self.surface_encoder_mode not in {"graph", "attention", "gvp"}:
            raise ValueError("surface_encoder_mode must be one of: graph, attention, gvp")
        if self.surface_graph_flow_mode == "k16_then_k32":
            if self.surface_encoder_mode == "attention":
                raise ValueError("surface_graph_flow_mode=k16_then_k32 requires graph or gvp surface_encoder_mode")
            if self.surface_graph_k != 16 or 32 not in set(self.surface_extra_graph_ks):
                raise ValueError("surface_graph_flow_mode=k16_then_k32 requires s1_surface_graph_k=16 and surface_extra_graph_ks including 32")
        self.surface_fusion_mode = str(surface_fusion_mode)
        if self.surface_fusion_mode not in {"delta", "same_level"}:
            raise ValueError("surface_fusion_mode must be one of: delta, same_level")
        self.global_feedback_rounds = max(1, int(global_feedback_rounds))
        self.global_feedback_interleave_residue_graph = bool(global_feedback_interleave_residue_graph)
        self.surface_global_feedback_before_residue = bool(surface_global_feedback_before_residue)
        self.surface_global_feedback_rounds = max(1, int(surface_global_feedback_rounds))
        self.surface_global_feedback = None
        if self.surface_global_feedback_before_residue:
            self.surface_global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.core_cycles = max(1, int(core_cycles))
        self.surface_vector_bridge_mode = str(surface_vector_bridge_mode)
        if self.surface_vector_bridge_mode not in {"none", "invariant", "equivariant"}:
            raise ValueError("surface_vector_bridge_mode must be one of: none, invariant, equivariant")
        self.surface_co_update_mode = str(surface_co_update_mode)
        if self.surface_co_update_mode not in {"none", "s2r", "bidir"}:
            raise ValueError("surface_co_update_mode must be one of: none, s2r, bidir")
        self.surface_co_update_layers = max(1, int(surface_co_update_layers))
        self.surface_serial_stage_mode = str(surface_serial_stage_mode)
        if self.surface_serial_stage_mode not in {"none", "pair_then_triplet", "perresidue_then_pair"}:
            raise ValueError("surface_serial_stage_mode must be one of: none, pair_then_triplet, perresidue_then_pair")
        self.surface_round_end_r2s_mode = str(surface_round_end_r2s_mode)
        if self.surface_round_end_r2s_mode not in {"attention", "skip"}:
            raise ValueError("surface_round_end_r2s_mode must be one of: attention, skip")
        self.post_final_residue_global_feedback_rounds = max(0, int(post_final_residue_global_feedback_rounds))
        self.post_final_residue_global_feedback = None
        if self.post_final_residue_global_feedback_rounds > 0:
            self.post_final_residue_global_feedback = GlobalContextFeedback(dim=dim, dropout=dropout)
        self.final_residue_skip_mode = str(final_residue_skip_mode)
        if self.final_residue_skip_mode not in {"none", "gated"}:
            raise ValueError("final_residue_skip_mode must be one of: none, gated")
        self.final_residue_skip_source = str(final_residue_skip_source)
        if self.final_residue_skip_source not in {"multiview"}:
            raise ValueError("final_residue_skip_source must be one of: multiview")
        self.final_residue_skip_beta = float(final_residue_skip_beta)
        self.final_residue_skip_proj = None
        self.final_residue_skip_gate = None
        self.final_residue_skip_norm = None
        if self.final_residue_skip_mode == "gated":
            self.final_residue_skip_proj = pt.nn.Linear(dim, dim)
            self.final_residue_skip_gate = pt.nn.Sequential(
                pt.nn.LayerNorm(dim * 4),
                pt.nn.Linear(dim * 4, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, dim),
            )
            self.final_residue_skip_norm = pt.nn.LayerNorm(dim)
            if bool(final_residue_skip_init_zero):
                last = self.final_residue_skip_gate[-1]
                pt.nn.init.zeros_(last.weight)
                pt.nn.init.constant_(last.bias, -2.0)
        self.struct_community_mode = str(struct_community_mode)
        if self.struct_community_mode not in {"none", "dynamic_attn"}:
            raise ValueError("struct_community_mode must be one of: none, dynamic_attn")
        self.struct_community = None
        if self.struct_community_mode == "dynamic_attn":
            self.struct_community = StructuralCommunityDynamicAttention(
                dim=dim,
                num_communities=struct_community_count,
                geom_bias=struct_community_geom_bias,
                dropout=dropout,
            )
        self.motif_patch_mode = str(motif_patch_mode)
        if self.motif_patch_mode not in {"none", "feedback", "dual_cross"}:
            raise ValueError("motif_patch_mode must be one of: none, feedback, dual_cross")
        self.motif_patch = None
        if self.motif_patch_mode != "none":
            self.motif_patch = MotifPatchCrossAttention(
                dim=dim,
                num_motifs=motif_patch_count,
                mode=self.motif_patch_mode,
                beta=motif_patch_beta,
                heads=motif_patch_attention_heads,
                dropout=dropout,
                geom_bias=motif_patch_geom_bias,
            )
        self.surface_vector_state_mode = str(surface_vector_state_mode)
        if self.surface_vector_state_mode not in {"reinit", "persistent"}:
            raise ValueError("surface_vector_state_mode must be one of: reinit, persistent")
        self.residue_vector_surface_mode = str(residue_vector_surface_mode)
        if self.residue_vector_surface_mode not in {"none", "cross_attn"}:
            raise ValueError("residue_vector_surface_mode must be one of: none, cross_attn")
        self.residue_vector_surface_topk = max(1, int(residue_vector_surface_topk))
        self.patch_state_mode = str(patch_state_mode)
        if self.patch_state_mode not in {"stateless", "persistent"}:
            raise ValueError("patch_state_mode must be one of: stateless, persistent")
        self.pair_gaussian_mode = str(pair_gaussian_mode)
        if self.pair_gaussian_mode not in {"none", "k_nearest", "random"}:
            raise ValueError("pair_gaussian_mode must be one of: none, k_nearest, random")
        self.pair_gaussian_only = bool(pair_gaussian_only)
        self.pair_gaussian_fusion_mode = str(pair_gaussian_fusion_mode)
        if self.pair_gaussian_fusion_mode not in {"init_only", "deep", "coattn"}:
            raise ValueError("pair_gaussian_fusion_mode must be one of: init_only, deep, coattn")
        self.pair_gaussian_joint_mode = str(pair_gaussian_joint_mode)
        if self.pair_gaussian_joint_mode not in {"independent", "joint_residual", "joint_only"}:
            raise ValueError("pair_gaussian_joint_mode must be one of: independent, joint_residual, joint_only")
        self.pair_gaussian_multiview_mode = str(pair_gaussian_multiview_mode)
        if self.pair_gaussian_multiview_mode not in {"none", "init", "deep"}:
            raise ValueError("pair_gaussian_multiview_mode must be one of: none, init, deep")
        self.pair_gaussian_multiview_beta = float(pair_gaussian_multiview_beta)
        self.pair_gaussian_multiview_chunk_size = max(1, int(pair_gaussian_multiview_chunk_size))
        self.pair_gaussian_beta_mode = str(pair_gaussian_beta_mode)
        if self.pair_gaussian_beta_mode not in {"fixed", "learned_deep"}:
            raise ValueError("pair_gaussian_beta_mode must be one of: fixed, learned_deep")
        self.pair_gaussian_beta_min = float(pair_gaussian_beta_min)
        self.pair_gaussian_beta_max = float(pair_gaussian_beta_max)
        if self.pair_gaussian_beta_max <= self.pair_gaussian_beta_min:
            raise ValueError("pair_gaussian_beta_max must be greater than pair_gaussian_beta_min")
        self.pair_gaussian_beta_init = float(pair_gaussian_beta_init)
        self.pair_gaussian_deep_beta_logits = None
        self.pair_gaussian_content_mode = str(pair_gaussian_content_mode)
        self.pair_gaussian_content_weight = float(pair_gaussian_content_weight)
        self.pair_gaussian_rho_condition_mode = str(pair_gaussian_rho_condition_mode)
        if self.pair_gaussian_rho_condition_mode not in {"none", "surface_dot"}:
            raise ValueError("pair_gaussian_rho_condition_mode must be one of: none, surface_dot")
        self.pair_gaussian_vector_condition_mode = str(pair_gaussian_vector_condition_mode)
        if self.pair_gaussian_vector_condition_mode not in {"none", "surface_frame"}:
            raise ValueError("pair_gaussian_vector_condition_mode must be one of: none, surface_frame")
        self.pair_gaussian_single_compete_mode = str(pair_gaussian_single_compete_mode)
        if self.pair_gaussian_single_compete_mode not in {"none", "conservative_rescue", "strong_gate"}:
            raise ValueError("pair_gaussian_single_compete_mode must be one of: none, conservative_rescue, strong_gate")
        self.pair_gaussian_single_rescue_beta = float(pair_gaussian_single_rescue_beta)
        self.pair_gaussian_chunk_size = max(0, int(pair_gaussian_chunk_size))
        self.pair_gaussian_chunk_checkpoint = bool(pair_gaussian_chunk_checkpoint)
        self.pair_gaussian_chunk_diagnostics = str(pair_gaussian_chunk_diagnostics)
        if self.pair_gaussian_chunk_diagnostics not in {"full", "none"}:
            raise ValueError("pair_gaussian_chunk_diagnostics must be one of: full, none")
        self.triplet_gaussian_mode = str(triplet_gaussian_mode)
        if self.triplet_gaussian_mode not in {"none", "k_nearest", "random"}:
            raise ValueError("triplet_gaussian_mode must be one of: none, k_nearest, random")
        self.triplet_gaussian_beta_mode = str(triplet_gaussian_beta_mode)
        if self.triplet_gaussian_beta_mode not in {"fixed", "learned_deep"}:
            raise ValueError("triplet_gaussian_beta_mode must be one of: fixed, learned_deep")
        self.triplet_gaussian_beta_min = float(triplet_gaussian_beta_min)
        self.triplet_gaussian_beta_max = float(triplet_gaussian_beta_max)
        if self.triplet_gaussian_beta_max <= self.triplet_gaussian_beta_min:
            raise ValueError("triplet_gaussian_beta_max must be greater than triplet_gaussian_beta_min")
        self.triplet_gaussian_beta_init = float(triplet_gaussian_beta_init)
        self.triplet_gaussian_fusion_mode = str(triplet_gaussian_fusion_mode)
        if self.triplet_gaussian_fusion_mode not in {"init_only", "deep"}:
            raise ValueError("triplet_gaussian_fusion_mode must be one of: init_only, deep")
        self.triplet_gaussian_deep_beta_logits = None
        self.residue_amplitude_mode = str(residue_amplitude_mode)
        if self.residue_amplitude_mode not in {"none", "bounded"}:
            raise ValueError("residue_amplitude_mode must be one of: none, bounded")

        # Stage 3: residue-conditioned Gaussian surface field.
        self.projection_mode = str(projection_mode)
        self.res_to_surface = build_gaussian_projector(
            projection_mode=self.projection_mode,
            dim=dim,
            surface_residue_topk=surface_residue_topk,
            sigma_tangent=sigma_tangent,
            sigma_normal=sigma_normal,
            multiscale_sigmas=multiscale_sigmas,
            residue_amplitude_mode=self.residue_amplitude_mode,
            residue_amplitude_min=residue_amplitude_min,
            residue_amplitude_max=residue_amplitude_max,
            phys_feature_dim=phys_feature_dim,
            true_feature_dim=true_feature_dim,
            atom_feature_dim=atom_feature_dim,
            surface_patch_count=surface_patch_count,
            dropout=dropout,
        )
        if self.surface_encoder_mode == "gvp":
            self.surface_encoder = SurfacePointGVPGraphEncoder(
                dim=dim,
                vector_dim=vector_dim,
                geom_dim=6,
                dropout=dropout,
                layers=surface_layers,
            )
        else:
            self.surface_encoder = SurfacePointGraphEncoder(dim=dim, geom_dim=6, dropout=dropout, layers=surface_layers)
        if self.surface_extra_graph_ks and self.surface_encoder_mode == "attention":
            raise ValueError("surface_extra_graph_ks requires graph or gvp surface_encoder_mode.")
        self.surface_extra_encoders = pt.nn.ModuleList()
        self.surface_graph_logits = None
        self.surface_graph_fusion_norm = None
        if self.surface_extra_graph_ks:
            for _k in self.surface_extra_graph_ks:
                if self.surface_encoder_mode == "gvp":
                    self.surface_extra_encoders.append(
                        SurfacePointGVPGraphEncoder(
                            dim=dim,
                            vector_dim=vector_dim,
                            geom_dim=6,
                            dropout=dropout,
                            layers=surface_layers,
                        )
                    )
                else:
                    self.surface_extra_encoders.append(
                        SurfacePointGraphEncoder(dim=dim, geom_dim=6, dropout=dropout, layers=surface_layers)
                    )
            self.surface_graph_logits = pt.nn.Parameter(pt.zeros(1 + len(self.surface_extra_graph_ks)))
            self.surface_graph_fusion_norm = pt.nn.LayerNorm(dim)
        self.last_surface_graph_diagnostics = {}
        heads = 4 if dim % 4 == 0 else 1
        self.surface_attention_geom = pt.nn.Sequential(pt.nn.LayerNorm(6), pt.nn.Linear(6, dim))
        self.surface_attention_encoder = pt.nn.TransformerEncoder(
            pt.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=max(1, int(surface_layers)),
        )
        self.surface_to_residue = AttentionSurfaceToResiduePool(
            dim=dim,
            surface_topk=residue_surface_topk,
            dropout=dropout,
            mode=surface_to_residue_mode,
            gaussian_prior_weight=surface_attention_prior_weight,
            sigma_tangent=surface_attention_sigma_tangent,
            sigma_normal=surface_attention_sigma_normal,
            phys_feature_dim=phys_feature_dim,
            phys_attention_bias_weight=phys_attention_bias_weight,
            phys_context_weight=phys_context_weight,
            phys_message_weight=phys_message_weight,
            surface_vector_bridge_mode=self.surface_vector_bridge_mode,
            surface_vector_dim=vector_dim,
        )
        self.residue_vector_to_surface = None
        if self.residue_vector_surface_mode == "cross_attn":
            self.residue_vector_to_surface = ResidueVectorToSurfaceCrossAttention(
                dim=dim,
                vector_dim=vector_dim,
                geom_dim=6,
                residue_topk=self.residue_vector_surface_topk,
                dropout=dropout,
            )
        self.residue_to_surface = AttentionResidueToSurfacePool(
            dim=dim,
            residue_topk=surface_residue_topk,
            dropout=dropout,
        )
        self.surface_patch_cluster = None
        if str(surface_patch_mode) != "none":
            self.surface_patch_cluster = DynamicSurfacePatchCluster(
                dim=dim,
                geom_dim=6,
                num_patches=surface_patch_count,
                mode=surface_patch_mode,
                beta=surface_patch_beta,
                dropout=dropout,
                attention_heads=surface_patch_attention_heads,
                lappe_dim=self.surface_lappe_dim if self.surface_lappe_mode in {"patch", "both"} else 0,
            )
        self.semantic_patch_mode = str(semantic_patch_mode)
        self.semantic_patch_head = None
        if self.semantic_patch_mode != "none":
            if str(surface_patch_mode) != "connected_pool_adj":
                raise ValueError("semantic patch experiments require surface_patch_mode=connected_pool_adj")
            self.semantic_patch_head = SemanticConnectedPatchHead(
                dim=dim,
                query_count=semantic_patch_query_count,
                mode=self.semantic_patch_mode,
                beta=semantic_patch_beta,
                radius_min=semantic_patch_radius_min,
                radius_max=semantic_patch_radius_max,
                adjacency_weight=semantic_patch_adjacency_weight,
                active_topk=semantic_patch_active_topk,
                graph_hops=semantic_patch_graph_hops,
                geodesic_max_steps=semantic_patch_geodesic_max_steps,
                geodesic_normal_penalty=semantic_patch_geodesic_normal_penalty,
                geodesic_tangent_penalty=semantic_patch_geodesic_tangent_penalty,
                detach_backbone=semantic_patch_detach_backbone,
                boundary_mix=semantic_patch_boundary_mix,
                boundary_temperature=semantic_patch_boundary_temperature,
                dropout=dropout,
                attention_heads=surface_patch_attention_heads,
            )

        # Stage 4: pair/triplet field corrections in registration order.
        self.pair_gaussian = None
        self.pair_gaussian_multiview = None
        self.pair_gaussian_multiview_mix = None
        self.pair_gaussian_multiview_gate = None
        self.pair_gaussian_multiview_norm = None
        self.triplet_gaussian = None
        gaussian_modules = build_pair_gaussian_modules(
            dim=dim,
            vector_dim=vector_dim,
            graph_views=graph_views,
            dropout=dropout,
            pair_mode=self.pair_gaussian_mode,
            pair_topk=pair_gaussian_topk,
            pair_beta=pair_gaussian_beta,
            pair_sigma_tangent=pair_gaussian_sigma_tangent,
            pair_sigma_normal=pair_gaussian_sigma_normal,
            pair_sigma_mode=pair_gaussian_sigma_mode,
            pair_sigma_min=pair_gaussian_sigma_min,
            pair_joint_mode=self.pair_gaussian_joint_mode,
            pair_fusion_mode=self.pair_gaussian_fusion_mode,
            pair_content_mode=self.pair_gaussian_content_mode,
            pair_content_weight=self.pair_gaussian_content_weight,
            pair_rho_condition_mode=self.pair_gaussian_rho_condition_mode,
            pair_vector_condition_mode=pair_gaussian_vector_condition_mode,
            pair_single_compete_mode=self.pair_gaussian_single_compete_mode,
            pair_single_rescue_beta=self.pair_gaussian_single_rescue_beta,
            pair_beta_mode=self.pair_gaussian_beta_mode,
            pair_beta_min=self.pair_gaussian_beta_min,
            pair_beta_max=self.pair_gaussian_beta_max,
            pair_beta_init=self.pair_gaussian_beta_init,
            pair_multiview_mode=self.pair_gaussian_multiview_mode,
            pair_multiview_beta=self.pair_gaussian_multiview_beta,
            surface_co_update_mode=self.surface_co_update_mode,
            surface_co_update_layers=self.surface_co_update_layers,
            core_cycles=self.core_cycles,
            triplet_mode=self.triplet_gaussian_mode,
            triplet_topk=triplet_gaussian_topk,
            triplet_beta=triplet_gaussian_beta,
            triplet_sigma_tangent=triplet_gaussian_sigma_tangent,
            triplet_sigma_normal=triplet_gaussian_sigma_normal,
            triplet_score_temperature=triplet_gaussian_score_temperature,
            triplet_score_clamp=triplet_gaussian_score_clamp,
            triplet_preselect_topm=triplet_gaussian_preselect_topm,
            triplet_beta_mode=self.triplet_gaussian_beta_mode,
            triplet_beta_min=self.triplet_gaussian_beta_min,
            triplet_beta_max=self.triplet_gaussian_beta_max,
            triplet_beta_init=self.triplet_gaussian_beta_init,
            global_feedback_rounds=self.global_feedback_rounds,
            surface_global_feedback_rounds=self.surface_global_feedback_rounds,
        )
        self.pair_gaussian = gaussian_modules["pair"]
        self.pair_gaussian_deep_beta_logits = gaussian_modules["pair_deep_beta"]
        self.pair_gaussian_multiview = gaussian_modules["pair_multiview"]
        self.pair_gaussian_multiview_mix = gaussian_modules["pair_multiview_mix"]
        self.pair_gaussian_multiview_gate = gaussian_modules["pair_multiview_gate"]
        self.pair_gaussian_multiview_norm = gaussian_modules["pair_multiview_norm"]
        self.triplet_gaussian = gaussian_modules["triplet"]
        self.triplet_gaussian_deep_beta_logits = gaussian_modules["triplet_deep_beta"]

        # Stage 5: cross-scale fusion and final residue-level prediction heads.
        self.surface_head = InterfaceHead(dim=dim, dropout=dropout)
        self.same_level_head = InterfaceHead(dim=dim, dropout=dropout)
        self.same_level_fusion = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.core_residue_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.core_residue_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.core_norm = pt.nn.LayerNorm(dim)
        self.surface_cross_update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.surface_cross_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.surface_cross_norm = pt.nn.LayerNorm(dim)
        self.residue_vector_scalar_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(vector_dim),
            pt.nn.Linear(vector_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.residue_vector_scalar_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.teacher_adapter = TeacherHiddenAdapter(dim, teacher_hidden_dim, dropout=dropout) if int(teacher_hidden_dim) > 0 else None
