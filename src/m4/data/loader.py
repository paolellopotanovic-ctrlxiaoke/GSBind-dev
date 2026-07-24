"""Canonical four-cache data loader used by training, evaluation, and inference."""

import torch

from .dataset import S1SurfaceFieldDataset, collate_s1_surface_field_batch


def dataloader_kwargs(args, shuffle=False):
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


def build_loader(rows, args, shuffle=False):
    dataset = S1SurfaceFieldDataset(
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
    )
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs(args, shuffle=shuffle))
