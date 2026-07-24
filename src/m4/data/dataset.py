"""Four-cache M4 dataset and deterministic batch collation."""

import hashlib
from pathlib import Path

import torch as pt

from .manifest import M4ManifestDataset, cache_key_for_row, collate_manifest_batch
from .soft_labels import residue_centers_from_atom_mask, residue_soft_labels
from .surface_proxy import build_surface_proxy_cache, surface_cache_path_for_row
from .teacher_cache import teacher_cache_path_for_row
from ..models.components.surface_cache import build_s1_surface_proxy_cache, s1_surface_cache_path_for_row


class SurfaceAuxManifestDataset(M4ManifestDataset):
    def __init__(
        self,
        *args,
        surface_cache_dir=None,
        require_surface_cache=False,
        build_surface_on_the_fly=True,
        surface_points=96,
        surface_residue_topk=8,
        surface_graph_k=12,
        surface_lappe_dim=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.surface_cache_dir = Path(surface_cache_dir) if surface_cache_dir else None
        self.require_surface_cache = bool(require_surface_cache)
        self.build_surface_on_the_fly = bool(build_surface_on_the_fly)
        self.surface_points = int(surface_points)
        self.surface_residue_topk = int(surface_residue_topk)
        self.surface_graph_k = int(surface_graph_k)
        self.surface_lappe_dim = int(surface_lappe_dim)

    def _attach_surface_cache(self, item, row):
        if item.get("skip"):
            return item
        cache = None
        if self.surface_cache_dir is not None:
            path = surface_cache_path_for_row(
                self.surface_cache_dir,
                row,
                num_surface_points=self.surface_points,
                residue_topk=self.surface_residue_topk,
                graph_k=self.surface_graph_k,
                lappe_dim=self.surface_lappe_dim,
            )
            if path.is_file():
                cache = pt.load(path, map_location="cpu")
            elif self.require_surface_cache:
                return {
                    "skip": True,
                    "skip_reason": f"Missing surface cache for apex_case_id={row.get('apex_case_id', '')}: {path}",
                    "rows": row,
                }
        if cache is None and self.build_surface_on_the_fly:
            cache = build_surface_proxy_cache(
                item,
                num_surface_points=self.surface_points,
                residue_topk=self.surface_residue_topk,
                graph_k=self.surface_graph_k,
                lappe_dim=self.surface_lappe_dim,
            )
        if cache is not None:
            item["surface_cache"] = cache
        return item

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = item.get("rows", self.rows[index])
        return self._attach_surface_cache(item, row)


def collate_surface_aux_batch(batch):
    out = collate_manifest_batch(batch)
    if out.get("skip_batch"):
        return out
    kept = [item for item in batch if not item.get("skip")]
    if any("surface_cache" in item for item in kept):
        if not all("surface_cache" in item for item in kept):
            raise ValueError("Mixed batch with partial surface_cache is not supported.")
        out["surface_caches"] = [item["surface_cache"] for item in kept]
    return out


def true_feature_cache_key_for_row(row, feature_version="true_sasa_dssp_v1"):
    raw = "|".join([cache_key_for_row(row), str(feature_version)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def true_feature_cache_path_for_row(cache_dir, row, feature_version="true_sasa_dssp_v1"):
    return Path(cache_dir) / true_feature_cache_key_for_row(row, feature_version=feature_version)


class MultigraphPlusManifestDataset(SurfaceAuxManifestDataset):
    def __init__(
        self,
        *args,
        true_feature_cache_dir=None,
        require_true_features=False,
        true_feature_version="true_sasa_dssp_v1",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.true_feature_cache_dir = Path(true_feature_cache_dir) if true_feature_cache_dir else None
        self.require_true_features = bool(require_true_features)
        self.true_feature_version = str(true_feature_version)

    def _attach_true_feature_cache(self, item, row):
        if item.get("skip") or self.true_feature_cache_dir is None:
            return item
        path = true_feature_cache_path_for_row(
            self.true_feature_cache_dir,
            row,
            feature_version=self.true_feature_version,
        )
        if not path.is_file():
            message = f"Missing true SASA/DSSP feature cache for apex_case_id={row.get('apex_case_id', '')}: {path}"
            if self.require_true_features:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        cache = pt.load(path, map_location="cpu")
        features = cache["features"].float()
        n_res = int(item["M"].shape[1])
        if int(features.shape[0]) != n_res:
            message = (
                f"True feature cache size mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"features={features.shape[0]} tensor={n_res}"
            )
            if self.require_true_features:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        item["true_residue_features"] = features
        item["true_residue_feature_names"] = cache.get("feature_names", [])
        item["true_residue_feature_report"] = cache.get("report", {})
        return item

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = item.get("rows", self.rows[index])
        return self._attach_true_feature_cache(item, row)


def collate_multigraph_plus_batch(batch):
    out = collate_surface_aux_batch(batch)
    if out.get("skip_batch"):
        return out
    kept = [item for item in batch if not item.get("skip")]
    if any("true_residue_features" in item for item in kept):
        if not all("true_residue_features" in item for item in kept):
            raise ValueError("Mixed batch with partial true_residue_features is not supported.")
        out["true_residue_features"] = pt.cat([item["true_residue_features"] for item in kept], dim=0)
        out["true_residue_feature_names"] = kept[0].get("true_residue_feature_names", [])
    return out


def atom_to_residue_index_from_mask(mask):
    if not isinstance(mask, pt.Tensor) or mask.ndim != 2:
        raise ValueError("Expected a rank-2 atom-to-residue assignment mask.")
    if mask.shape[1] <= 0:
        raise ValueError("Cannot derive atom_to_residue_index for zero residues.")
    atom_mass = mask.sum(dim=1)
    if not pt.allclose(atom_mass, pt.ones_like(atom_mass), atol=1e-5, rtol=0.0):
        raise ValueError("Each atom must map to exactly one residue.")
    return mask.argmax(dim=1).long()


def _sanitize_surface_cache(cache):
    if not isinstance(cache, dict):
        return cache
    return {key: value for key, value in cache.items() if key != "proxy_surface_labels"}


def s1_static_geometry_cache_key_for_row(row, version="e89_dense_geometry_v1"):
    raw = "|".join([str(row.get("receptor_pdb_path", "")), str(version)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24] + ".pt"


def s1_static_geometry_cache_path_for_row(cache_dir, row, version="e89_dense_geometry_v1"):
    return Path(cache_dir) / s1_static_geometry_cache_key_for_row(row, version=version)


class S1SurfaceFieldDataset(MultigraphPlusManifestDataset):
    def __init__(
        self,
        *args,
        s1_surface_cache_dir=None,
        require_s1_surface_cache=False,
        build_s1_surface_on_the_fly=True,
        s1_surface_points=1024,
        s1_surface_graph_k=16,
        s1_surface_lappe_dim=0,
        s1_surface_cache_version="s1_surface_proxy_v1",
        teacher_cache_dir=None,
        teacher_name="",
        teacher_cache_version="teacher_prior_v1",
        require_teacher_cache=False,
        build_soft_residue_labels=False,
        soft_label_sigma=8.0,
        soft_label_max_value=0.7,
        soft_label_cutoff=24.0,
        s1_static_geometry_cache_dir=None,
        s1_static_geometry_cache_version="e89_dense_geometry_v1",
        require_s1_static_geometry_cache=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.s1_surface_cache_dir = Path(s1_surface_cache_dir) if s1_surface_cache_dir else None
        self.require_s1_surface_cache = bool(require_s1_surface_cache)
        self.build_s1_surface_on_the_fly = bool(build_s1_surface_on_the_fly)
        self.s1_surface_points = int(s1_surface_points)
        self.s1_surface_graph_k = int(s1_surface_graph_k)
        self.s1_surface_lappe_dim = int(s1_surface_lappe_dim)
        self.s1_surface_cache_version = str(s1_surface_cache_version)
        self.teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir else None
        self.teacher_name = str(teacher_name)
        self.teacher_cache_version = str(teacher_cache_version)
        self.require_teacher_cache = bool(require_teacher_cache)
        self.build_soft_residue_labels = bool(build_soft_residue_labels)
        self.soft_label_sigma = float(soft_label_sigma)
        self.soft_label_max_value = float(soft_label_max_value)
        self.soft_label_cutoff = float(soft_label_cutoff)
        self.s1_static_geometry_cache_dir = Path(s1_static_geometry_cache_dir) if s1_static_geometry_cache_dir else None
        self.s1_static_geometry_cache_version = str(s1_static_geometry_cache_version)
        self.require_s1_static_geometry_cache = bool(require_s1_static_geometry_cache)

    def _attach_static_geometry_cache(self, item, row):
        if item.get("skip") or self.s1_static_geometry_cache_dir is None:
            return item
        path = s1_static_geometry_cache_path_for_row(
            self.s1_static_geometry_cache_dir, row, version=self.s1_static_geometry_cache_version
        )
        if not path.is_file():
            if self.require_s1_static_geometry_cache:
                return {"skip": True, "skip_reason": f"Missing E89 static geometry cache: {path}", "rows": row}
            return item
        cache = pt.load(path, map_location="cpu")
        if int(cache.get("num_residues", -1)) != int(item["M"].shape[1]):
            return {"skip": True, "skip_reason": f"E89 static geometry residue count mismatch: {path}", "rows": row}
        item["s1_static_geometry_cache"] = cache
        return item

    def _attach_s1_surface_cache(self, item, row):
        if item.get("skip"):
            return item
        cache = None
        if self.s1_surface_cache_dir is not None:
            path = s1_surface_cache_path_for_row(
                self.s1_surface_cache_dir,
                row,
                num_surface_points=self.s1_surface_points,
                graph_k=self.s1_surface_graph_k,
                lappe_dim=self.s1_surface_lappe_dim,
                version=self.s1_surface_cache_version,
            )
            if path.is_file():
                cache = pt.load(path, map_location="cpu")
            elif self.s1_surface_lappe_dim > 0:
                base_path = s1_surface_cache_path_for_row(
                    self.s1_surface_cache_dir,
                    row,
                    num_surface_points=self.s1_surface_points,
                    graph_k=self.s1_surface_graph_k,
                    lappe_dim=0,
                    version=self.s1_surface_cache_version,
                )
                if base_path.is_file():
                    cache = pt.load(base_path, map_location="cpu")
            elif self.require_s1_surface_cache:
                return {
                    "skip": True,
                    "skip_reason": f"Missing S1 surface cache for apex_case_id={row.get('apex_case_id', '')}: {path}",
                    "rows": row,
                }
            if cache is None and self.require_s1_surface_cache:
                return {
                    "skip": True,
                    "skip_reason": f"Missing S1 surface cache for apex_case_id={row.get('apex_case_id', '')}: {path}",
                    "rows": row,
                }
        if cache is None and self.build_s1_surface_on_the_fly:
            cache = build_s1_surface_proxy_cache(
                item,
                num_surface_points=self.s1_surface_points,
                graph_k=self.s1_surface_graph_k,
                lappe_dim=self.s1_surface_lappe_dim,
            )
        if cache is not None:
            if int(cache.get("num_residues", item["M"].shape[1])) != int(item["M"].shape[1]):
                return {
                    "skip": True,
                    "skip_reason": f"S1 surface cache residue count mismatch for apex_case_id={row.get('apex_case_id', '')}",
                    "rows": row,
                }
            lappe = cache.get("surface_lappe")
            if self.s1_surface_lappe_dim > 0 and (
                not isinstance(lappe, pt.Tensor) or int(lappe.shape[-1]) < self.s1_surface_lappe_dim
            ):
                graph_idx = cache.get("surface_graph_idx")
                if isinstance(graph_idx, pt.Tensor):
                    cache["surface_lappe"] = _normalized_lappe_from_edges(
                        int(cache["surface_pos"].shape[0]),
                        graph_idx.long(),
                        self.s1_surface_lappe_dim,
                    )
            item["s1_surface_cache"] = cache
        return item

    def _attach_teacher_cache(self, item, row):
        if item.get("skip") or self.teacher_cache_dir is None or not self.teacher_name:
            return item
        path = teacher_cache_path_for_row(
            self.teacher_cache_dir,
            row,
            teacher_name=self.teacher_name,
            version=self.teacher_cache_version,
        )
        if not path.is_file():
            if self.require_teacher_cache:
                return {
                    "skip": True,
                    "skip_reason": f"Missing teacher cache for apex_case_id={row.get('apex_case_id', '')}: {path}",
                    "rows": row,
                }
            return item
        cache = pt.load(path, map_location="cpu")
        logits = cache.get("teacher_logits")
        n_res = int(item["M"].shape[1])
        if isinstance(logits, pt.Tensor) and int(logits.shape[0]) != n_res:
            if self.require_teacher_cache:
                return {
                    "skip": True,
                    "skip_reason": f"Teacher cache residue count mismatch for apex_case_id={row.get('apex_case_id', '')}: {path}",
                    "rows": row,
                }
            return item
        item["teacher_cache"] = cache
        return item

    def _attach_soft_residue_labels(self, item):
        if item.get("skip") or not self.build_soft_residue_labels:
            return item
        centers = residue_centers_from_atom_mask(item["X"].float(), item["M"].float())
        mask = item.get("mask", pt.ones_like(item["y"].float()))
        item["soft_y"] = residue_soft_labels(
            item["y"].float(),
            mask.float(),
            centers,
            sigma=self.soft_label_sigma,
            max_value=self.soft_label_max_value,
            cutoff=self.soft_label_cutoff,
        )
        return item

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = item.get("rows", self.rows[index])
        item = self._attach_s1_surface_cache(item, row)
        item = self._attach_teacher_cache(item, row)
        item = self._attach_soft_residue_labels(item)
        item = self._attach_static_geometry_cache(item, row)
        if not item.get("skip"):
            item["atom_to_residue_index"] = atom_to_residue_index_from_mask(item["M"])
            item["num_residues"] = int(item["M"].shape[1])
            if "s1_surface_cache" in item:
                item["s1_surface_cache"] = _sanitize_surface_cache(item["s1_surface_cache"])
        return item


def collate_s1_surface_field_batch(batch):
    skipped = [item for item in batch if item.get("skip")]
    kept = [item for item in batch if not item.get("skip")]
    if not kept:
        return {
            "skip_batch": True,
            "skipped_count": len(skipped),
            "skipped_rows": [item.get("rows", {}) for item in skipped],
            "skip_reasons": [item.get("skip_reason", "") for item in skipped],
        }

    atom_counts = pt.tensor([item["X"].shape[0] for item in kept], dtype=pt.long)
    residue_counts = pt.tensor([item["num_residues"] for item in kept], dtype=pt.long)
    residue_offsets = pt.cumsum(pt.cat([pt.zeros(1, dtype=pt.long), residue_counts[:-1]]), dim=0)
    atom_to_residue_index = pt.cat(
        [item["atom_to_residue_index"] + residue_offsets[idx] for idx, item in enumerate(kept)], dim=0
    )
    out = {
        "X": pt.cat([item["X"] for item in kept], dim=0),
        "q": pt.cat([item["q"] for item in kept], dim=0),
        "atom_to_residue_index": atom_to_residue_index,
        "atom_counts": atom_counts,
        "residue_counts": residue_counts,
        "y": pt.cat([item["y"] for item in kept], dim=0),
        "mask": pt.cat([item["mask"] for item in kept], dim=0),
        "residue_chain_ids": pt.cat(
            [item.get("residue_chain_ids", pt.zeros(item["num_residues"], dtype=pt.long)) for item in kept], dim=0
        ),
        "rows": [item["rows"] for item in kept],
        "skip_batch": False,
        "skipped_count": len(skipped),
        "skipped_rows": [item.get("rows", {}) for item in skipped],
        "skip_reasons": [item.get("skip_reason", "") for item in skipped],
    }
    if any("residue_graph_cache" in item for item in kept):
        if not all("residue_graph_cache" in item for item in kept):
            raise ValueError("Mixed batch with partial residue_graph_cache is not supported.")
        out["residue_graph_caches"] = [item["residue_graph_cache"] for item in kept]
    if any("true_residue_features" in item for item in kept):
        if not all("true_residue_features" in item for item in kept):
            raise ValueError("Mixed batch with partial true_residue_features is not supported.")
        out["true_residue_features"] = pt.cat([item["true_residue_features"] for item in kept], dim=0)
        out["true_residue_feature_names"] = kept[0].get("true_residue_feature_names", [])
    if any("s1_surface_cache" in item for item in kept):
        if not all("s1_surface_cache" in item for item in kept):
            raise ValueError("Mixed batch with partial s1_surface_cache is not supported.")
        out["s1_surface_caches"] = [item["s1_surface_cache"] for item in kept]
    if any("s1_static_geometry_cache" in item for item in kept):
        if not all("s1_static_geometry_cache" in item for item in kept):
            raise ValueError("Mixed batch with partial s1_static_geometry_cache is not supported.")
        out["s1_static_geometry_caches"] = [item["s1_static_geometry_cache"] for item in kept]
    if any("teacher_cache" in item for item in kept):
        if not all("teacher_cache" in item for item in kept):
            raise ValueError("Mixed batch with partial teacher_cache is not supported.")
        logits = [item["teacher_cache"].get("teacher_logits") for item in kept]
        if all(isinstance(x, pt.Tensor) for x in logits):
            out["teacher_logits"] = pt.cat([x.float().reshape(-1) for x in logits], dim=0)
        hidden = [item["teacher_cache"].get("teacher_hidden") for item in kept]
        if all(isinstance(x, pt.Tensor) for x in hidden):
            out["teacher_hidden"] = pt.cat([x.float() for x in hidden], dim=0)
    if any("soft_y" in item for item in kept):
        if not all("soft_y" in item for item in kept):
            raise ValueError("Mixed batch with partial soft_y is not supported.")
        out["soft_y"] = pt.cat([item["soft_y"].float().reshape(-1) for item in kept], dim=0)
    return out
