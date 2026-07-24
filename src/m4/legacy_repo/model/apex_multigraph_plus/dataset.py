import hashlib
import sys
from pathlib import Path

import torch as pt


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_dataset import cache_key_for_row
from apex_surface_aux_gvp.dataset import SurfaceAuxManifestDataset, collate_surface_aux_batch


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
