import sys
from pathlib import Path

import torch as pt


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"
for path in [str(ROOT_DIR), str(MODEL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from apex_dataset import ApexPairedManifestDataset, collate_apex_batch
from apex_surface_aux_gvp.surface_cache import build_surface_proxy_cache, surface_cache_path_for_row


class SurfaceAuxManifestDataset(ApexPairedManifestDataset):
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
    out = collate_apex_batch(batch)
    if out.get("skip_batch"):
        return out
    kept = [item for item in batch if not item.get("skip")]
    if any("surface_cache" in item for item in kept):
        if not all("surface_cache" in item for item in kept):
            raise ValueError("Mixed batch with partial surface_cache is not supported.")
        out["surface_caches"] = [item["surface_cache"] for item in kept]
    return out
