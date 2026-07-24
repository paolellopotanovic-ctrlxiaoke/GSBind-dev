"""LightningDataModule backed by the exact M4 dataset and cache contract."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import lightning as L

from .._bootstrap import activate_validated_implementation

activate_validated_implementation()

from apex_dataset import read_manifest  # noqa: E402
from apex_s1_surface_field.train_s1_surface_field import build_loader  # noqa: E402
from train_apex import estimate_pos_weight, split_rows_grouped  # noqa: E402


@dataclass(frozen=True)
class M4DataConfig:
    manifest: str
    tensor_cache_dir: str
    residue_graph_cache_dir: str
    s1_surface_cache_dir: str
    s1_static_geometry_cache_dir: str
    fixed_train_manifest: str = ""
    fixed_val_manifest: str = ""
    states: str = "holo"
    max_cases: int = 0
    val_fraction: float = 0.1
    val_split_group: str = "cluster_id"
    seed: int = 42
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 1
    allow_residue_mismatch: bool = False
    require_tensor_cache: bool = True
    require_residue_graph_cache: bool = True
    residue_graph_cache_ks: tuple[int, ...] = (16, 24, 32)
    true_feature_cache_dir: str = ""
    require_true_features: bool = False
    true_feature_version: str = "true_sasa_dssp_v1"
    require_s1_surface_cache: bool = True
    disable_s1_surface_on_the_fly: bool = True
    s1_surface_points: int = 1024
    s1_surface_graph_k: int = 16
    s1_surface_lappe_dim: int = 0
    s1_surface_cache_version: str = "s1_surface_shrake_v1"
    teacher_cache_dir: str = ""
    teacher_name: str = ""
    teacher_cache_version: str = "teacher_prior_v1"
    require_teacher_cache: bool = False
    use_soft_residue_labels: bool = False
    soft_label_sigma: float = 8.0
    soft_label_max_value: float = 0.7
    soft_label_cutoff: float = 24.0
    s1_static_geometry_cache_version: str = "e89_dense_geometry_v1"
    require_s1_static_geometry_cache: bool = True
    pos_weight_factor: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "M4DataConfig":
        known = {field.name for field in fields(cls)}
        parsed = {key: value for key, value in values.items() if key in known}
        if isinstance(parsed.get("residue_graph_cache_ks"), list):
            parsed["residue_graph_cache_ks"] = tuple(parsed["residue_graph_cache_ks"])
        return cls(**parsed)

    def loader_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


class M4DataModule(L.LightningDataModule):
    """Own manifest filtering, grouped splitting, and four-cache loaders."""

    def __init__(self, config: M4DataConfig):
        super().__init__()
        self.config = config
        self.train_rows: list[dict[str, Any]] = []
        self.val_rows: list[dict[str, Any]] = []
        self.label_stats: dict[str, Any] = {}
        self.pos_weight: float | None = None

    def prepare_data(self) -> None:
        for path in filter(None, (self.config.manifest, self.config.fixed_train_manifest, self.config.fixed_val_manifest)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"M4 manifest not found: {path}")
        for path in (
            self.config.tensor_cache_dir,
            self.config.residue_graph_cache_dir,
            self.config.s1_surface_cache_dir,
            self.config.s1_static_geometry_cache_dir,
        ):
            if not Path(path).is_dir():
                raise FileNotFoundError(f"M4 cache directory not found: {path}")

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate", "test", "predict"):
            return
        states = {value.strip() for value in self.config.states.split(",") if value.strip()}
        if self.config.fixed_train_manifest:
            train_rows = read_manifest(self.config.fixed_train_manifest)
            val_rows = read_manifest(self.config.fixed_val_manifest) if self.config.fixed_val_manifest else []
            self.train_rows = [row for row in train_rows if row.get("input_state") in states]
            self.val_rows = [row for row in val_rows if row.get("input_state") in states]
        else:
            rows = [row for row in read_manifest(self.config.manifest) if row.get("input_state") in states]
            if self.config.max_cases:
                rows = rows[: self.config.max_cases]
            self.train_rows, self.val_rows = split_rows_grouped(
                rows,
                val_fraction=self.config.val_fraction,
                seed=self.config.seed,
                group_key=self.config.val_split_group,
            )
        if not self.train_rows:
            raise RuntimeError("No rows left in the M4 training manifest after state filtering")
        self.pos_weight, self.label_stats = estimate_pos_weight(
            self.train_rows,
            factor=self.config.pos_weight_factor,
        )

    def train_dataloader(self):
        return build_loader(self.train_rows, self.config.loader_namespace(), shuffle=True)

    def val_dataloader(self):
        if not self.val_rows:
            return []
        return build_loader(self.val_rows, self.config.loader_namespace(), shuffle=False)
