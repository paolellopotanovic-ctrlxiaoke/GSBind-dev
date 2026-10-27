#!/usr/bin/env python3
"""Rebuild a training/test split directly from the public PINDER 2024-02 API."""

from __future__ import annotations

import hashlib
import shutil
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SUPPORTED_RELEASE = "2024-02"
TRAIN_SPLITS = ("train", "val", "invalid")
MANIFEST_COLUMNS = [
    "apex_case_id",
    "system_id",
    "cluster_id",
    "receptor_uniprot",
    "input_state",
    "receptor_side",
    "receptor_chain_id",
    "labels",
    "label_mask",
    "input_residue_ids",
    "mapped_uniprot_positions",
    "sequence",
    "num_residues",
    "receptor_pdb_relpath",
    "label_contract",
    "receptor_pdb_path",
    "compact_path",
]
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass(frozen=True)
class PipelineOptions:
    output_dir: Path
    release: str = SUPPORTED_RELEASE
    min_length: int = 40
    max_length: int = 1000
    min_positives: int = 3
    min_mask_coverage: float = 0.95
    max_positive_rate: float = 0.75
    cluster_cap: int = 5
    contact_cutoff: float = 5.0
    workers: int = 8
    download_workers: int = 8
    chunk_size: int = 512
    limit: int = 0
    anchor_manifest: Path | None = None
    update_index: bool = False


class PinderApiError(RuntimeError):
    pass


class PinderApiAdapter:
    def __init__(self, release: str = SUPPORTED_RELEASE, cache_dir: Path | None = None) -> None:
        if release != SUPPORTED_RELEASE:
            raise PinderApiError(f"PINDER_RELEASE={release} is unsupported; use {SUPPORTED_RELEASE}")
        if cache_dir is not None:
            os.environ["PINDER_DATA_DIR"] = str(cache_dir)
        os.environ["PINDER_RELEASE"] = release
        self.release = release
        self._mapping_stems: set[str] | None = None

    def _pinder(self):
        try:
            import pinder
            from pinder.core.index import utils as pinder_utils
            from pinder.core.utils import dataset
        except ImportError as exc:
            raise PinderApiError(
                "The official pinder package is required. Install it with: "
                "pip install -e '.[pinder]'"
            ) from exc
        return pinder, pinder_utils, dataset

    def load_index(self, update: bool = False) -> pd.DataFrame:
        _, pinder_utils, _ = self._pinder()
        frame = pinder_utils.get_index(update=update)
        required = {
            "id", "split", "cluster_id_R", "cluster_id_L", "uniprot_R",
            "uniprot_L", "holo_R_pdb", "holo_L_pdb",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise PinderApiError(f"PINDER index lacks columns: {sorted(missing)}")
        return frame.copy()

    @property
    def cache_dir(self) -> Path:
        _, pinder_utils, _ = self._pinder()
        return pinder_utils.get_pinder_location()

    def download_native(self, system_ids: Iterable[str], workers: int = 8) -> None:
        pinder, _, dataset = self._pinder()
        root = pinder.core.index.utils.get_pinder_bucket_root()
        sources: list[str] = []
        destinations: list[Path] = []
        cache = self.cache_dir
        for system_id in system_ids:
            destination = cache / "pdbs" / f"{system_id}.pdb"
            if not destination.is_file():
                sources.append(f"{root}/pdbs/{system_id}.pdb")
                destinations.append(destination)
        if sources:
            dataset._DOWNLOAD_WORKERS = max(1, workers)
            dataset.download_files(sources, destinations)

    def native_path(self, system_id: str) -> Path:
        return self.cache_dir / "pdbs" / f"{system_id}.pdb"

    def holo_path(self, holo_name: str) -> Path:
        return self.cache_dir / "pdbs" / holo_name

    def mapping_path(self, holo_name: str) -> Path:
        return self.cache_dir / "mappings" / f"{Path(holo_name).stem}.parquet"

    def _download_release_files(self, paths: list[str], workers: int = 8) -> None:
        pinder, _, dataset = self._pinder()
        root = pinder.core.index.utils.get_pinder_bucket_root()
        sources = [f"{root}/{path}" for path in paths]
        destinations = [self.cache_dir / path for path in paths]
        dataset._DOWNLOAD_WORKERS = max(1, workers)
        dataset.download_files(sources, destinations)

    def download_holo_pdbs(self, holo_names: Iterable[str], workers: int = 8) -> None:
        missing = [
            name for name in holo_names
            if not self.holo_path(name).is_file()
        ]
        if missing:
            self._download_release_files(
                [f"pdbs/{name}" for name in missing], workers=workers
            )

    def download_mappings(self, holo_names: Iterable[str], workers: int = 8) -> None:
        if self._mapping_stems is None:
            self._mapping_stems = {path.stem for path in (self.cache_dir / "mappings").iterdir()}
        missing = [
            name for name in holo_names
            if Path(name).stem not in self._mapping_stems
        ]
        if missing:
            self._download_release_files(
                [f"mappings/{Path(name).stem}.parquet" for name in missing],
                workers=workers,
            )
            self._mapping_stems.update(Path(name).stem for name in missing)

    def load_historical_metadata(
        self, system_ids: Iterable[str]
    ) -> pd.DataFrame:
        ids = sorted({str(system_id) for system_id in system_ids})
        files = ("metadata.parquet", "supplementary_metadata.parquet", "chain_metadata.parquet")
        self._download_release_files(list(files), workers=1)
        cache = self.cache_dir
        metadata = pd.read_parquet(
            cache / "metadata.parquet",
            columns=["id", "resolution", "length_resolved_1", "length_resolved_2"],
            filters=[("id", "in", ids)],
        )
        supplementary = pd.read_parquet(
            cache / "supplementary_metadata.parquet",
            columns=["id", "chain_1_residues", "chain_2_residues"],
            filters=[("id", "in", ids)],
        )
        chains = pd.read_parquet(
            cache / "chain_metadata.parquet",
            columns=[
                "id", "resi_R", "resi_uniprot_R", "resi_L", "resi_uniprot_L",
            ],
            filters=[("id", "in", ids)],
        )
        source = supplementary.merge(chains, on="id", validate="one_to_one").merge(
            metadata, on="id", validate="one_to_one"
        )
        records: list[dict[str, object]] = []
        for row in source.to_dict(orient="records"):
            for side, chain_index in (("R", 1), ("L", 2)):
                interface = [
                    value.strip() for value in str(row.get(f"chain_{chain_index}_residues") or "").split(",")
                    if value.strip()
                ]
                native_ids = [
                    value.strip() for value in str(row.get(f"resi_{side}") or "").split(",")
                    if value.strip()
                ]
                uniprot_ids = str(row.get(f"resi_uniprot_{side}") or "").split(",")
                mapped_native_ids = {
                    native_id for native_id, uniprot_id in zip(native_ids, uniprot_ids)
                    if str(uniprot_id).strip()
                }
                try:
                    length_value = float(row.get(f"length_resolved_{chain_index}"))
                except (TypeError, ValueError):
                    length_value = float("nan")
                try:
                    resolution_value = float(row.get("resolution"))
                except (TypeError, ValueError):
                    resolution_value = float("nan")
                coverage = (
                    sum(native_id in mapped_native_ids for native_id in interface)
                    / max(len(interface), 1)
                )
                positive_count = len(interface)
                records.append({
                    "system_id": str(row["id"]),
                    "receptor_side": side,
                    "metadata_positive_count": positive_count,
                    "interface_mapping_coverage": coverage,
                    "receptor_length_audit": length_value,
                    "metadata_resolution": resolution_value,
                    "quality_strict_core_metadata": bool(
                        np.isfinite(length_value)
                        and 40 <= length_value <= 1000
                        and positive_count >= 3
                        and positive_count <= length_value
                        and coverage >= 0.95
                        and (not np.isfinite(resolution_value) or resolution_value <= 4.0)
                    ),
                })
        return pd.DataFrame(records)

    def describe(self) -> dict[str, object]:
        pinder, _, _ = self._pinder()
        return {
            "package": "pinder.core.get_index",
            "pinder_version": getattr(pinder, "__version__", "unknown"),
            "release": self.release,
            "cache_dir": str(self.cache_dir),
        }


def strict_cluster(value: object) -> str:
    cluster = str(value or "")
    return "cluster_p" if cluster.startswith("seq:") else cluster


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def official_test_side_clusters(index: pd.DataFrame) -> set[str]:
    test = index.loc[index["split"].eq("test"), ["cluster_id_R", "cluster_id_L"]]
    return set(test.values.ravel().tolist())


def select_candidate_and_test_systems(
    index: pd.DataFrame, limit: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    frame = index.copy()
    for column in ["id", "split", "cluster_id_R", "cluster_id_L"]:
        frame[column] = frame[column].astype(str)
    test = frame.loc[frame["split"].eq("test")].sort_values("id").copy()
    test_sides = official_test_side_clusters(frame)
    training = frame.loc[
        frame["split"].isin(TRAIN_SPLITS)
        & ~frame["cluster_id_R"].isin(test_sides)
        & ~frame["cluster_id_L"].isin(test_sides)
    ].sort_values("id").copy()
    if limit > 0:
        training = training.head(limit).copy()
        test = test.head(limit).copy()
    audit = {
        "index_systems": int(len(frame)),
        "official_test_side_clusters": int(len(test_sides)),
        "training_source_systems": int(len(training)),
        "official_test_systems": int(len(test)),
        **{f"training_source_{split}": int((training["split"] == split).sum()) for split in TRAIN_SPLITS},
    }
    return training.reset_index(drop=True), test.reset_index(drop=True), audit


def load_anchor_cases(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "holo_pdb" not in frame:
        if "receptor_pdb_relpath" in frame:
            frame = frame.assign(holo_pdb=frame["receptor_pdb_relpath"].map(lambda value: Path(str(value)).name))
        elif "apex_case_id" in frame:
            frame = frame.assign(holo_pdb=frame["apex_case_id"].astype(str).str.removeprefix("pinder_full_holo::"))
        else:
            raise ValueError("anchor/test target manifest lacks holo_pdb or receptor_pdb_relpath")
    required = {"holo_pdb", "receptor_side", "cluster_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"anchor/test target manifest lacks columns: {sorted(missing)}")
    return frame[["holo_pdb", "receptor_side", "cluster_id"]].copy()


def _directed_index_events(index: pd.DataFrame) -> pd.DataFrame:
    receptor = index.rename(columns={
        "id": "system_id", "cluster_id_R": "official_cluster_id", "holo_R_pdb": "receptor_holo_pdb",
    })[["system_id", "split", "official_cluster_id", "receptor_holo_pdb"]].assign(receptor_side="R")
    partner = index.rename(columns={
        "id": "system_id", "cluster_id_L": "official_cluster_id", "holo_L_pdb": "receptor_holo_pdb",
    })[["system_id", "split", "official_cluster_id", "receptor_holo_pdb"]].assign(receptor_side="L")
    return pd.concat([receptor, partner], ignore_index=True)


def select_anchor_source_systems(
    index: pd.DataFrame, anchor_path: Path
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], dict[str, object]]:
    anchor = load_anchor_cases(anchor_path)
    anchor["holo_pdb"] = anchor["holo_pdb"].astype(str)
    anchor["receptor_side"] = anchor["receptor_side"].astype(str)
    if anchor.duplicated(["holo_pdb", "receptor_side"]).any():
        raise ValueError("anchor target contains duplicate holo/side rows")
    events = _directed_index_events(index)
    target = set(zip(anchor["holo_pdb"], anchor["receptor_side"]))
    mapped = events.loc[
        [key in target for key in zip(events["receptor_holo_pdb"], events["receptor_side"])], "system_id"
    ].astype(str)
    matched = events.loc[
        [key in target for key in zip(events["receptor_holo_pdb"], events["receptor_side"])]
    ].copy()
    missing = target.difference(zip(matched["receptor_holo_pdb"], matched["receptor_side"]))
    if missing:
        raise RuntimeError(f"{len(missing)} anchor holo/side targets are absent from the official index")

    index_by_id = index.assign(id=index["id"].astype(str)).set_index("id", drop=False)
    matched["cluster_id_R"] = matched["system_id"].map(index_by_id["cluster_id_R"])
    matched["cluster_id_L"] = matched["system_id"].map(index_by_id["cluster_id_L"])
    test_sides = official_test_side_clusters(index)
    leak = matched["cluster_id_R"].isin(test_sides) | matched["cluster_id_L"].isin(test_sides)
    clean = matched.loc[~leak].copy()
    still_missing = target.difference(zip(clean["receptor_holo_pdb"], clean["receptor_side"]))
    if still_missing:
        raise RuntimeError(
            f"{len(still_missing)} anchor targets have no non-test-side-leaking official source event"
        )
    system_ids = set(clean["system_id"].astype(str))
    systems = index.loc[index["id"].astype(str).isin(system_ids)].copy()
    systems = systems.sort_values("id").reset_index(drop=True)
    cluster_map = {
        (str(row.holo_pdb), str(row.receptor_side)): str(row.cluster_id)
        for row in anchor.itertuples(index=False)
    }
    audit = {
        "anchor_rows": int(len(anchor)),
        "anchor_holo_sides": int(len(target)),
        "mapped_directed_events": int(len(matched)),
        "test_side_leaking_directed_events_excluded": int(leak.sum()),
        "official_source_systems": int(len(systems)),
        "official_source_split_counts": {
            str(split): int(count)
            for split, count in systems["split"].value_counts().sort_index().items()
        },
    }
    return systems, cluster_map, audit


def apply_anchor_clusters(events: pd.DataFrame, cluster_map: dict[tuple[str, str], str]) -> pd.DataFrame:
    result = events.copy()
    keys = list(zip(result["receptor_holo_pdb"].astype(str), result["receptor_side"].astype(str)))
    result["cluster_id"] = [cluster_map.get(key, value) for key, value in zip(keys, result["cluster_id"])]
    result = result.loc[[key in cluster_map for key in keys]].copy()
    target_keys = (
        result["receptor_holo_pdb"].astype(str) + "\0" + result["receptor_side"].astype(str)
    )
    train_priority_keys = set(target_keys.loc[
        result["official_split"].eq("train")
        & pd.to_numeric(result["coordinate_event_trainable"], errors="coerce").fillna(0).astype(bool)
    ])
    return result.loc[
        ~target_keys.isin(train_priority_keys) | result["official_split"].eq("train")
    ].copy()


def historical_mapping_pool(
    index: pd.DataFrame, historical_metadata: pd.DataFrame
) -> tuple[set[str], dict[str, int]]:
    test_sides = official_test_side_clusters(index)
    non_test = index.loc[
        index["split"].isin(("val", "invalid"))
        & ~index["cluster_id_R"].isin(test_sides)
        & ~index["cluster_id_L"].isin(test_sides),
        ["id", "holo_R_pdb", "holo_L_pdb"],
    ].copy()
    strict = historical_metadata.loc[
        pd.to_numeric(
            historical_metadata["quality_strict_core_metadata"], errors="coerce"
        ).fillna(0).astype(bool)
    ]
    strict_sides = set(zip(
        strict["system_id"].astype(str), strict["receptor_side"].astype(str)
    ))
    selected = non_test.loc[
        [
            any((str(row.id), side) in strict_sides for side in ("R", "L"))
            for row in non_test.itertuples(index=False)
        ]
    ].copy()
    mappings = {
        Path(str(row[f"holo_{side}_pdb"])).stem
        for row in selected.to_dict(orient="records")
        for side in ("R", "L")
        if (str(row["id"]), side) in strict_sides
    }
    strict_directed_events = sum(
        (str(row.id), side) in strict_sides
        for row in selected.itertuples(index=False)
        for side in ("R", "L")
    )
    audit = {
        "historical_mapping_source_systems": int(len(selected)),
        "historical_mapping_strict_directed_events": int(strict_directed_events),
        "historical_mapping_holos": int(len(mappings)),
    }
    return mappings, audit


_WORKER_CACHE_DIR: Path | None = None
_WORKER_CUTOFF = 5.0


def _chain_profile(model: gemmi.Model, chain_name: str) -> list[dict[str, object]] | None:
    chains = [chain for chain in model if chain.name == chain_name]
    if len(chains) != 1:
        return None
    residues: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for residue in chains[0]:
        residue_name = residue.name.upper()
        if residue_name not in AA3_TO_1:
            continue
        key = (str(residue.seqid.num), residue.seqid.icode.strip())
        if key in seen:
            continue
        seen.add(key)
        coordinates = [
            np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
            for atom in residue
            if atom.element.name != "H"
        ]
        if not coordinates:
            continue
        residues.append({
            "residue_id": f"{key[0]}:{key[1]}".rstrip(":"),
            "resname": residue_name,
            "coordinates": coordinates,
        })
    return residues or None


def _contact_labels(
    receptor: Sequence[dict[str, object]],
    partner: Sequence[dict[str, object]],
    cutoff: float,
) -> np.ndarray:
    receptor_atoms = np.concatenate([item["coordinates"] for item in receptor])
    partner_atoms = np.concatenate([item["coordinates"] for item in partner])
    partners = cKDTree(partner_atoms)
    nearby = partners.query_ball_point(receptor_atoms, r=cutoff, return_length=True)
    labels = np.zeros(len(receptor), dtype=np.int8)
    offset = 0
    for index, residue in enumerate(receptor):
        count = len(residue["coordinates"])
        labels[index] = int(nearby[offset:offset + count].max(initial=0) > 0)
        offset += count
    return labels


def _atom_is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    atom_name = line[12:16].strip().upper()
    if element:
        return element in {"H", "D"}
    stripped = atom_name.lstrip("0123456789")
    return stripped.startswith("H") or stripped.startswith("D")


def _parse_historical_pdb(path: Path) -> dict[str, object]:
    residues: list[dict[str, object]] = []
    residue_index: dict[tuple[str, int, str], int] = {}
    atom_xyz: list[np.ndarray] = []
    atom_residue_index: list[int] = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")) or len(line) < 54:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3_TO_1:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A", "1"} or _atom_is_hydrogen(line):
            continue
        chain = line[21:22].strip() or "_"
        try:
            resseq = int(line[22:26])
            xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
        except ValueError:
            continue
        if not np.isfinite(xyz).all():
            continue
        icode = line[26:27].strip()
        key = (chain, resseq, icode)
        index = residue_index.get(key)
        if index is None:
            index = len(residues)
            residue_index[key] = index
            residues.append({
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "uid": f"{resseq}{icode}" if icode else str(resseq),
                "aa": AA3_TO_1[resname],
            })
        atom_xyz.append(xyz)
        atom_residue_index.append(index)
    return {
        "residues": residues,
        "xyz": np.asarray(atom_xyz, dtype=np.float32).reshape((-1, 3)),
        "atom_residue_index": np.asarray(atom_residue_index, dtype=np.int32),
    }


def _mapping_from_parquet(path: Path) -> dict[int, int]:
    table = pd.read_parquet(path)
    if "resi_uniprot" not in table.columns:
        raise ValueError(f"official mapping lacks resi_uniprot: {path}")
    mapping: dict[int, int] = {}
    for row in table.to_dict(orient="records"):
        try:
            uniprot_position = int(float(row.get("resi_uniprot")))
        except (TypeError, ValueError):
            continue
        for field in ("resi_pdb", "resi_auth"):
            try:
                mapping[int(float(row.get(field)))] = uniprot_position
            except (TypeError, ValueError):
                continue
    return mapping


def _native_contact_residues(
    native: dict[str, object], side: str, partner_side: str, cutoff: float
) -> set[int]:
    residues = native["residues"]
    atom_residue_index = native["atom_residue_index"]
    atom_sides = np.asarray(
        [residues[index]["chain"] for index in atom_residue_index], dtype=object
    )
    side_indices = np.flatnonzero(atom_sides == side)
    partner_indices = np.flatnonzero(atom_sides == partner_side)
    if side_indices.size == 0 or partner_indices.size == 0:
        raise ValueError(f"native dimer missing chains {side}/{partner_side}")
    neighbors = cKDTree(native["xyz"][partner_indices]).query_ball_point(
        native["xyz"][side_indices], r=cutoff
    )
    return {
        int(atom_residue_index[side_indices[index]])
        for index, values in enumerate(neighbors)
        if values
    }


def _process_historical_event_task(payload: dict[str, object]) -> list[dict[str, object]]:
    global _WORKER_CACHE_DIR, _WORKER_CUTOFF
    assert _WORKER_CACHE_DIR is not None
    system_id = str(payload["system_id"])
    try:
        native = _parse_historical_pdb(_WORKER_CACHE_DIR / "pdbs" / f"{system_id}.pdb")
        rows: list[dict[str, object]] = []
        for side, partner_side in (("R", "L"), ("L", "R")):
            holo_name = str(payload[f"holo_{side}_pdb"])
            key = (holo_name, side)
            if key not in set(payload["_target_keys"]):
                continue
            receptor = _parse_historical_pdb(_WORKER_CACHE_DIR / "pdbs" / holo_name)
            partner_name = str(payload[f"holo_{partner_side}_pdb"])
            if (
                str(payload.get("split")) != "train"
                and not bool(payload.get(f"{side}_partner_mapping_available", False))
            ):
                raise FileNotFoundError(
                    _WORKER_CACHE_DIR / "mappings" / f"{Path(partner_name).stem}.parquet"
                )
            mapping = _mapping_from_parquet(
                _WORKER_CACHE_DIR / "mappings" / f"{Path(holo_name).stem}.parquet"
            )
            native_positive_indices = _native_contact_residues(
                native, side, partner_side, _WORKER_CUTOFF
            )
            native_positive_uniprot = {
                mapping[native["residues"][index]["resseq"]]
                for index in native_positive_indices
                if native["residues"][index]["resseq"] in mapping
            }
            labels: list[int] = []
            masks: list[int] = []
            residue_ids: list[str] = []
            sequence_parts: list[str] = []
            for residue in receptor["residues"]:
                position = mapping.get(residue["resseq"])
                labels.append(int(position is not None and position in native_positive_uniprot))
                masks.append(int(position is not None))
                residue_ids.append(str(residue["uid"]))
                sequence_parts.append(str(residue["aa"]))
            sequence = "".join(sequence_parts)
            valid_count = sum(masks)
            positive_count = sum(labels)
            coverage = valid_count / max(len(labels), 1)
            rows.append({
                "system_id": system_id,
                "official_split": str(payload["split"]),
                "receptor_side": side,
                "partner_side": partner_side,
                "receptor_holo_pdb": holo_name,
                "partner_holo_pdb": partner_name,
                "cluster_id": strict_cluster(payload[f"cluster_id_{side}"]),
                "official_cluster_id": str(payload[f"cluster_id_{side}"]),
                "receptor_uniprot": str(payload[f"uniprot_{side}"] or ""),
                "sequence": sequence,
                "sequence_identity": hashlib.sha1(sequence.encode("ascii", "ignore")).hexdigest(),
                "num_residues": len(labels),
                "num_positive_labels": positive_count,
                "mask_coverage": coverage,
                "low_mapping_quality": int(coverage < 0.95),
                "coordinate_union_status": "ok",
                "holo_event_count": 1,
                "holo_event_success_count": 1,
                "holo_event_failure_count": 0,
                "labels": ";".join(str(value) for value in labels),
                "label_mask": ";".join(str(value) for value in masks),
                "input_residue_ids": ";".join(residue_ids),
                "coordinate_event_trainable": bool(
                    (
                        bool(payload[f"{side}_quality_strict_core_metadata"])
                        if str(payload.get("split")) == "train"
                        else bool(payload.get("historical_system_strict_core_metadata"))
                    )
                    and 40 <= len(labels) <= 1000
                    and positive_count >= 3
                    and coverage >= 0.95
                ),
                "selection_origin": "official_api_holo_mapping_union",
            })
        return rows
    except Exception as exc:
        return [{
            "system_id": system_id,
            "official_split": str(payload.get("split", "")),
            "receptor_holo_pdb": str(payload.get("holo_R_pdb", "")),
            "coordinate_union_status": f"error:{type(exc).__name__}:{str(exc)[:180]}",
        }]


def _process_system_task(payload: dict[str, object]) -> list[dict[str, object]]:
    global _WORKER_CACHE_DIR, _WORKER_CUTOFF
    assert _WORKER_CACHE_DIR is not None
    system_id = str(payload["system_id"])
    path = _WORKER_CACHE_DIR / "pdbs" / f"{system_id}.pdb"
    try:
        structure = gemmi.read_structure(str(path))
        structure.remove_alternative_conformations()
        structure.remove_hydrogens()
        model = structure[0]
        receptor = _chain_profile(model, "R")
        partner = _chain_profile(model, "L")
        if receptor is None or partner is None:
            raise ValueError("native PDB does not contain exactly one nonempty R and L protein chain")
        labels = _contact_labels(receptor, partner, _WORKER_CUTOFF)
        rows: list[dict[str, object]] = []
        partner_labels = _contact_labels(partner, receptor, _WORKER_CUTOFF)
        for side, profile, profile_labels, cluster_column, uniprot_column in (
            ("R", receptor, labels, "cluster_id_R", "uniprot_R"),
            ("L", partner, partner_labels, "cluster_id_L", "uniprot_L"),
        ):
            sequence = "".join(AA3_TO_1[str(item["resname"])] for item in profile)
            rows.append({
                "system_id": system_id,
                "official_split": str(payload["split"]),
                "receptor_side": side,
                "partner_side": "L" if side == "R" else "R",
                "receptor_holo_pdb": str(payload[f"holo_{side}_pdb"]),
                "partner_holo_pdb": str(payload[f"holo_{'L' if side == 'R' else 'R'}_pdb"]),
                "cluster_id": strict_cluster(payload[cluster_column]),
                "official_cluster_id": str(payload[cluster_column]),
                "receptor_uniprot": str(payload[uniprot_column] or ""),
                "sequence": sequence,
                "sequence_identity": hashlib.sha1(sequence.encode("ascii", "ignore")).hexdigest(),
                "num_residues": len(profile),
                "num_positive_labels": int(profile_labels.sum()),
                "mask_coverage": 1.0,
                "low_mapping_quality": 0,
                "coordinate_union_status": "ok",
                "holo_event_count": 1,
                "holo_event_success_count": 1,
                "holo_event_failure_count": 0,
                "labels": ";".join(str(int(value)) for value in profile_labels),
                "label_mask": ";".join(["1"] * len(profile)),
                "input_residue_ids": ";".join(str(index + 1) for index in range(len(profile))),
                "coordinate_event_trainable": True,
                "selection_origin": "official_api_direct_holo_union",
            })
        return rows
    except Exception as exc:
        return [{
            "system_id": system_id,
            "official_split": str(payload["split"]),
            "receptor_holo_pdb": str(payload.get("holo_R_pdb", "")),
            "coordinate_union_status": f"error:{type(exc).__name__}:{str(exc)[:180]}",
        }]


def _init_worker(cache_dir: Path, cutoff: float) -> None:
    global _WORKER_CACHE_DIR, _WORKER_CUTOFF
    _WORKER_CACHE_DIR = cache_dir
    _WORKER_CUTOFF = cutoff


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _events_to_frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "system_id", "official_split", "receptor_side", "partner_side",
        "receptor_holo_pdb", "partner_holo_pdb", "cluster_id",
        "official_cluster_id", "receptor_uniprot", "sequence",
        "sequence_identity", "num_residues", "num_positive_labels",
        "mask_coverage", "low_mapping_quality", "coordinate_union_status",
        "holo_event_count", "holo_event_success_count",
        "holo_event_failure_count", "labels", "label_mask",
        "input_residue_ids", "selection_origin",
        "coordinate_event_trainable",
    ]
    return pd.DataFrame(rows, columns=columns)


def process_systems(
    systems: pd.DataFrame,
    adapter: PinderApiAdapter,
    options: PipelineOptions,
    shard_dir: Path,
    target_keys: set[tuple[str, str]] | None = None,
    historical_metadata_frame: pd.DataFrame | None = None,
    historical_mapping_holos: set[str] | None = None,
) -> pd.DataFrame:
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = adapter.cache_dir
    rows: list[dict[str, object]] = []
    total = len(systems)
    metadata_cache = shard_dir / "historical_metadata.parquet"
    if target_keys is not None and historical_metadata_frame is None:
        if metadata_cache.is_file():
            historical_metadata_frame = pd.read_parquet(metadata_cache)
        else:
            historical_metadata_frame = adapter.load_historical_metadata(
                systems["id"].astype(str)
            )
            temporary = metadata_cache.with_suffix(".parquet.partial")
            historical_metadata_frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(metadata_cache)
    if target_keys is not None and metadata_cache is not None and not metadata_cache.is_file():
        temporary = metadata_cache.with_suffix(".parquet.partial")
        historical_metadata_frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(metadata_cache)
    historical_metadata = (
        {
            (str(row["system_id"]), str(row["receptor_side"])): {
                key: value
                for key, value in row.items()
                if key not in {"system_id", "receptor_side"}
            }
            for row in historical_metadata_frame.to_dict(orient="records")
        }
        if historical_metadata_frame is not None else None
    )
    executor = None
    try:
        if options.workers > 1 and total:
            executor = ProcessPoolExecutor(
                max_workers=options.workers,
                initializer=_init_worker,
                initargs=(cache_dir, options.contact_cutoff),
            )
        for start in range(0, total, options.chunk_size):
            chunk = systems.iloc[start:start + options.chunk_size]
            shard = shard_dir / f"events_{start:012d}.parquet"
            if not shard.is_file():
                adapter.download_native(
                    chunk["id"].astype(str).tolist(), workers=options.download_workers
                )
                payloads = chunk.rename(columns={"id": "system_id"}).to_dict(orient="records")
                if historical_metadata is not None:
                    for payload in payloads:
                        strict_sides = {
                            side for side in ("R", "L")
                            if bool(historical_metadata.get(
                                (str(payload["system_id"]), side), {}
                            ).get("quality_strict_core_metadata", False))
                        }
                        payload["historical_system_strict_core_metadata"] = bool(strict_sides)
                        for side in ("R", "L"):
                            metadata = historical_metadata.get((str(payload["system_id"]), side))
                            if metadata is not None:
                                payload.update({
                                    f"{side}_{key}": value
                                    for key, value in metadata.items()
                                })
                if target_keys is not None:
                    required_holos = {
                        holo_name
                        for payload in payloads
                        for holo_name, side in (
                            (str(payload["holo_R_pdb"]), "R"),
                            (str(payload["holo_L_pdb"]), "L"),
                        )
                        if (holo_name, side) in target_keys
                    }
                    required_mappings = set(required_holos)
                    adapter.download_holo_pdbs(required_holos, workers=options.download_workers)
                    adapter.download_mappings(required_mappings, workers=options.download_workers)
                    mapping_stems = (
                        historical_mapping_holos
                        if historical_mapping_holos is not None
                        else (adapter._mapping_stems or set())
                    )
                    for payload in payloads:
                        payload["_target_keys"] = target_keys
                        for side, partner_side in (("R", "L"), ("L", "R")):
                            if (str(payload[f"holo_{side}_pdb"]), side) in target_keys:
                                payload[f"{side}_partner_mapping_available"] = (
                                    Path(str(payload[f"holo_{partner_side}_pdb"])).stem in mapping_stems
                                )
                if executor is None:
                    _init_worker(cache_dir, options.contact_cutoff)
                    task = _process_historical_event_task if target_keys is not None else _process_system_task
                    chunk_rows = [row for payload in payloads for row in task(payload)]
                else:
                    task = _process_historical_event_task if target_keys is not None else _process_system_task
                    chunk_rows = [
                        row
                        for event_rows in executor.map(task, payloads, chunksize=4)
                        for row in event_rows
                    ]
                frame = _events_to_frame(chunk_rows)
                temporary = shard.with_suffix(".parquet.partial")
                frame.to_parquet(temporary, index=False, compression="zstd")
                temporary.replace(shard)
            rows.extend(pd.read_parquet(shard).to_dict(orient="records"))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return _events_to_frame(rows)


def union_receptor_holos(events: pd.DataFrame) -> pd.DataFrame:
    valid = events.loc[events["coordinate_union_status"].eq("ok")].copy()
    groups: list[dict[str, object]] = []
    grouping = ["receptor_holo_pdb", "sequence", "cluster_id", "receptor_side"]
    for _, group in valid.groupby(grouping, sort=True):
        ordered = group.sort_values(
            ["receptor_side", "system_id", "partner_holo_pdb"], kind="stable"
        )
        if "coordinate_event_trainable" in ordered:
            trainable = ordered.loc[pd.to_numeric(
                ordered["coordinate_event_trainable"], errors="coerce"
            ).fillna(0).astype(bool)]
            reference_pool = trainable if len(trainable) else ordered
            reference = reference_pool.assign(
                _coverage=pd.to_numeric(reference_pool["mask_coverage"], errors="coerce").fillna(0),
                _length=pd.to_numeric(reference_pool["num_residues"], errors="coerce").fillna(0),
            ).sort_values(["_coverage", "_length"], kind="stable").iloc[-1]
            ordered = trainable if len(trainable) else ordered
        else:
            reference = ordered.iloc[0]
        base = reference.to_dict()
        base_residue_ids = str(base["input_residue_ids"])
        base_sequence = str(base["sequence"])
        labels = np.zeros(int(base["num_residues"]), dtype=np.int8)
        raw_mask = str(base.get("label_mask", ""))
        masks = np.fromstring(raw_mask, sep=";", dtype=np.int64).astype(np.int8)
        if len(masks) != len(labels):
            masks = np.ones(len(labels), dtype=np.int8)
        success = 0
        for _, row in ordered.iterrows():
            if str(row["input_residue_ids"]) != base_residue_ids or str(row["sequence"]) != base_sequence:
                continue
            row_labels = np.fromstring(str(row["labels"]), sep=";", dtype=np.int64).astype(np.int8)
            raw_row_mask = str(row.get("label_mask", ""))
            row_masks = np.fromstring(raw_row_mask, sep=";", dtype=np.int64).astype(np.int8)
            if len(row_masks) != len(labels):
                row_masks = np.ones(len(labels), dtype=np.int8) if not raw_row_mask else row_masks
            if len(row_labels) != len(labels) or len(row_masks) != len(labels):
                continue
            labels |= np.fromstring(str(row["labels"]), sep=";", dtype=np.int64).astype(np.int8)
            masks &= row_masks
            success += 1
        labels &= masks
        base.update({
            "system_id": str(base["system_id"]),
            "receptor_side": str(base["receptor_side"]),
            "num_positive_labels": int(labels.sum()),
            "labels": ";".join(str(int(value)) for value in labels),
            "label_mask": ";".join(str(int(value)) for value in masks),
            "input_residue_ids": base_residue_ids,
            "holo_event_count": len(ordered),
            "holo_event_success_count": success,
            "holo_event_failure_count": 0,
            "mask_coverage": float(masks.mean()) if len(masks) else 0.0,
            "low_mapping_quality": int(masks.mean() < 0.95) if len(masks) else 1,
            "coordinate_union_status": "ok",
            "positive_rate": float(labels.sum() / max(int(masks.sum()), 1)),
        })
        groups.append(base)
    return pd.DataFrame(groups)


def quality_mask(frame: pd.DataFrame, options: PipelineOptions) -> pd.Series:
    residues = pd.to_numeric(frame["num_residues"], errors="coerce")
    positives = pd.to_numeric(frame["num_positive_labels"], errors="coerce")
    coverage = pd.to_numeric(frame["mask_coverage"], errors="coerce")
    rate = positives / residues
    return (
        residues.between(options.min_length, options.max_length, inclusive="both")
        & positives.ge(options.min_positives)
        & coverage.ge(options.min_mask_coverage)
        & frame["low_mapping_quality"].eq(0)
        & frame["coordinate_union_status"].eq("ok")
        & rate.le(options.max_positive_rate)
        & frame["sequence_identity"].notna()
    )


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    target = float(pd.to_numeric(scored["num_residues"], errors="coerce").median())
    coverage = pd.to_numeric(scored["mask_coverage"], errors="coerce").fillna(0.0)
    residues = pd.to_numeric(scored["num_residues"], errors="coerce").fillna(target)
    failures = pd.to_numeric(scored["holo_event_failure_count"], errors="coerce").fillna(999.0)
    successes = pd.to_numeric(scored["holo_event_success_count"], errors="coerce").fillna(0.0)
    scored["_coverage_deficit"] = 1.0 - coverage
    scored["_event_failures"] = failures
    scored["_event_success"] = -successes
    scored["_length_distance"] = (residues - target).abs() / max(target, 1.0)
    scored["_stable"] = [stable_hash(str(holo)) for holo in scored["receptor_holo_pdb"]]
    scored["_score"] = list(zip(
        scored["_coverage_deficit"], scored["_event_failures"],
        scored["_event_success"], scored["_length_distance"], scored["_stable"],
    ))
    return scored


def sequence_diverse_select(frame: pd.DataFrame, cap: int) -> dict[str, int]:
    selected: dict[str, int] = {}
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        row["case_id"] = str(row["receptor_holo_pdb"])
        by_cluster[str(row["cluster_id"])].append(row)
    for cluster in sorted(by_cluster):
        best = min(by_cluster[cluster], key=lambda item: item["_score"])
        selected[str(best["case_id"])] = 1
    cluster_of = {
        str(row["case_id"]): cluster
        for cluster, rows in by_cluster.items() for row in rows
    }
    sequence_of = {
        str(row["case_id"]): str(row["sequence_identity"])
        for rows in by_cluster.values() for row in rows
    }
    sequence_counts = Counter(sequence_of.values())
    for slot in range(2, cap + 1):
        used: dict[str, set[str]] = defaultdict(set)
        for case_id, existing_slot in selected.items():
            if existing_slot < slot:
                used[cluster_of[case_id]].add(sequence_of[case_id])
        choices = []
        for cluster in sorted(by_cluster):
            candidates = [
                row for row in by_cluster[cluster]
                if str(row["case_id"]) not in selected
                and str(row["sequence_identity"]) not in used[cluster]
            ]
            if not candidates:
                continue
            row = min(candidates, key=lambda item: item["_score"])
            sequence = str(row["sequence_identity"])
            choices.append((sequence_counts[sequence], row["_score"], cluster, sequence, str(row["case_id"])))
        for _, _, cluster, sequence, case_id in sorted(choices):
            if cluster_of[case_id] != cluster or sequence in used[cluster]:
                continue
            selected[case_id] = slot
            sequence_counts[sequence] += 1
            used[cluster].add(sequence)
    return selected


def anchor_selection(
    frame: pd.DataFrame, anchor_path: Path, cap: int, preserve_all_anchors: bool = False
) -> tuple[dict[str, int], dict[str, int]]:
    anchor = pd.read_csv(anchor_path)
    if "holo_pdb" not in anchor:
        if "receptor_pdb_relpath" in anchor:
            anchor = anchor.assign(holo_pdb=anchor["receptor_pdb_relpath"].map(lambda value: Path(str(value)).name))
        elif "apex_case_id" in anchor:
            anchor = anchor.assign(holo_pdb=anchor["apex_case_id"].astype(str).str.removeprefix("pinder_full_holo::"))
        else:
            raise ValueError("anchor manifest lacks holo_pdb or receptor_pdb_relpath")
    if "selection_origin" not in anchor:
        anchor["selection_origin"] = "historical_manifest"
    if "sequence_diversity_slot" not in anchor:
        anchor["sequence_diversity_slot"] = 1
    candidate = frame.copy()
    candidate["case_id"] = candidate["receptor_holo_pdb"]
    candidate_ids = set(candidate["case_id"])
    anchor["case_id"] = anchor["holo_pdb"].astype(str)
    anchor = anchor.loc[anchor["case_id"].isin(candidate_ids)].copy()
    sequence_of = dict(zip(candidate["case_id"], candidate["sequence_identity"]))
    cluster_of = dict(zip(candidate["case_id"], candidate["cluster_id"]))
    priority = {
        "relaxed10477_strict_official_cluster_recollapse_v1": 0,
        "official_validation_holo_quality_only": 1,
        "invalid_official_train_side_overlap": 2,
        "invalid_official_val_side_overlap": 3,
    }
    anchor["_priority"] = anchor["selection_origin"].map(priority).fillna(9)
    anchor["_slot"] = pd.to_numeric(anchor["sequence_diversity_slot"], errors="coerce").fillna(999999)
    selected: dict[str, int] = {}
    used_by_cluster: dict[str, set[str]] = defaultdict(set)
    audit = {"anchor_rows_not_in_candidate": int(len(pd.read_csv(anchor_path)) - len(anchor))}
    if preserve_all_anchors:
        for case_id, cluster in sorted(zip(anchor["case_id"].astype(str), anchor["cluster_id"].astype(str))):
            selected[case_id] = 0
        audit["anchor_rows_preserved_by_contract"] = int(len(selected))
        return selected, audit
    for _, group in anchor.sort_values(["_priority", "_slot", "case_id"]).groupby("cluster_id", sort=True):
        slot = 0
        for row in group.to_dict(orient="records"):
            case_id = str(row["case_id"])
            cluster = str(cluster_of[case_id])
            sequence = str(sequence_of[case_id])
            if slot >= cap:
                audit["anchor_rows_dropped_by_global_cap"] = audit.get("anchor_rows_dropped_by_global_cap", 0) + 1
            elif sequence in used_by_cluster[cluster]:
                audit["anchor_rows_dropped_by_duplicate_sequence"] = audit.get("anchor_rows_dropped_by_duplicate_sequence", 0) + 1
            else:
                selected[case_id] = slot + 1
                used_by_cluster[cluster].add(sequence)
                slot += 1
    return selected, audit


def enforce_cluster_cap(
    frame: pd.DataFrame, selected: dict[str, int], cap: int
) -> tuple[dict[str, int], int]:
    frame = frame.assign(case_id=frame["receptor_holo_pdb"])
    cluster_of = dict(zip(frame["case_id"], frame["cluster_id"]))
    counts: Counter[str] = Counter()
    retained: dict[str, int] = {}
    dropped = 0
    for case_id in sorted(selected):
        cluster = str(cluster_of[case_id])
        if counts[cluster] >= cap:
            dropped += 1
            continue
        counts[cluster] += 1
        retained[case_id] = counts[cluster]
    return retained, dropped


def _write_single_chain(source: Path, chain_name: str, destination: Path) -> None:
    structure = gemmi.read_structure(str(source))
    model = structure[0]
    for chain in list(model):
        if chain.name != chain_name:
            model.remove_chain(chain.name)
    structure.setup_entities()
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(destination))


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def make_manifest(
    frame: pd.DataFrame,
    adapter: PinderApiAdapter,
    output_dir: Path,
    selected_ids: dict[str, int],
) -> pd.DataFrame:
    work = frame.copy()
    work["case_id"] = work["receptor_holo_pdb"]
    work = work.loc[work["case_id"].isin(selected_ids)].copy()
    work["sequence_diversity_slot"] = work["case_id"].map(selected_ids)
    records: list[dict[str, object]] = []
    structure_dir = output_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    for row in work.to_dict(orient="records"):
        case_id = str(row["case_id"])
        structure_path = structure_dir / f"{stable_hash(case_id)}.pdb"
        if not structure_path.is_file():
            if str(row.get("selection_origin", "")) == "official_api_holo_mapping_union":
                shutil.copyfile(adapter.holo_path(case_id), structure_path)
            else:
                _write_single_chain(
                    adapter.native_path(str(row["system_id"])),
                    str(row["receptor_side"]),
                    structure_path,
                )
        relpath = _relative_or_absolute(structure_path)
        records.append({
            "apex_case_id": f"pinder_api_2024_02::{case_id}",
            "system_id": row["system_id"],
            "cluster_id": row["cluster_id"],
            "receptor_uniprot": row["receptor_uniprot"],
            "input_state": "holo",
            "receptor_side": row["receptor_side"],
            "receptor_chain_id": "",
            "labels": row["labels"],
            "label_mask": row["label_mask"],
            "input_residue_ids": row["input_residue_ids"],
            "mapped_uniprot_positions": "",
            "sequence": row["sequence"],
            "num_residues": row["num_residues"],
            "receptor_pdb_relpath": relpath,
            "label_contract": "native_holo_nonhydrogen_contact_le_5A",
            "receptor_pdb_path": relpath,
            "compact_path": "",
        })
    return pd.DataFrame(records, columns=MANIFEST_COLUMNS).sort_values("apex_case_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(train: pd.DataFrame, test: pd.DataFrame, test_sides: set[str]) -> dict[str, object]:
    errors: list[str] = []
    if train["apex_case_id"].duplicated().any():
        errors.append("duplicate train apex_case_id")
    if test["apex_case_id"].duplicated().any():
        errors.append("duplicate test apex_case_id")
    if set(train["apex_case_id"]) & set(test["apex_case_id"]):
        errors.append("train/test apex overlap")
    if set(train["cluster_id"]) & test_sides:
        errors.append("official test side-cluster leakage")
    sizes = train.groupby("cluster_id").size()
    if len(sizes) and int(sizes.max()) > 5:
        errors.append(f"cluster cap exceeded: {int(sizes.max())}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "train_rows": int(len(train)),
        "train_clusters": int(train["cluster_id"].nunique()),
        "max_train_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "test_rows": int(len(test)),
        "official_test_side_clusters": int(len(test_sides)),
        "train_test_case_overlap": 0,
    }


def _manifest_holo_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(
        frame["receptor_pdb_relpath"].astype(str).map(lambda value: Path(value).name),
        frame["receptor_side"].astype(str),
    ))


def _array_contract(frame: pd.DataFrame) -> bool:
    return all(
        str(row.labels).count(";") + 1 == int(row.num_residues)
        and str(row.label_mask).count(";") + 1 == int(row.num_residues)
        and str(row.input_residue_ids).count(";") + 1 == int(row.num_residues)
        for row in frame.itertuples(index=False)
    )


def validate_published_pinder_splits(
    index: pd.DataFrame,
    train: pd.DataFrame,
    holo_test: pd.DataFrame,
    apo_af2_test: pd.DataFrame,
) -> tuple[dict[str, object], bool]:
    test_index = index.assign(id=index["id"].astype(str)).set_index("id", drop=False)
    official_test_ids = set(test_index.index[test_index["split"].eq("test")])
    test_sides = official_test_side_clusters(index)
    train_keys = _manifest_holo_keys(train)
    train_test_case_overlap = set(train["system_id"].dropna().astype(str)) & official_test_ids
    official_directed = _directed_index_events(index)
    official_keys = set(zip(
        official_directed["receptor_holo_pdb"].astype(str),
        official_directed["receptor_side"].astype(str),
    ))

    def test_contract(frame: pd.DataFrame, allowed_states: set[str]) -> dict[str, object]:
        system_ids = frame["system_id"].astype(str)
        side_uniprot = []
        official_holo_sides = []
        for row in frame.itertuples(index=False):
            source = test_index.loc[str(row.system_id)]
            side = str(row.receptor_side)
            side_uniprot.append(str(row.receptor_uniprot) == str(source[f"uniprot_{side}"]))
            official_holo_sides.append(
                (str(source[f"holo_{side}_pdb"]), side) in official_keys
            )
        return {
            "rows": int(len(frame)),
            "systems": int(system_ids.nunique()),
            "states": {
                str(key): int(value)
                for key, value in frame["input_state"].value_counts().sort_index().items()
            },
            "all_systems_official_test": bool(system_ids.isin(official_test_ids).all()),
            "all_states_allowed": bool(set(frame["input_state"].astype(str)).issubset(allowed_states)),
            "all_receptor_sides_valid": bool(frame["receptor_side"].isin(("R", "L")).all()),
            "all_receptor_uniprots_match_official_index": bool(all(side_uniprot)),
            "all_holo_sides_exist_in_official_index": bool(all(official_holo_sides)),
            "label_mask_residue_array_lengths_match": _array_contract(frame),
        }

    holo_contract = test_contract(holo_test, {"holo"})
    apo_contract = test_contract(apo_af2_test, {"apo", "af2"})
    holo_systems = set(holo_test["system_id"].astype(str))
    apo_systems = set(apo_af2_test["system_id"].astype(str))
    holo_source_keys = {
        (str(test_index.loc[system_id, f"holo_{side}_pdb"]), str(side))
        for system_id in holo_systems
        for side in ("R", "L")
    }
    train_contract = {
        "rows": int(len(train)),
        "unique_holo_sides": int(len(train_keys)),
        "all_holo_sides_in_official_index": train_keys.issubset(official_keys),
        "no_official_test_side_cluster": not (set(train["cluster_id"].astype(str)) & test_sides),
        "no_official_test_system_id": len(train_test_case_overlap) == 0,
        "label_mask_residue_array_lengths_match": _array_contract(train),
    }
    relationships = {
        "apo_af2_systems_subset_of_holo_test": apo_systems.issubset(holo_systems),
        "train_holo_side_overlap_with_holo_test_sources": int(len(train_keys & holo_source_keys)),
    }
    passed = (
        all(value for value in train_contract.values() if isinstance(value, bool))
        and all(value for value in holo_contract.values() if isinstance(value, bool))
        and all(value for value in apo_contract.values() if isinstance(value, bool))
        and relationships["apo_af2_systems_subset_of_holo_test"]
        and relationships["train_holo_side_overlap_with_holo_test_sources"] == 0
    )
    report = {
        "schema_version": "gsbind_published_pinder_split_contract_v1",
        "release": SUPPORTED_RELEASE,
        "train": train_contract,
        "holo_test": holo_contract,
        "apo_af2_test": apo_contract,
        "relationships": relationships,
        "all_checks_passed": bool(passed),
    }
    return report, passed


def validate_published_pinder_splits_main(argv: Sequence[str] | None = None) -> dict[str, object]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate published PINDER train/test manifests against the official API index."
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--train", type=Path, default=Path("datasets/main/train.csv"))
    parser.add_argument("--holo-test", type=Path, default=Path("datasets/main/holo_test.csv"))
    parser.add_argument("--apo-af2-test", type=Path, default=Path("datasets/main/apo_af2_test.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    adapter = PinderApiAdapter(cache_dir=args.cache_dir)
    report, passed = validate_published_pinder_splits(
        adapter.load_index(),
        pd.read_csv(args.train, keep_default_na=False),
        pd.read_csv(args.holo_test, keep_default_na=False),
        pd.read_csv(args.apo_af2_test, keep_default_na=False),
    )
    if args.output is not None:
        _atomic_json(args.output, report)
    if not passed:
        raise RuntimeError(f"published PINDER split contract failed: {report}")
    return report


def _target_key_from_manifest(row: pd.Series) -> tuple[str, str]:
    if "receptor_pdb_relpath" in row and str(row.get("receptor_pdb_relpath", "")).endswith(".pdb"):
        holo = Path(str(row["receptor_pdb_relpath"])).name
    else:
        holo = str(row.get("holo_pdb", ""))
    return holo, str(row.get("receptor_side", ""))


def compare_target_manifest(
    rebuilt: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, object]:
    rebuilt = rebuilt.copy()
    reference = pd.read_csv(reference) if isinstance(reference, Path) else reference.copy()
    rebuilt["receptor_holo_pdb"] = rebuilt["apex_case_id"].astype(str).str.removeprefix("pinder_api_2024_02::")
    rebuilt["receptor_side_key"] = rebuilt["receptor_side"].astype(str)
    reference["receptor_holo_pdb"] = reference["receptor_pdb_relpath"].map(lambda value: Path(str(value)).name)
    reference["receptor_side_key"] = reference["receptor_side"].astype(str)
    rebuilt_keys = set(zip(rebuilt["receptor_holo_pdb"], rebuilt["receptor_side_key"]))
    reference_keys = set(zip(reference["receptor_holo_pdb"], reference["receptor_side_key"]))
    joined = reference.merge(
        rebuilt,
        on=["receptor_holo_pdb", "receptor_side_key"],
        how="outer",
        suffixes=("_reference", "_rebuilt"),
        indicator=True,
    )
    both = joined.loc[joined["_merge"].eq("both")].copy()
    both["reference_labels"] = both["labels_reference"].astype(str)
    both["rebuilt_labels"] = both["labels_rebuilt"].astype(str)
    both["reference_mask"] = both["label_mask_reference"].astype(str)
    both["rebuilt_mask"] = both["label_mask_rebuilt"].astype(str)
    both["reference_positive_count"] = both["reference_labels"].map(lambda value: value.count("1"))
    both["rebuilt_positive_count"] = both["rebuilt_labels"].map(lambda value: value.count("1"))
    both["reference_valid_count"] = both["reference_mask"].map(lambda value: value.count("1"))
    exact_labels = both["reference_labels"].eq(both["rebuilt_labels"])
    exact_masks = both["reference_mask"].eq(both["rebuilt_mask"])
    exact_residue_ids = both["input_residue_ids_reference"].astype(str).eq(
        both["input_residue_ids_rebuilt"].astype(str)
    )
    exact_residues = pd.to_numeric(both["num_residues_reference"], errors="coerce").eq(
        pd.to_numeric(both["num_residues_rebuilt"], errors="coerce")
    )
    exact_positives = both["reference_positive_count"].eq(both["rebuilt_positive_count"])
    exact_clusters = both["cluster_id_reference"].astype(str).eq(both["cluster_id_rebuilt"].astype(str))
    exact_uniprot = both["receptor_uniprot_reference"].fillna("").astype(str).eq(
        both["receptor_uniprot_rebuilt"].fillna("").astype(str)
    )
    return {
        "reference_rows": int(len(reference)),
        "rebuilt_rows": int(len(rebuilt)),
        "reference_unique_holo_sides": int(len(reference_keys)),
        "rebuilt_unique_holo_sides": int(len(rebuilt_keys)),
        "missing_from_rebuild": int(len(reference_keys - rebuilt_keys)),
        "extra_in_rebuild": int(len(rebuilt_keys - reference_keys)),
        "matched_rows": int(len(both)),
        "exact_label_rows": int(exact_labels.sum()),
        "exact_label_mask_rows": int(exact_masks.sum()),
        "exact_residue_id_rows": int(exact_residue_ids.sum()),
        "exact_residue_count_rows": int(exact_residues.sum()),
        "exact_positive_count_rows": int(exact_positives.sum()),
        "exact_cluster_rows": int(exact_clusters.sum()),
        "exact_uniprot_rows": int(exact_uniprot.sum()),
        "reference_valid_residues": int(both["reference_valid_count"].sum()),
        "rebuilt_valid_residues": int(both["rebuilt_mask"].map(lambda value: value.count("1")).sum()),
        "reference_positive_residues": int(both["reference_positive_count"].sum()),
        "rebuilt_positive_residues": int(both["rebuilt_positive_count"].sum()),
        "label_mismatch_examples": [
            {
                "holo": row["receptor_holo_pdb"],
                "side": row["receptor_side_key"],
                "reference": row["reference_labels"],
                "rebuilt": row["rebuilt_labels"],
            }
            for row in both.loc[~exact_labels].head(10).to_dict(orient="records")
        ],
    }


def reproduce_anchor_from_api(
    index: pd.DataFrame,
    adapter: PinderApiAdapter,
    options: PipelineOptions,
    reference_manifest: Path,
) -> dict[str, object]:
    if options.anchor_manifest is not None:
        reference_manifest = options.anchor_manifest
    systems, cluster_map, source_audit = select_anchor_source_systems(index, reference_manifest)
    metadata_cache = options.output_dir / "events" / "anchor_sources" / "historical_metadata.parquet"
    if metadata_cache.is_file():
        historical_metadata_frame = pd.read_parquet(metadata_cache)
    else:
        test_sides = official_test_side_clusters(index)
        non_test_ids = index.loc[
            index["split"].isin(("val", "invalid"))
            & ~index["cluster_id_R"].isin(test_sides)
            & ~index["cluster_id_L"].isin(test_sides),
            "id",
        ].astype(str)
        metadata_ids = sorted(set(non_test_ids) | set(systems["id"].astype(str)))
        historical_metadata_frame = adapter.load_historical_metadata(metadata_ids)
        metadata_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_cache.with_suffix(".parquet.partial")
        historical_metadata_frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(metadata_cache)
    historical_mapping_holos, mapping_audit = historical_mapping_pool(
        index, historical_metadata_frame
    )
    source_audit.update(mapping_audit)
    events = process_systems(
        systems,
        adapter,
        options,
        options.output_dir / "events" / "anchor_sources",
        target_keys=set(cluster_map),
        historical_metadata_frame=historical_metadata_frame,
        historical_mapping_holos=historical_mapping_holos,
    )
    events = apply_anchor_clusters(events, cluster_map)
    unions = union_receptor_holos(events)
    eligible = unions.loc[quality_mask(unions, options)].copy()
    selected, anchor_audit = anchor_selection(
        eligible, reference_manifest, options.cluster_cap, preserve_all_anchors=True
    )
    selected, cap_dropped = enforce_cluster_cap(eligible, selected, options.cluster_cap)
    manifest = make_manifest(eligible, adapter, options.output_dir, selected)
    output = options.output_dir / "train.csv"
    manifest.to_csv(output, index=False, lineterminator="\n")
    comparison = compare_target_manifest(manifest, reference_manifest)
    test_sides = official_test_side_clusters(index)
    checks = {
        "all_reference_targets_present": comparison["missing_from_rebuild"] == 0,
        "no_extra_targets": comparison["extra_in_rebuild"] == 0,
        "row_count_matches_reference": len(manifest) == len(pd.read_csv(reference_manifest)),
        "all_exact_labels": comparison["exact_label_rows"] == comparison["matched_rows"],
        "all_exact_label_masks": comparison["exact_label_mask_rows"] == comparison["matched_rows"],
        "all_exact_residue_ids": comparison["exact_residue_id_rows"] == comparison["matched_rows"],
        "all_exact_residue_counts": comparison["exact_residue_count_rows"] == comparison["matched_rows"],
        "no_official_test_side_cluster": not (set(manifest["cluster_id"]) & test_sides),
        "cluster_cap": int(manifest.groupby("cluster_id").size().max()) <= options.cluster_cap,
    }
    report = {
        "schema_version": "gsbind_pinder_anchor_full_reproduction_v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": adapter.describe(),
        "source_audit": source_audit,
        "anchor_audit": anchor_audit,
        "rows_dropped_by_final_global_cap_enforcement": int(cap_dropped),
        "directed_events_processed": int(len(events)),
        "receptor_unions": int(len(unions)),
        "eligible_receptor_unions": int(len(eligible)),
        "comparison": comparison,
        "checks": checks,
        "artifacts": {
            "train": str(output.resolve()),
            "train_sha256": _sha256(output),
            "reference_manifest": str(reference_manifest.resolve()),
        },
    }
    _atomic_json(options.output_dir / "anchor_reproduction_report.json", report)
    if not all(checks.values()):
        raise RuntimeError(f"anchor reproduction contract failed: {checks}")
    (options.output_dir / "ANCHOR_REPRODUCTION_COMPLETE").write_text("complete\n", encoding="ascii")
    return report


def run_pipeline(
    index_or_path: pd.DataFrame | Path,
    adapter: PinderApiAdapter,
    options: PipelineOptions,
) -> dict[str, object]:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(index_or_path) if isinstance(index_or_path, Path) else index_or_path
    training_systems, test_systems, source_audit = select_candidate_and_test_systems(index, options.limit)
    event_dir = options.output_dir / "events"
    train_events = process_systems(training_systems, adapter, options, event_dir / "training")
    test_events = process_systems(test_systems, adapter, options, event_dir / "test")
    train_union = union_receptor_holos(train_events)
    test_union = union_receptor_holos(test_events)
    eligible = train_union.loc[quality_mask(train_union, options)].copy()
    eligible = add_scores(eligible)
    if options.anchor_manifest is not None:
        selected, anchor_audit = anchor_selection(eligible, options.anchor_manifest, options.cluster_cap)
    else:
        selected = sequence_diverse_select(eligible, options.cluster_cap)
        anchor_audit = {"anchor_mode": "deterministic_api_selection"}
    selected, cap_dropped = enforce_cluster_cap(eligible, selected, options.cluster_cap)
    selected_frame = eligible.assign(case_id=eligible["receptor_holo_pdb"]).loc[
        lambda frame: frame["case_id"].isin(selected)
    ]
    train_manifest = make_manifest(eligible, adapter, options.output_dir, selected)
    test_eligible = test_union.loc[
        test_union["num_residues"].between(options.min_length, options.max_length)
        & test_union["num_positive_labels"].ge(options.min_positives)
    ].copy()
    test_manifest = make_manifest(
        test_eligible,
        adapter,
        options.output_dir,
        {str(row.receptor_holo_pdb): index + 1 for index, row in test_eligible.iterrows()},
    )
    train_path = options.output_dir / "train.csv"
    test_path = options.output_dir / "test.csv"
    train_manifest.to_csv(train_path, index=False, lineterminator="\n")
    test_manifest.to_csv(test_path, index=False, lineterminator="\n")
    train_union.to_parquet(event_dir / "training_receptor_unions.parquet", index=False, compression="zstd")
    test_union.to_parquet(event_dir / "test_receptor_unions.parquet", index=False, compression="zstd")
    test_sides = official_test_side_clusters(index)
    checks = validate_split(train_manifest, test_manifest, test_sides)
    residues = pd.to_numeric(train_manifest["num_residues"])
    positives = train_manifest["labels"].map(lambda value: str(value).count("1"))
    report = {
        "schema_version": "gsbind_pinder_api_training_test_rebuild_v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": adapter.describe(),
        "source_audit": source_audit,
        "contract": {
            "training_sources": list(TRAIN_SPLITS),
            "leakage_rule": "exclude any train/val/invalid system whose either side cluster is in official test side-cluster union",
            "coordinate_label": "native holo non-hydrogen atom contact <= 5A",
            "receptor_union": "same receptor holo, sequence and residue contract; labels OR, masks intersection",
            "quality": asdict(options),
        },
        **anchor_audit,
        "rows_dropped_by_final_global_cap_enforcement": int(cap_dropped),
        "directed_events": int(len(train_events)),
        "training_receptor_unions": int(len(train_union)),
        "eligible_receptor_unions": int(len(eligible)),
        "selected_source_split_counts": {
            str(split): int(count)
            for split, count in selected_frame["official_split"].value_counts().sort_index().items()
        },
        "test_directed_events": int(len(test_events)),
        "test_manifest_rows": int(len(test_manifest)),
        "masked_valid_residues": int(residues.sum()),
        "masked_positive_residues": int(positives.sum()),
        "pooled_positive_rate": float(positives.sum() / residues.sum()) if int(residues.sum()) else 0.0,
        "checks": checks,
        "artifacts": {
            "train": str(train_path.resolve()),
            "train_sha256": _sha256(train_path),
            "test": str(test_path.resolve()),
            "test_sha256": _sha256(test_path),
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "argv": sys.argv},
    }
    _atomic_json(options.output_dir / "build_report.json", report)
    split_contract = {
        "schema_version": "gsbind_pinder_api_split_contract_v1",
        "release": options.release,
        "train": {"path": "train.csv", "sha256": report["artifacts"]["train_sha256"]},
        "test": {"path": "test.csv", "sha256": report["artifacts"]["test_sha256"]},
        "checks": checks,
    }
    _atomic_json(options.output_dir / "split_contract.json", split_contract)
    (options.output_dir / "REBUILD_COMPLETE").write_text("complete\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release", default=SUPPORTED_RELEASE)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cluster-cap", type=int, default=5)
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--anchor-manifest", type=Path)
    parser.add_argument("--anchor-only", action="store_true")
    parser.add_argument("--update-index", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.contact_cutoff != 5.0:
        raise ValueError("The GSbind training contract requires --contact-cutoff=5.0")
    adapter = PinderApiAdapter(args.release, args.cache_dir)
    index = adapter.load_index(update=args.update_index)
    options = PipelineOptions(
        output_dir=args.output_dir.resolve(),
        release=args.release,
        workers=args.workers,
        download_workers=args.download_workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
        cluster_cap=args.cluster_cap,
        contact_cutoff=args.contact_cutoff,
        anchor_manifest=args.anchor_manifest.resolve() if args.anchor_manifest else None,
        update_index=args.update_index,
    )
    if args.anchor_only:
        if args.anchor_manifest is None:
            raise ValueError("--anchor-only requires --anchor-manifest")
        return reproduce_anchor_from_api(
            index, adapter, options, args.anchor_manifest.resolve()
        )
    return run_pipeline(index, adapter, options)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
