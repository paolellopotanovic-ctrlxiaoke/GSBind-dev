#!/usr/bin/env python3
"""Rebuild a training/test split directly from the public PINDER 2024-02 API."""

from __future__ import annotations

import hashlib
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

    def download_native(self, system_ids: Iterable[str]) -> None:
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
            dataset.download_files(sources, destinations)

    def native_path(self, system_id: str) -> Path:
        return self.cache_dir / "pdbs" / f"{system_id}.pdb"

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
    ]
    return pd.DataFrame(rows, columns=columns)


def process_systems(
    systems: pd.DataFrame,
    adapter: PinderApiAdapter,
    options: PipelineOptions,
    shard_dir: Path,
) -> pd.DataFrame:
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = adapter.cache_dir
    rows: list[dict[str, object]] = []
    total = len(systems)
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
                adapter.download_native(chunk["id"].astype(str).tolist())
                payloads = chunk.rename(columns={"id": "system_id"}).to_dict(orient="records")
                if executor is None:
                    _init_worker(cache_dir, options.contact_cutoff)
                    chunk_rows = [row for payload in payloads for row in _process_system_task(payload)]
                else:
                    chunk_rows = [
                        row
                        for event_rows in executor.map(_process_system_task, payloads, chunksize=4)
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
    grouping = ["receptor_holo_pdb", "sequence", "cluster_id"]
    for _, group in valid.groupby(grouping, sort=True):
        ordered = group.sort_values(
            ["receptor_side", "system_id", "partner_holo_pdb"], kind="stable"
        )
        base = ordered.iloc[0].to_dict()
        if not all(
            row["input_residue_ids"] == base["input_residue_ids"]
            for _, row in ordered.iterrows()
        ):
            continue
        labels = np.zeros(int(base["num_residues"]), dtype=np.int8)
        success = 0
        for _, row in ordered.iterrows():
            labels |= np.fromstring(str(row["labels"]), sep=";", dtype=np.int64).astype(np.int8)
            success += 1
        base.update({
            "system_id": str(base["system_id"]),
            "receptor_side": str(base["receptor_side"]),
            "num_positive_labels": int(labels.sum()),
            "labels": ";".join(str(int(value)) for value in labels),
            "label_mask": ";".join(["1"] * len(labels)),
            "input_residue_ids": ";".join(str(index + 1) for index in range(len(labels))),
            "holo_event_count": len(ordered),
            "holo_event_success_count": success,
            "holo_event_failure_count": 0,
            "mask_coverage": 1.0,
            "low_mapping_quality": 0,
            "coordinate_union_status": "ok",
            "positive_rate": float(labels.sum() / len(labels)),
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
    frame: pd.DataFrame, anchor_path: Path, cap: int
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
    for row in work.to_dict(orient="records"):
        case_id = str(row["case_id"])
        source = adapter.native_path(str(row["system_id"]))
        structure_path = structure_dir / f"{stable_hash(case_id)}.pdb"
        if not structure_path.is_file():
            _write_single_chain(source, str(row["receptor_side"]), structure_path)
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
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cluster-cap", type=int, default=5)
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--anchor-manifest", type=Path)
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
        chunk_size=args.chunk_size,
        limit=args.limit,
        cluster_cap=args.cluster_cap,
        contact_cutoff=args.contact_cutoff,
        anchor_manifest=args.anchor_manifest.resolve() if args.anchor_manifest else None,
        update_index=args.update_index,
    )
    return run_pipeline(index, adapter, options)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
