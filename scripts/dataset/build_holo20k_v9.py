#!/usr/bin/env python3
"""Rebuild a leakage-clean Holo20k v9 by deterministic matched replacement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _common import (
    length_bucket,
    read_csv,
    row_length,
    row_positive_rate,
    sha256,
    stable_digest,
    structure_key,
    write_csv,
    write_json,
)


DEFAULT_EDGES = (0, 50, 100, 200, 400, 600, 851)


def audit_is_clean(path: str | Path) -> bool:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("pass") is True:
        return True
    hit_keys = ("leaking_unique_train_structures", "leakage_hits", "leaky_unique_holo_paths")
    return any(key in payload for key in hit_keys) and all(int(payload.get(key, 0)) == 0 for key in hit_keys)


def leak_keys(path: str | Path) -> set[str]:
    rows = read_csv(path)
    keys: set[str] = set()
    candidate_fields = (
        "query_structure", "holo_pdb", "receptor_pdb_path", "query", "path", "structure",
    )
    for row in rows:
        value = next((row.get(field, "") for field in candidate_fields if row.get(field)), "")
        if value:
            keys.add(structure_key(value))
    return keys


def build_v9(
    base_rows: list[dict[str, str]],
    reserve_rows: list[dict[str, str]],
    leaking: set[str],
    cluster_cap: int,
    edges=DEFAULT_EDGES,
) -> tuple[list[dict[str, str]], list[dict[str, object]], dict[str, object]]:
    base_keys = [structure_key(row["receptor_pdb_path"]) for row in base_rows]
    if len(set(base_keys)) != len(base_keys):
        raise ValueError("Base manifest contains duplicate receptor structures")
    leaking_indices = [index for index, key in enumerate(base_keys) if key in leaking]
    if len(leaking_indices) != len(leaking):
        absent = sorted(leaking - set(base_keys))
        raise ValueError(f"Leak list does not map one-to-one to base rows; absent={absent[:10]}")

    retained = [row for index, row in enumerate(base_rows) if index not in set(leaking_indices)]
    retained_keys = {structure_key(row["receptor_pdb_path"]) for row in retained}
    cluster_counts = Counter(row.get("cluster_id", "") for row in retained)
    reserve = []
    for row in reserve_rows:
        key = structure_key(row.get("receptor_pdb_path", row.get("holo_pdb", "")))
        cluster = row.get("cluster_id", "")
        if not key or key in retained_keys or key in leaking or not cluster:
            continue
        if row.get("receptor_pdb_path", "") == "" and row.get("holo_pdb"):
            row = dict(row)
            row["receptor_pdb_path"] = row["holo_pdb"]
        reserve.append(row)
    reserve_by_key = {structure_key(row["receptor_pdb_path"]): row for row in reserve}
    reserve = list(reserve_by_key.values())

    replacements: list[dict[str, object]] = []
    used: set[str] = set()
    output = list(base_rows)
    for index in leaking_indices:
        removed = base_rows[index]
        removed_length = row_length(removed)
        removed_rate = row_positive_rate(removed)
        bucket = length_bucket(removed_length, edges)
        candidates = []
        for candidate in reserve:
            key = structure_key(candidate["receptor_pdb_path"])
            cluster = candidate.get("cluster_id", "")
            if key in used or cluster_counts[cluster] >= cluster_cap:
                continue
            candidate_length = row_length(candidate)
            if length_bucket(candidate_length, edges) != bucket:
                continue
            candidate_rate = row_positive_rate(candidate)
            new_cluster_penalty = 0 if cluster_counts[cluster] == 0 else 1
            score = (
                new_cluster_penalty,
                abs(candidate_rate - removed_rate),
                abs(candidate_length - removed_length),
                stable_digest(key, cluster),
            )
            candidates.append((score, candidate, candidate_length, candidate_rate))
        if not candidates:
            raise RuntimeError(
                f"No clean reserve candidate satisfies bucket={bucket} and cap={cluster_cap} "
                f"for {removed.get('apex_case_id')}"
            )
        _, selected, selected_length, selected_rate = min(candidates, key=lambda item: item[0])
        selected = dict(selected)
        selected_key = structure_key(selected["receptor_pdb_path"])
        selected_cluster = selected.get("cluster_id", "")
        used.add(selected_key)
        cluster_counts[selected_cluster] += 1
        output[index] = selected
        replacements.append({
            "row_index": index,
            "removed_case_id": removed.get("apex_case_id", ""),
            "removed_structure": structure_key(removed["receptor_pdb_path"]),
            "removed_cluster": removed.get("cluster_id", ""),
            "removed_length": removed_length,
            "removed_positive_rate": removed_rate,
            "replacement_case_id": selected.get("apex_case_id", ""),
            "replacement_structure": selected_key,
            "replacement_cluster": selected_cluster,
            "replacement_length": selected_length,
            "replacement_positive_rate": selected_rate,
            "length_bucket": bucket,
        })

    output_keys = [structure_key(row["receptor_pdb_path"]) for row in output]
    observed_cap = max(Counter(row.get("cluster_id", "") for row in output).values(), default=0)
    if len(output) != len(base_rows) or len(set(output_keys)) != len(output):
        raise RuntimeError("Output row count or unique structure invariant failed")
    if observed_cap > cluster_cap:
        raise RuntimeError(f"Observed cluster cap {observed_cap} exceeds {cluster_cap}")
    summary = {
        "base_rows": len(base_rows),
        "removed_leaking_rows": len(leaking_indices),
        "replacement_rows": len(replacements),
        "output_rows": len(output),
        "output_unique_structures": len(set(output_keys)),
        "output_clusters": len({row.get("cluster_id", "") for row in output if row.get("cluster_id")}),
        "cluster_cap": cluster_cap,
        "observed_cluster_cap": observed_cap,
        "row_positions_preserved": True,
        "matching_order": [
            "same_length_bucket", "new_cluster_preferred", "positive_rate_distance",
            "length_distance", "stable_sha256_tie_break",
        ],
    }
    return output, replacements, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--leak-hits", required=True)
    parser.add_argument("--reserve-manifest", required=True)
    parser.add_argument("--reserve-audit-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replacement-log", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--cluster-cap", type=int, default=20)
    parser.add_argument("--length-edges", default=",".join(map(str, DEFAULT_EDGES)))
    parser.add_argument("--expected-replacements", type=int, default=36)
    args = parser.parse_args()
    if not audit_is_clean(args.reserve_audit_summary):
        raise SystemExit("Reserve audit is missing or has nonzero leakage")
    edges = tuple(int(value) for value in args.length_edges.split(","))
    rows, replacements, summary = build_v9(
        read_csv(args.base_manifest), read_csv(args.reserve_manifest), leak_keys(args.leak_hits),
        args.cluster_cap, edges,
    )
    if len(replacements) != args.expected_replacements:
        raise SystemExit(
            f"Expected {args.expected_replacements} replacements, observed {len(replacements)}"
        )
    write_csv(args.output, rows)
    write_csv(args.replacement_log, replacements)
    summary["base_manifest_sha256"] = sha256(args.base_manifest)
    summary["reserve_manifest_sha256"] = sha256(args.reserve_manifest)
    summary["output_manifest_sha256"] = sha256(args.output)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
