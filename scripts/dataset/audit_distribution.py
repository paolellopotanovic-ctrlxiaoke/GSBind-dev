#!/usr/bin/env python3
"""Audit manifest identity, cluster, length, mask, and positive-label distributions."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from _common import length_bucket, read_csv, row_length, sha256, split_field, structure_key, write_json


DEFAULT_LENGTH_EDGES = (0, 29, 50, 100, 200, 400, 600, 851)
RATE_EDGES = (0.0, 0.05, 0.10, 0.16, 0.30, 0.50, 0.75, 1.0000001)


def audit(rows: list[dict[str, str]], edges=DEFAULT_LENGTH_EDGES) -> dict[str, object]:
    lengths: list[int] = []
    positives = 0
    valid_residues = 0
    length_buckets: Counter[str] = Counter()
    rate_buckets: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    invalid_rows: list[str] = []
    for row in rows:
        case_id = row.get("apex_case_id", "")
        labels = split_field(row.get("labels", ""), int)
        mask = split_field(row.get("label_mask", ""), int)
        length = row_length(row)
        if length <= 0 or len(labels) != length or len(mask) != length:
            invalid_rows.append(case_id)
            continue
        valid = sum(mask)
        positive = sum(y * m for y, m in zip(labels, mask))
        rate = positive / valid if valid else 0.0
        lengths.append(length)
        positives += positive
        valid_residues += valid
        length_buckets[length_bucket(length, edges)] += 1
        for low, high in zip(RATE_EDGES, RATE_EDGES[1:]):
            if low <= rate < high:
                rate_buckets[f"{low:.2f}-{min(high, 1.0):.2f}"] += 1
                break
        cluster_counts[row.get("cluster_id", "")] += 1
    case_ids = [row.get("apex_case_id", "") for row in rows]
    paths = [structure_key(row.get("receptor_pdb_path", "")) for row in rows]
    populated_clusters = {key: count for key, count in cluster_counts.items() if key}
    return {
        "rows": len(rows),
        "unique_case_ids": len(set(case_ids)),
        "unique_structure_paths": len(set(paths)),
        "duplicate_case_ids": len(case_ids) - len(set(case_ids)),
        "duplicate_structure_paths": len(paths) - len(set(paths)),
        "clusters": len(populated_clusters),
        "cluster_cap_observed": max(populated_clusters.values(), default=0),
        "invalid_rows": invalid_rows,
        "zero_valid_rows": sum(
            sum(split_field(row.get("label_mask", ""), int)) == 0 for row in rows
        ),
        "length": {
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "mean": statistics.fmean(lengths) if lengths else 0,
            "max": max(lengths, default=0),
            "buckets": dict(sorted(length_buckets.items())),
        },
        "positive": {
            "residues": positives,
            "valid_residues": valid_residues,
            "global_rate": positives / valid_residues if valid_residues else 0.0,
            "case_rate_buckets": dict(sorted(rate_buckets.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length-edges", default=",".join(map(str, DEFAULT_LENGTH_EDGES)))
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--max-cluster-cap", type=int)
    parser.add_argument("--require-unique-paths", action="store_true")
    args = parser.parse_args()
    edges = tuple(int(value) for value in args.length_edges.split(","))
    report = audit(read_csv(args.manifest), edges)
    report["manifest"] = str(Path(args.manifest))
    report["manifest_sha256"] = sha256(args.manifest)
    failures = []
    if args.expected_rows is not None and report["rows"] != args.expected_rows:
        failures.append("rows")
    if args.max_cluster_cap is not None and report["cluster_cap_observed"] > args.max_cluster_cap:
        failures.append("cluster_cap")
    if args.require_unique_paths and report["duplicate_structure_paths"]:
        failures.append("duplicate_structure_paths")
    if report["duplicate_case_ids"] or report["invalid_rows"] or report["zero_valid_rows"]:
        failures.append("manifest_integrity")
    report["pass"] = not failures
    report["failures"] = failures
    write_json(args.output, report)
    print(report)
    if failures:
        raise SystemExit(f"Distribution audit failed: {failures}")


if __name__ == "__main__":
    main()
