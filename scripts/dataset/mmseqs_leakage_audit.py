#!/usr/bin/env python3
"""Run a strict MMseqs2 identity-30/coverage-80 leakage audit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from _common import pdb_sequence, read_csv, resolve_structure, structure_key, write_csv, write_json


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=MANIFEST_OR_FASTA")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=MANIFEST_OR_FASTA")
    return name, Path(path)


def fasta_from_manifest(
    manifest: Path,
    output: Path,
    source: str,
    roots: list[Path],
) -> tuple[int, list[str]]:
    rows = read_csv(manifest)
    missing: list[str] = []
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        path_value = row.get("receptor_pdb_path") or row.get("holo_pdb") or ""
        key = structure_key(path_value)
        if not key or key in seen:
            continue
        try:
            sequence = pdb_sequence(resolve_structure(path_value, roots))
        except FileNotFoundError:
            sequence = ""
        if not sequence:
            missing.append(row.get("apex_case_id", key or str(index)))
            continue
        seen.add(key)
        records.append((f"{source}::{key}", sequence))
    with output.open("w", encoding="ascii") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n{sequence}\n")
    return len(records), missing


def normalize_source(source: str, path: Path, work: Path, roots: list[Path]) -> tuple[Path, int, list[str]]:
    if path.suffix.lower() in {".fa", ".faa", ".fasta"}:
        output = work / f"{source}.fa"
        records = 0
        with path.open(encoding="utf-8") as source_handle, output.open("w", encoding="ascii") as target:
            for line in source_handle:
                if line.startswith(">"):
                    records += 1
                    target.write(f">{source}::{line[1:].strip()}\n")
                else:
                    target.write(line.strip().upper() + "\n")
        return output, records, []
    output = work / f"{source}.fa"
    count, missing = fasta_from_manifest(path, output, source, roots)
    return output, count, missing


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    roots = [Path(root) for root in args.structure_root]
    query_fasta, query_count, query_missing = normalize_source(
        "train", Path(args.train_manifest), work, roots
    )
    source_fastas: list[Path] = []
    source_counts: dict[str, int] = {}
    missing_by_source: dict[str, list[str]] = {}
    for source, source_path in args.test_source:
        fasta, count, missing = normalize_source(source, source_path, work, roots)
        source_fastas.append(fasta)
        source_counts[source] = count
        missing_by_source[source] = missing
    if query_missing or any(missing_by_source.values()):
        report = {
            "pass": False,
            "stage": "fasta_generation",
            "query_records": query_count,
            "query_missing": query_missing,
            "test_records": source_counts,
            "test_missing": missing_by_source,
        }
        write_json(args.output_summary, report)
        raise RuntimeError("Sequence extraction was incomplete; MMseqs2 audit was not run")

    combined = work / "test_sources.fa"
    with combined.open("wb") as destination:
        for fasta in source_fastas:
            with fasta.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, destination)
    result_tsv = work / "mmseq30_cov80_hits.tsv"
    tmp = work / "mmseqs_tmp"
    command = [
        args.mmseqs, "easy-search", str(query_fasta), str(combined), str(result_tsv), str(tmp),
        "--min-seq-id", str(args.min_identity), "-c", str(args.min_coverage),
        "--cov-mode", "0", "--threads", str(args.threads),
        "--format-output", "query,target,pident,qcov,tcov,alnlen,evalue,bits",
    ]
    subprocess.run(command, check=True)
    hit_rows: list[dict[str, object]] = []
    unique_queries: set[str] = set()
    by_source: Counter[str] = Counter()
    if result_tsv.exists():
        with result_tsv.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 8:
                    raise ValueError(f"Unexpected MMseqs2 row: {line!r}")
                query, target, pident, qcov, tcov, alnlen, evalue, bits = fields
                source = target.split("::", 1)[0]
                query_key = query.split("::", 1)[-1]
                unique_queries.add(query_key)
                by_source[source] += 1
                hit_rows.append({
                    "query_structure": query_key,
                    "target": target,
                    "target_source": source,
                    "pident_percent": float(pident),
                    "query_coverage": float(qcov),
                    "target_coverage": float(tcov),
                    "alignment_length": int(alnlen),
                    "evalue": evalue,
                    "bits": float(bits),
                })
    write_csv(args.output_hits, hit_rows)
    report = {
        "pass": len(unique_queries) == 0,
        "stage": "complete",
        "rule": {
            "minimum_identity_fraction": args.min_identity,
            "minimum_coverage_fraction": args.min_coverage,
            "coverage_mode": 0,
        },
        "query_records": query_count,
        "test_records": source_counts,
        "hit_records": len(hit_rows),
        "leaking_unique_train_structures": len(unique_queries),
        "hit_records_by_source": dict(sorted(by_source.items())),
        "command": command,
    }
    write_json(args.output_summary, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-source", type=parse_source, action="append", required=True)
    parser.add_argument("--structure-root", action="append", default=[])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-hits", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--min-identity", type=float, default=0.30)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    args = parser.parse_args()
    if args.min_identity != 0.30 or args.min_coverage != 0.80:
        raise SystemExit("Publication audit is fixed at identity 0.30 and coverage 0.80")
    report = run_audit(args)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Leakage audit failed: nonzero train structures hit a test source")


if __name__ == "__main__":
    main()
