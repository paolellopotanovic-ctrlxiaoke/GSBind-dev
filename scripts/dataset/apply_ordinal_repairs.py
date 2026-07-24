#!/usr/bin/env python3
"""Apply a published, checksum-guarded residue-ordinal repair table."""

from __future__ import annotations

import argparse
import hashlib
import json

from _common import read_csv, sha256, split_field, write_csv, write_json


def value_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def apply_repairs(
    rows: list[dict[str, str]], repairs: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    by_case = {row.get("apex_case_id", ""): index for index, row in enumerate(rows)}
    if len(by_case) != len(rows) or "" in by_case:
        raise ValueError("Manifest case IDs must be non-empty and unique")
    seen_repairs: set[str] = set()
    log: list[dict[str, object]] = []
    output = [dict(row) for row in rows]
    for repair in repairs:
        case_id = repair.get("apex_case_id", "")
        if not case_id or case_id in seen_repairs or case_id not in by_case:
            raise ValueError(f"Invalid or duplicate repair case: {case_id!r}")
        seen_repairs.add(case_id)
        row = output[by_case[case_id]]
        old_value = row.get("input_residue_ids", "")
        expected_hash = repair.get("expected_old_input_residue_ids_sha256", "")
        new_value = repair.get("input_residue_ids", "")
        if not expected_hash or value_hash(old_value) != expected_hash:
            raise ValueError(f"Old ordinal checksum mismatch for {case_id}")
        labels_before = row.get("labels", "")
        mask_before = row.get("label_mask", "")
        if len(split_field(new_value)) != len(split_field(labels_before, int)):
            raise ValueError(f"Repair length does not match labels for {case_id}")
        row["input_residue_ids"] = new_value
        if row.get("labels", "") != labels_before or row.get("label_mask", "") != mask_before:
            raise RuntimeError(f"Repair unexpectedly changed labels for {case_id}")
        log.append({
            "apex_case_id": case_id,
            "old_input_residue_ids_sha256": value_hash(old_value),
            "new_input_residue_ids_sha256": value_hash(new_value),
            "residues": len(split_field(new_value)),
            "labels_unchanged": True,
            "label_mask_unchanged": True,
        })
    return output, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repairs", required=True, help="JSON array of checksum-guarded repairs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--expected-repairs", type=int, default=2)
    args = parser.parse_args()
    repairs = json.loads(open(args.repairs, encoding="utf-8").read())
    if not isinstance(repairs, list) or len(repairs) != args.expected_repairs:
        raise SystemExit(f"Expected exactly {args.expected_repairs} repair records")
    rows, log = apply_repairs(read_csv(args.manifest), repairs)
    write_csv(args.output, rows)
    report = {
        "input_manifest_sha256": sha256(args.manifest),
        "repair_table_sha256": sha256(args.repairs),
        "output_manifest_sha256": sha256(args.output),
        "repairs": log,
    }
    write_json(args.audit, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
