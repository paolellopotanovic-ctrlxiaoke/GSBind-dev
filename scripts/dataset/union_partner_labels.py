#!/usr/bin/env python3
"""Union labels from all observed partners for each unique receptor holo structure."""

from __future__ import annotations

import argparse
from collections import defaultdict

from _common import read_csv, split_field, structure_key, write_csv, write_json


def union_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("input_state") not in {"", "holo"}:
            raise ValueError(f"Non-holo row cannot enter holo union: {row.get('apex_case_id')}")
        groups[structure_key(row["receptor_pdb_path"])].append(row)

    output: list[dict[str, object]] = []
    partner_events = 0
    for key in sorted(groups):
        members = groups[key]
        reference_ids = split_field(members[0].get("input_residue_ids", ""))
        if not reference_ids:
            raise ValueError(f"Missing residue ordinals for {key}")
        union_labels = [0] * len(reference_ids)
        union_mask = [0] * len(reference_ids)
        systems: list[str] = []
        partners: list[str] = []
        for member in members:
            ids = split_field(member.get("input_residue_ids", ""))
            labels = split_field(member.get("labels", ""), int)
            mask = split_field(member.get("label_mask", ""), int)
            if ids != reference_ids or len(labels) != len(ids) or len(mask) != len(ids):
                raise ValueError(f"Residue ordinal mismatch within receptor group {key}")
            union_labels = [int(a or (y and m)) for a, y, m in zip(union_labels, labels, mask)]
            union_mask = [int(a or m) for a, m in zip(union_mask, mask)]
            systems.append(member.get("system_id", ""))
            partners.append(member.get("source_partner_pdb", member.get("partner_pdb_path", "")))
        row: dict[str, object] = dict(members[0])
        row["apex_case_id"] = f"holo_union:{key}"
        row["receptor_case_id"] = row["apex_case_id"]
        row["input_state"] = "holo"
        row["partner_sequence"] = ""
        row["labels"] = ";".join(map(str, union_labels))
        row["label_mask"] = ";".join(map(str, union_mask))
        row["num_residues"] = len(reference_ids)
        row["num_positive_labels"] = sum(union_labels)
        row["mask_coverage"] = f"{sum(union_mask) / len(union_mask):.6f}"
        row["union_partner_count"] = len(members)
        row["union_system_ids"] = ";".join(sorted(set(filter(None, systems))))
        row["union_partner_structures"] = ";".join(sorted(set(filter(None, partners))))
        row["label_source"] = "all_partner_union_of_holo_heavy_atom_contacts_within_5A"
        output.append(row)
        partner_events += len(members)
    summary = {
        "input_rows": len(rows),
        "output_rows": len(output),
        "unique_receptor_structures": len(groups),
        "partner_events": partner_events,
        "multi_partner_receptors": sum(len(group) > 1 for group in groups.values()),
        "max_partners_per_receptor": max((len(group) for group in groups.values()), default=0),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    rows, summary = union_rows(read_csv(args.input))
    write_csv(args.output, rows)
    write_json(args.summary, summary)
    print(summary)


if __name__ == "__main__":
    main()
