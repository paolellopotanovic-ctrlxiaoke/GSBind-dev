from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .data.manifest import raw_residue_ids_and_sequence, residue_uid
from .data.structure import clean_structure, res3to1
from .data.structure_io import read_pdb


MANIFEST_FIELDS = (
    "apex_case_id",
    "system_id",
    "input_state",
    "receptor_pdb_path",
    "partner_sequence",
    "input_residue_ids",
    "labels",
    "label_mask",
    "num_residues",
    "num_positive_labels",
    "mask_coverage",
    "tensor_cache_labels_from_manifest",
)


def residue_records_from_pdb(path: str | Path) -> list[dict[str, object]]:
    path = Path(path).expanduser().resolve()
    structure = read_pdb(str(path))
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    records: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for index, resid in enumerate(structure["resid"]):
        if str(structure["het_flag"][index]) != "A":
            continue
        chain = str(structure["chain_name"][index]).split(":")[0]
        icode = str(icodes[index] or "").strip()
        key = (chain, int(resid), icode)
        if key in seen:
            continue
        seen.add(key)
        residue_name = str(structure["resname"][index]).strip().upper()
        records.append(
            {
                "chain_id": chain,
                "residue_number": int(resid),
                "insertion_code": icode,
                "residue_id": residue_uid(resid, icode),
                "amino_acid": res3to1.get(residue_name, "X"),
            }
        )

    raw_ids, raw_sequence = raw_residue_ids_and_sequence(str(path))
    clean = clean_structure(read_pdb(str(path)))
    encoded_count = len({
        (str(chain).split(":")[0], int(resid))
        for chain, resid in zip(clean["chain_name"].tolist(), clean["resid"].tolist())
    })
    record_ids = [str(record["residue_id"]) for record in records]
    record_sequence = [str(record["amino_acid"]) for record in records]
    if record_ids != raw_ids or record_sequence != raw_sequence or len(records) != encoded_count:
        raise ValueError(
            f"PDB residue order is not representable by the strict M4 encoder: {path} "
            f"(raw={len(records)}, encoded={encoded_count})."
        )
    if not records:
        raise ValueError(f"No protein residues found in {path}")
    return records


def inference_manifest_row(
    pdb_path: str | Path,
    case_id: str,
    input_state: str = "apo",
) -> dict[str, str]:
    path = Path(pdb_path).expanduser().resolve()
    records = residue_records_from_pdb(path)
    count = len(records)
    return {
        "apex_case_id": case_id,
        "system_id": case_id,
        "input_state": input_state,
        "receptor_pdb_path": str(path),
        "partner_sequence": "",
        "input_residue_ids": ";".join(str(record["residue_id"]) for record in records),
        # Technical placeholders make cache keys deterministic. predict never treats them as labels.
        "labels": ";".join("0" for _ in records),
        "label_mask": ";".join("1" for _ in records),
        "num_residues": str(count),
        "num_positive_labels": "0",
        "mask_coverage": "1.0",
        "tensor_cache_labels_from_manifest": "true",
    }


def write_inference_manifest(
    pdb_paths: list[str | Path],
    output: str | Path,
    case_ids: list[str] | None = None,
    input_state: str = "apo",
) -> Path:
    if case_ids is not None and len(case_ids) != len(pdb_paths):
        raise ValueError("--case-id count must equal --pdb count")
    rows = []
    for index, pdb_path in enumerate(pdb_paths):
        case_id = case_ids[index] if case_ids else Path(pdb_path).stem
        rows.append(inference_manifest_row(pdb_path, case_id, input_state=input_state))
    if len({row["apex_case_id"] for row in rows}) != len(rows):
        raise ValueError("Inference case IDs must be unique")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a strict M4 cache manifest from unlabeled receptor-only PDB files."
    )
    parser.add_argument("--pdb", action="append", required=True, help="Repeat for multiple inputs.")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--input-state", default="apo", choices=("apo", "af2", "holo"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = write_inference_manifest(
        args.pdb,
        args.output,
        case_ids=args.case_id or None,
        input_state=args.input_state,
    )
    print(path)


if __name__ == "__main__":
    main()
