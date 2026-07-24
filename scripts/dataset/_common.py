from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "PYL": "O", "SEC": "U", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_json(path: str | Path, payload: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_field(value: object, cast=str) -> list:
    if value is None or str(value) == "":
        return []
    return [cast(item) for item in str(value).split(";")]


def stable_digest(*values: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, values)).encode("utf-8")).hexdigest()


def structure_key(path: object) -> str:
    return Path(str(path)).name.lower()


def resolve_structure(path: object, roots: Iterable[str | Path] = ()) -> Path:
    candidate = Path(str(path)).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    for root in roots:
        for directory in (Path(root), Path(root) / "pdbs", Path(root) / "test_set_pdbs"):
            resolved = directory / candidate.name
            if resolved.is_file():
                return resolved.resolve()
    raise FileNotFoundError(str(path))


def pdb_sequence(path: str | Path) -> str:
    residues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or len(line) < 27:
                continue
            atom = line[12:16].strip()
            altloc = line[16:17]
            if atom != "CA" or altloc not in {" ", "A"}:
                continue
            key = (line[21:22], line[22:26], line[26:27])
            if key in seen:
                continue
            seen.add(key)
            residues.append(AA3_TO_AA1.get(line[17:20].strip().upper(), "X"))
    return "".join(residues)


def row_length(row: dict[str, str]) -> int:
    value = str(row.get("num_residues", "")).strip()
    if value:
        return int(float(value))
    return len(split_field(row.get("labels", ""), int))


def row_positive_rate(row: dict[str, str]) -> float:
    labels = split_field(row.get("labels", ""), int)
    mask = split_field(row.get("label_mask", ""), int)
    if labels and mask and len(labels) == len(mask):
        valid = sum(mask)
        return sum(y * m for y, m in zip(labels, mask)) / valid if valid else 0.0
    length = row_length(row)
    return float(row.get("num_positive_labels", 0) or 0) / length if length else 0.0


def length_bucket(length: int, edges: Sequence[int]) -> str:
    if not edges or edges[0] != 0 or sorted(set(edges)) != list(edges):
        raise ValueError("Length bucket edges must be sorted, unique, and begin at zero")
    for low, high in zip(edges, edges[1:]):
        if low <= length < high:
            return f"{low}-{high - 1}"
    return f">={edges[-1]}"
