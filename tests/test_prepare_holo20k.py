from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from scripts.download_dataset import download_verified
from scripts.prepare_holo20k import localize_manifest, structure_name


def test_structure_name_prefers_registered_holo_name() -> None:
    row = {
        "holo_pdb": "1abc__A1_P12345-R.pdb",
        "receptor_pdb_path": "/old/server/path/wrong.pdb",
    }
    assert structure_name(row) == "1abc__A1_P12345-R.pdb"


def test_localize_manifest_rewrites_only_structure_path(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    fields = ["apex_case_id", "holo_pdb", "receptor_pdb_path", "labels"]
    original = {
        "apex_case_id": "case-1",
        "holo_pdb": "1abc__A1_P12345-R.pdb",
        "receptor_pdb_path": "/old/server/path/1abc__A1_P12345-R.pdb",
        "labels": "1;0;1",
    }
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(original)

    pdb_dir = tmp_path / "pdbs"
    destination = tmp_path / "localized.csv"
    rows, structures = localize_manifest(source, destination, pdb_dir)
    with destination.open(newline="", encoding="utf-8") as handle:
        localized = next(csv.DictReader(handle))

    assert rows == structures == 1
    assert localized["receptor_pdb_path"] == str(
        (pdb_dir / "1abc__A1_P12345-R.pdb").resolve()
    )
    for field in ("apex_case_id", "holo_pdb", "labels"):
        assert localized[field] == original[field]


def test_download_verified_accepts_only_registered_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"case,label\nA,1\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "downloaded.csv"

    download_verified(source.as_uri(), destination, source.stat().st_size, expected)
    assert destination.read_bytes() == source.read_bytes()

    with pytest.raises(SystemExit, match="verification failed"):
        download_verified(source.as_uri(), destination, source.stat().st_size, "0" * 64)
