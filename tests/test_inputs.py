from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")

from m4.inputs import residue_records_from_pdb, write_inference_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "inference" / "1CRN.pdb"


def test_1crn_has_stable_46_residue_order() -> None:
    records = residue_records_from_pdb(EXAMPLE)
    assert len(records) == 46
    assert records[0]["residue_number"] == 1
    assert records[0]["amino_acid"] == "T"
    assert records[-1]["residue_number"] == 46
    assert records[-1]["amino_acid"] == "N"


def test_unlabeled_manifest_uses_technical_placeholders(tmp_path: Path) -> None:
    output = write_inference_manifest([EXAMPLE], tmp_path / "manifest.csv", ["1CRN"])
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["apex_case_id"] == "1CRN"
    assert row["num_residues"] == "46"
    assert set(row["labels"].split(";")) == {"0"}
    assert set(row["label_mask"].split(";")) == {"1"}
    assert row["tensor_cache_labels_from_manifest"] == "true"


def test_inference_case_ids_must_be_unique(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        write_inference_manifest([EXAMPLE, EXAMPLE], tmp_path / "manifest.csv", ["same", "same"])
