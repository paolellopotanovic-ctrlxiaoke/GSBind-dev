from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))

from audit_distribution import audit
from apply_ordinal_repairs import apply_repairs, value_hash
from build_holo20k_v9 import build_v9
from union_partner_labels import union_rows


def event(case: str, path: str, labels: str, cluster: str = "c1") -> dict[str, str]:
    count = len(labels.split(";"))
    return {
        "apex_case_id": case,
        "system_id": case,
        "input_state": "holo",
        "receptor_pdb_path": path,
        "input_residue_ids": ";".join(str(index + 1) for index in range(count)),
        "labels": labels,
        "label_mask": ";".join("1" for _ in range(count)),
        "num_residues": str(count),
        "num_positive_labels": str(sum(map(int, labels.split(";")))),
        "cluster_id": cluster,
    }


def test_partner_union_is_residuewise_or() -> None:
    rows, summary = union_rows([
        event("a", "/x/receptor.pdb", "1;0;0"),
        event("b", "/y/receptor.pdb", "0;0;1"),
    ])
    assert len(rows) == 1
    assert rows[0]["labels"] == "1;0;1"
    assert rows[0]["union_partner_count"] == 2
    assert summary["multi_partner_receptors"] == 1


def test_partner_union_rejects_ordinal_mismatch() -> None:
    first = event("a", "receptor.pdb", "1;0")
    second = event("b", "receptor.pdb", "0;1")
    second["input_residue_ids"] = "1;3"
    with pytest.raises(ValueError, match="ordinal mismatch"):
        union_rows([first, second])


def test_v9_replacement_preserves_position_and_cap() -> None:
    base = [event("a", "a.pdb", "1;0", "c1"), event("b", "b.pdb", "0;1", "c2")]
    reserve = [event("r", "r.pdb", "1;0", "c3")]
    output, replacements, summary = build_v9(base, reserve, {"a.pdb"}, 1, edges=(0, 10))
    assert output[0]["apex_case_id"] == "r"
    assert output[1]["apex_case_id"] == "b"
    assert replacements[0]["row_index"] == 0
    assert summary["observed_cluster_cap"] == 1


def test_distribution_counts_residues_not_samples() -> None:
    report = audit([event("a", "a.pdb", "1;0"), event("b", "b.pdb", "1;1;0;0", "c2")], edges=(0, 10))
    assert report["positive"]["residues"] == 3
    assert report["positive"]["valid_residues"] == 6
    assert report["positive"]["global_rate"] == 0.5


def test_ordinal_repair_is_checksum_guarded_and_label_preserving() -> None:
    row = event("a", "a.pdb", "1;0")
    old = row["input_residue_ids"]
    output, log = apply_repairs([row], [{
        "apex_case_id": "a",
        "expected_old_input_residue_ids_sha256": value_hash(old),
        "input_residue_ids": "10;11",
    }])
    assert output[0]["input_residue_ids"] == "10;11"
    assert output[0]["labels"] == "1;0"
    assert log[0]["labels_unchanged"] is True
