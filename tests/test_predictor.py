from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from m4.inputs import inference_manifest_row
from m4.predictor import OUTPUT_FIELDS, _validate_rows


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "inference" / "1CRN.pdb"


def test_prediction_schema_is_per_residue_and_metric_free() -> None:
    assert "interface_probability" in OUTPUT_FIELDS
    assert "residue_number" in OUTPUT_FIELDS
    assert not any(field in OUTPUT_FIELDS for field in ("label", "aupr", "auroc"))


def test_predictor_rejects_duplicate_cases() -> None:
    row = inference_manifest_row(EXAMPLE, "same")
    with pytest.raises(ValueError, match="unique"):
        _validate_rows([row, dict(row)])


def test_predictor_rejects_residue_order_mismatch() -> None:
    row = inference_manifest_row(EXAMPLE, "1CRN")
    row["input_residue_ids"] = row["input_residue_ids"].replace("1", "999", 1)
    with pytest.raises(ValueError, match="Residue order mismatch"):
        _validate_rows([row])
