import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_external_oracle_equivalence_evidence() -> None:
    report = json.loads(
        (ROOT / "docs" / "validation" / "final_external_oracle_equivalence.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["pass"] is True
    assert report["state_tensors"] == 791
    assert report["parameters"] == 4_360_516
    assert report["eval"]["worst"]["max_abs"] == 0.0
    assert report["train"]["worst"]["max_abs"] == 0.0
    assert report["loss"]["worst"]["max_abs"] == 0.0
    assert report["gradient"]["worst"]["max_abs"] <= report["gradient"]["atol"]
    assert report["optimizer_step"]["worst"]["max_abs"] <= report["optimizer_step"]["atol"]


def test_final_evaluation_smoke_evidence() -> None:
    report = json.loads(
        (ROOT / "docs" / "validation" / "final_eval_smoke.json").read_text(encoding="utf-8")
    )
    metrics = report["metrics"]
    assert report["num_manifest_rows"] == 2
    assert metrics["num_cases"] == 2
    assert metrics["valid_residues"] == 611
    assert metrics["skipped_examples"] == 0
    assert metrics["skipped_batches"] == 0
