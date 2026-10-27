import pandas as pd
import pytest
from dataclasses import replace
from pathlib import Path

from gaussian_direct.config import CheckpointSelectionConfig, ExperimentConfig
from gaussian_direct.validation import (
    PERFORMANCE_METRICS,
    composite_scores,
    discover_checkpoints,
    select_checkpoints,
    validate_manifest_contract,
)


def test_config_includes_gpsfun_selection_contract():
    config = ExperimentConfig.from_file("configs/main.json")
    selection = config.training.checkpoint_selection
    assert selection.enabled
    assert selection.expected_cases == 375
    assert selection.expected_valid_residues == 77_978
    assert selection.expected_positive_residues == 11_350
    assert selection.epoch_min == 0
    assert selection.epoch_max == 19
    assert len(PERFORMANCE_METRICS) == 16
    assert not any("loss" in metric or "brier" in metric for metric in PERFORMANCE_METRICS)


def test_historical_composite_reproduces_epoch11():
    frame = pd.read_csv("tests/fixtures/gpsfun_full375/composite_summary.csv")
    rows = composite_scores(frame[["epoch", *PERFORMANCE_METRICS]].to_dict("records"))
    selected = max(rows, key=lambda row: row["family_balanced_z"])
    assert selected["epoch"] == 11
    assert selected["family_balanced_z"] == pytest.approx(0.9788990205747575)
    historical = frame.set_index("epoch").loc[11]
    assert selected["all16_equal_z"] == pytest.approx(float(historical["all16_equal_z"]))


def test_manifest_contract_rejects_label_count_mismatch():
    selection = CheckpointSelectionConfig(
        expected_cases=1,
        expected_valid_residues=3,
        expected_positive_residues=1,
    )
    row = {
        "apex_case_id": "case",
        "labels": "1;0;1",
        "label_mask": "1;1;0",
    }
    with pytest.raises(ValueError, match="valid residues"):
        validate_manifest_contract([row], selection)


def test_checkpoint_epoch_contract(tmp_path):
    selection = CheckpointSelectionConfig(epoch_min=0, epoch_max=1)
    for epoch in range(2):
        (tmp_path / f"model_epoch_{epoch:02d}.pt").touch()
    checkpoints = discover_checkpoints(tmp_path, selection)
    assert sorted(checkpoints) == [0, 1]
    (tmp_path / "model_epoch_02.pt").touch()
    with pytest.raises(ValueError, match="extra=\\[2\\]"):
        discover_checkpoints(tmp_path, selection)


def test_selector_writes_complete_evidence(tmp_path, monkeypatch):
    import gaussian_direct.validation as validation

    for epoch in range(2):
        (tmp_path / f"model_epoch_{epoch:02d}.pt").write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("apex_case_id\nexample\n", encoding="utf-8")
    base = ExperimentConfig()
    selection = replace(
        base.training.checkpoint_selection,
        enabled=True,
        epoch_max=1,
        manifest=manifest,
        output_dir=tmp_path / "selection",
    )
    config = base.with_overrides(
        training=replace(
            base.training,
            output_dir=tmp_path,
            optimizer=replace(base.training.optimizer, epochs=2),
            checkpoint_selection=selection,
        )
    )
    metrics = {
        "epoch": 0,
        **{metric: 0.5 for metric in PERFORMANCE_METRICS},
    }
    later_metrics = {**metrics, "epoch": 1, "macro_recall": 0.6}

    def fake_evaluate(checkpoint, rows, settings, output_dir):
        del rows, settings, output_dir
        return later_metrics if Path(checkpoint).name == "model_epoch_01.pt" else metrics

    monkeypatch.setattr(validation, "validate_paths", lambda settings: None)
    monkeypatch.setattr(validation, "read_csv_rows", lambda path: [{"apex_case_id": "case"}])
    monkeypatch.setattr(
        validation,
        "validate_manifest_contract",
        lambda rows, settings: (2, 1),
    )
    monkeypatch.setattr(validation, "evaluate_checkpoint", fake_evaluate)
    report = select_checkpoints(config)

    assert report["selected_epoch"] == 1
    assert (tmp_path / "selection/composite_summary.csv").is_file()
    assert (tmp_path / "selection/selected_checkpoint.json").is_file()
    assert (tmp_path / "selection/selection_contract.json").is_file()
    assert (tmp_path / "selection/VALIDATION_SELECTION_COMPLETE").is_file()
