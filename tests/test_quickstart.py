from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_1crn_example", ROOT / "scripts" / "run_1crn_example.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_quickstart_plan_covers_download_four_caches_and_forward(tmp_path: Path) -> None:
    checkpoint = tmp_path / MODULE.CHECKPOINT_FILENAME
    commands = MODULE.build_command_plan(
        ROOT,
        tmp_path / "work",
        ROOT / "examples" / "inference" / "1CRN.pdb",
        checkpoint,
        download_checkpoint=True,
    )

    assert len(commands) == 8
    flattened = [" ".join(command) for command in commands]
    assert "download_checkpoint.py" in flattened[0]
    assert "validate_checkpoint.py" in flattened[1]
    assert "prepare-input" in flattened[2]
    assert "build-tensor-cache" in flattened[3]
    assert "build-graph-cache" in flattened[4]
    assert "build-surface-cache" in flattened[5]
    assert "build-static-cache" in flattened[6]
    assert "inference.py" in flattened[7]
    assert "--s1-static-geometry-cache-dir" in commands[-1]


def test_quickstart_plan_can_use_existing_checkpoint(tmp_path: Path) -> None:
    commands = MODULE.build_command_plan(
        ROOT,
        tmp_path / "work",
        ROOT / "examples" / "inference" / "1CRN.pdb",
        tmp_path / MODULE.CHECKPOINT_FILENAME,
        download_checkpoint=False,
    )

    assert len(commands) == 7
    assert "validate_checkpoint.py" in " ".join(commands[0])


def test_quickstart_plan_propagates_case_and_state(tmp_path: Path) -> None:
    commands = MODULE.build_command_plan(
        ROOT,
        tmp_path / "work",
        tmp_path / "receptor.pdb",
        tmp_path / MODULE.CHECKPOINT_FILENAME,
        download_checkpoint=False,
        case_id="TARGET_A",
        input_state="holo",
    )

    prepare = commands[1]
    surface = commands[4]
    assert prepare[prepare.index("--case-id") + 1] == "TARGET_A"
    assert prepare[prepare.index("--input-state") + 1] == "holo"
    assert surface[surface.index("--states") + 1] == "holo"
