from pathlib import Path

from m4.cli import COMMAND_MODULES


ROOT = Path(__file__).resolve().parents[1]


def test_release_has_one_scientific_source_tree() -> None:
    assert not (ROOT / "src" / "m4" / "legacy_repo").exists()
    assert not (ROOT / "src" / "m4" / "_bootstrap.py").exists()
    assert "train-legacy" not in COMMAND_MODULES


def test_public_source_has_no_removed_runtime_imports() -> None:
    forbidden = (
        "legacy_repo",
        "m4._bootstrap",
        "from apex_",
        "import apex_",
        "from train_apex",
        "import train_apex",
    )
    violations = []
    for path in sorted((ROOT / "src" / "m4").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert violations == []
