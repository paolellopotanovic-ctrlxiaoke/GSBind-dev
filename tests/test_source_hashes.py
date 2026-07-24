from pathlib import Path

from utils.reproducibility import verify_validated_sources


def test_validated_source_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    mismatches = {
        name: values
        for name, values in verify_validated_sources(root).items()
        if values[0] != values[1]
    }
    assert mismatches == {}
