from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_registry_has_immutable_identity() -> None:
    registry = json.loads((ROOT / "assets" / "checkpoints.json").read_text(encoding="utf-8"))
    checkpoint = registry["checkpoints"]["m4_holo20k_v9_clean_epoch4"]
    assert checkpoint["bytes"] == 46437755
    assert len(checkpoint["sha256"]) == 64
    int(checkpoint["sha256"], 16)
    assert checkpoint["strict_load_required"] is True


def test_dataset_registry_pins_exact_holo20k_v9() -> None:
    registry = json.loads((ROOT / "assets" / "datasets.json").read_text(encoding="utf-8"))
    dataset = registry["datasets"]["holo20k_v9_mmseqclean"]
    assert dataset["rows"] == dataset["unique_case_ids"] == dataset["unique_structure_paths"] == 20000
    assert dataset["clusters"] == 9812
    assert dataset["cluster_cap"] == 20
    assert dataset["leakage_hits"] == 0
    assert dataset["manifest_bytes"] == 46063708
    assert dataset["manifest_download_url"].endswith(dataset["manifest_filename"])
    assert dataset["audit_bundle_url"].endswith(dataset["audit_bundle_filename"])
    assert dataset["audit_bundle_bytes"] == 2649357
    assert len(dataset["audit_bundle_sha256"]) == hashlib.sha256().digest_size * 2
    assert dataset["release_status"] == "published_github_release_v0.1.0"
    assert len(dataset["manifest_sha256"]) == hashlib.sha256().digest_size * 2


def test_example_pdb_hash_is_pinned() -> None:
    actual = hashlib.sha256((ROOT / "examples" / "inference" / "1CRN.pdb").read_bytes()).hexdigest()
    assert actual == "42199a30a0701864a2a5cc76cd7f35cc544cd0e65fbcf63e03c166543249b811"
