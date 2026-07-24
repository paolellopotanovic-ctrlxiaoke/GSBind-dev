from __future__ import annotations

import hashlib
from pathlib import Path


VALIDATED_SHA256 = {
    "model/apex_dataset.py": "8616e594d326f36776109b974a7509d10accbc4c4a45f67821e3fa74a0076561",
    "model/apex_s1_surface_field/model.py": "394a67d4e42a41dc104c9fbb1a52b58489110e783f43c68a5fd737464a3359be",
    "model/apex_s1_surface_field/train_s1_surface_field.py": "98aff4919a10d5936e662b09f80eab51305b879d166d39cefedd96032f2a5210",
    "model/apex_s1_surface_field/eval_s1_surface_field.py": "977a33e41dd9ad1bf4e771abe1163acdef8cf34f67709eabbc92f1ac96e48aaf",
    "model/apex_s1_surface_field/semantic_patch.py": "bfb7d3397159e10c80fad4f198a13563f07c9d8f169de3c734f5b3ba6d1f131a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_validated_sources(repo_root: Path) -> dict[str, tuple[str, str]]:
    base = repo_root / "src" / "m4" / "legacy_repo"
    return {
        relative: (expected, sha256(base / relative))
        for relative, expected in VALIDATED_SHA256.items()
    }
