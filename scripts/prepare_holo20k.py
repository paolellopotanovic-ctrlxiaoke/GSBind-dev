from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PINDER_PDB_URL = "https://storage.googleapis.com/pinder/2024-02/pdbs"
MANIFEST_SHA256 = "65b1df43dd4e00420dea8f59c8bbdc03d5a8fb6b33ed511fc469768ec0799d0d"
OVERFLOW_PDB = "6z3m__S1_Q6NW40-L.pdb"
OVERFLOW_BEFORE_SHA256 = "be54687c9aa1f535e2c3779c95480b98271e0a56a360c1de8ffeed7af4f9dd33"
OVERFLOW_AFTER_SHA256 = "9e2e9b3d8be4e9f73f695c2edf53f12def562bc8f4f650077af284a5125b74be"
OVERFLOW = re.compile(r"^(?P<occupancy>.{6})(?P<bfactor>\d{4}\.\d{2})(?P<suffix>.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_name(row: dict[str, str]) -> str:
    name = row.get("holo_pdb", "").strip()
    if not name:
        name = Path(row.get("receptor_pdb_path", "")).name
    if not name or Path(name).name != name or not name.endswith(".pdb"):
        raise ValueError(f"Invalid Pinder structure name: {name!r}")
    return name


def download_one(name: str, pdb_dir: Path, base_url: str, timeout: int, retries: int) -> str:
    destination = pdb_dir / name
    if destination.is_file() and destination.stat().st_size > 0:
        return "existing"
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{base_url.rstrip('/')}/{name}"
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Gaussian-PPBS/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if not data or not any(token in data[:8192] for token in (b"ATOM  ", b"HETATM")):
                raise RuntimeError("downloaded file is empty or does not contain PDB atoms")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return "downloaded"
        except Exception as exc:  # pragma: no cover - network errors depend on the mirror
            error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to download {name}: {error}")


def repair_overflow_bfactor(path: Path) -> dict[str, object]:
    before = sha256(path)
    if before == OVERFLOW_AFTER_SHA256:
        return {"status": "already_repaired", "before_sha256": before, "after_sha256": before}
    if before != OVERFLOW_BEFORE_SHA256:
        raise RuntimeError(
            f"Refusing to repair unexpected {path.name}: sha256={before}; "
            f"expected {OVERFLOW_BEFORE_SHA256}"
        )

    changes: list[dict[str, object]] = []
    output: list[str] = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.rstrip("\n")
            if raw.startswith(("ATOM  ", "HETATM")) and len(raw) >= 67:
                match = OVERFLOW.match(raw[54:])
                if match:
                    original = float(match.group("bfactor"))
                    raw = raw[:60] + "999.99" + match.group("suffix")
                    changes.append(
                        {"line": line_number, "original_bfactor": original, "new_bfactor": 999.99}
                    )
            output.append(raw + ("\n" if line.endswith("\n") else ""))
    if len(changes) != 14:
        raise RuntimeError(f"Expected 14 overflowing B-factors, observed {len(changes)}")
    temporary = path.with_suffix(path.suffix + ".repair.part")
    temporary.write_text("".join(output), encoding="ascii", newline="")
    after = sha256(temporary)
    if after != OVERFLOW_AFTER_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Repaired PDB hash mismatch: {after}")
    os.replace(temporary, path)
    return {
        "status": "repaired",
        "before_sha256": before,
        "after_sha256": after,
        "rule": "clip four-digit PDB B-factors to 999.99",
        "changes": changes,
    }


def localize_manifest(
    source: Path,
    destination: Path,
    pdb_dir: Path,
) -> tuple[int, int]:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "receptor_pdb_path" not in reader.fieldnames:
            raise RuntimeError("Manifest is missing receptor_pdb_path")
        rows = list(reader)
        fieldnames = reader.fieldnames
    for row in rows:
        row["receptor_pdb_path"] = str((pdb_dir / structure_name(row)).resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)
    return len(rows), len({structure_name(row) for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download exact Holo20k Pinder structures and create a runnable local manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pdb-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--report", default="", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-url", default=PINDER_PDB_URL)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    source_hash = sha256(args.manifest)
    if source_hash != MANIFEST_SHA256:
        raise SystemExit(f"Exact manifest SHA-256 mismatch: {source_hash}")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = sorted({structure_name(row) for row in rows})
    if len(rows) != 20000 or len(names) != 20000:
        raise SystemExit(f"Expected 20000 rows and structures; found rows={len(rows)}, structures={len(names)}")

    args.pdb_dir.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "existing": 0}
    failures: list[str] = []
    if not args.no_download:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one, name, args.pdb_dir, args.base_url, args.timeout, args.retries
                ): name
                for name in names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    counts[future.result()] += 1
                except Exception as exc:
                    failures.append(f"{name}: {exc}")
    missing = [name for name in names if not (args.pdb_dir / name).is_file()]
    if failures or missing:
        raise SystemExit(
            f"Structure hydration failed: failures={len(failures)}, missing={len(missing)}; "
            f"first={failures[:3] or missing[:3]}"
        )

    repair = repair_overflow_bfactor(args.pdb_dir / OVERFLOW_PDB)
    row_count, structure_count = localize_manifest(
        args.manifest, args.output_manifest, args.pdb_dir
    )
    report = {
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": source_hash,
        "localized_manifest": str(args.output_manifest.resolve()),
        "localized_manifest_sha256": sha256(args.output_manifest),
        "rows": row_count,
        "unique_structures": structure_count,
        "pdb_dir": str(args.pdb_dir.resolve()),
        "downloads": counts,
        "pdb_repair": {"structure": OVERFLOW_PDB, **repair},
    }
    report_path = args.report or args.output_manifest.with_suffix(".prepare.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
