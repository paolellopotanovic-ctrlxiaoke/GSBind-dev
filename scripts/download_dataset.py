from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

try:
    from download_checkpoint import sha256
except ModuleNotFoundError:  # imported as scripts.download_dataset in tests or notebooks
    from scripts.download_checkpoint import sha256


def download_verified(url: str, destination: Path, expected_size: int, expected_hash: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual_hash = sha256(temporary)
    actual_size = temporary.stat().st_size
    if actual_hash != expected_hash or actual_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Dataset verification failed: file={destination.name}, "
            f"bytes={actual_size}, sha256={actual_hash}"
        )
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a registered M4 dataset manifest.")
    parser.add_argument("name")
    parser.add_argument("--registry", default="assets/datasets.json")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--url", default="", help="Authorized temporary mirror override.")
    parser.add_argument("--include-audit", action="store_true")
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    try:
        record = registry["datasets"][args.name]
    except KeyError as exc:
        raise SystemExit(f"Unknown dataset: {args.name}") from exc
    url = args.url or record.get("manifest_download_url")
    if not url:
        raise SystemExit(
            "No public dataset asset is registered yet. Use --url for an authorized mirror "
            "or place the exact manifest manually and verify its SHA-256."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / record["manifest_filename"]
    download_verified(
        url, destination, int(record["manifest_bytes"]), record["manifest_sha256"]
    )
    print(destination.resolve())
    if args.include_audit:
        audit_url = record.get("audit_bundle_url")
        if not audit_url:
            raise SystemExit("No audit bundle is registered for this dataset")
        audit = output_dir / record["audit_bundle_filename"]
        download_verified(
            audit_url,
            audit,
            int(record["audit_bundle_bytes"]),
            record["audit_bundle_sha256"],
        )
        print(audit.resolve())


if __name__ == "__main__":
    main()
