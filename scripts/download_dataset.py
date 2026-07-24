from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from download_checkpoint import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a registered M4 dataset manifest.")
    parser.add_argument("name")
    parser.add_argument("--registry", default="assets/datasets.json")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--url", default="", help="Authorized temporary mirror override.")
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
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual_hash = sha256(temporary)
    if actual_hash != record["manifest_sha256"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Dataset verification failed: sha256={actual_hash}")
    temporary.replace(destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
