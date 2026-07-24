from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a registered M4 checkpoint.")
    parser.add_argument("name")
    parser.add_argument("--registry", default="assets/checkpoints.json")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--url", default="", help="Authorized temporary mirror override.")
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    try:
        record = registry["checkpoints"][args.name]
    except KeyError as exc:
        raise SystemExit(f"Unknown checkpoint: {args.name}") from exc
    url = args.url or record.get("download_url")
    if not url:
        raise SystemExit(
            "No public release asset is registered yet. Use --url for an authorized mirror "
            "or place the file manually and run scripts/validate_checkpoint.py."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / record["filename"]
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    actual_size = temporary.stat().st_size
    actual_hash = sha256(temporary)
    if actual_size != int(record["bytes"]) or actual_hash != record["sha256"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Checkpoint verification failed: bytes={actual_size}, sha256={actual_hash}")
    temporary.replace(destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
