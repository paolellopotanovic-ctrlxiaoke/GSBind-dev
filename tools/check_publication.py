"""Verify a public checkout; optionally require and hash all external data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_inventory(path):
    missing = []
    mismatched = []
    verified = 0
    with path.open() as handle:
        for row in csv.DictReader(handle):
            target = ROOT / row['path']
            if not target.is_file():
                missing.append(row['path'])
                continue
            digest = hashlib.sha256()
            with target.open('rb') as content:
                for block in iter(lambda: content.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if target.stat().st_size != int(row['bytes']) or digest.hexdigest() != row['sha256']:
                mismatched.append(row['path'])
            else:
                verified += 1
    return dict(verified=verified, missing_count=len(missing), mismatched_count=len(mismatched),
                missing_examples=missing[:20], mismatched_examples=mismatched[:20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true', help='Require every bulk-data file as well')
    args = parser.parse_args()
    public = verify_inventory(ROOT / 'FILE_INVENTORY.csv')
    bundle = json.loads((ROOT / 'data_release/bundle.json').read_text())
    result = dict(public=public, bulk_data_status=bundle['status'], bulk_url=bundle['url'])
    if args.external:
        result['external'] = verify_inventory(ROOT / 'data_release/EXTERNAL_FILES.csv')
    result['status'] = 'failed' if any(check['missing_count'] or check['mismatched_count'] for check in [public] + ([result['external']] if args.external else [])) else 'passed'
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
