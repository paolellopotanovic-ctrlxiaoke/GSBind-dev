import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'__pycache__', '.pytest_cache', 'runs', '.git'}


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=checksum(path))


def main():
    parser = argparse.ArgumentParser(description='Create a self-contained GSbind paper archive with SHA256 inventory')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    paths = sorted(path for path in ROOT.rglob('*') if path.is_file() and not set(path.relative_to(ROOT).parts) & EXCLUDED
                   and path.name not in ['FILE_INVENTORY.csv', 'DELIVERY.json'] and path.suffix != '.log')
    if any(path.is_symlink() for path in paths):
        raise ValueError('External symlinks are not allowed')
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(record, paths))
    inventory = ROOT / 'FILE_INVENTORY.csv'
    with inventory.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['path', 'bytes', 'sha256'], lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(package='GSbind public paper engineering', date='2026-10-23', files=len(records), bytes=sum(row['bytes'] for row in records),
                    inventory_sha256=checksum(inventory), excludes=sorted(EXCLUDED),
                    inventory_self_excluded=True, hardlinks_are_archived_with_content=True,
                    entrypoint='python tools/check_publication.py; python main.py summarize --task all',
                    bulk_data='data_release/bundle.json; public URL pending',
                    archive=str(args.archive.resolve()) if args.archive else None)
    (ROOT / 'DELIVERY.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.archive:
        archive = args.archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise FileExistsError(str(archive))
        command = ['tar', '-C', str(ROOT.parent)]
        command += ['--exclude=' + ROOT.name + '/' + name for name in EXCLUDED]
        command += ['--exclude=*/__pycache__', '--exclude=*/.pytest_cache', '--exclude=' + ROOT.name + '/provenance/package.log', '--exclude=' + ROOT.name + '/assembly.log']
        if archive.suffix == '.zst':
            executable = shutil.which('zstd')
            if executable is None:
                raise FileNotFoundError('zstd is required for .tar.zst')
            command += ['-I', executable + ' -T4 -1']
        command += ['-cf', str(archive), ROOT.name]
        subprocess.run(command, check=True)
        sidecar = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=checksum(archive), inventory_sha256=metadata['inventory_sha256'])
        archive.with_name(archive.name + '.json').write_text(json.dumps(sidecar, indent=2) + '\n')
        print(json.dumps(sidecar, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
