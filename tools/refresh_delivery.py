"""Refresh current table index and content inventory after evidence updates."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet


EXCLUDES = {'.git', '.pytest_cache', '__pycache__', 'runs'}


def paths(root):
    return sorted(path for path in root.rglob('*') if path.is_file()
        and not EXCLUDES.intersection(path.relative_to(root).parts)
        and path.suffix != '.log'
        and not any(part.endswith('.egg-info') for part in path.relative_to(root).parts))


def refresh(root):
    tables = []
    for path in paths(root):
        relative = str(path.relative_to(root))
        if not (path.suffix in ['.csv', '.parquet'] or path.name.endswith('.csv.gz')) or relative in ['FILE_INVENTORY.csv', 'docs/TABLE_INDEX.csv']:
            continue
        if relative.startswith('data_release/'):
            continue
        if path.suffix == '.csv' or path.name.endswith('.csv.gz'):
            opener = gzip.open if path.name.endswith('.csv.gz') else open
            with opener(path, 'rt', newline='') as handle:
                reader = csv.reader(handle)
                columns = next(reader, [])
                count = sum(1 for row in reader)
        else:
            metadata = parquet.ParquetFile(path)
            columns = metadata.schema_arrow.names
            count = metadata.metadata.num_rows
        role = 'figure_source' if relative.startswith('figures/') else (
            'metric_or_prediction' if relative.startswith('results/') else 'data_or_protocol')
        tables.append(dict(path=relative, role=role, format='csv.gz' if path.name.endswith('.csv.gz') else path.suffix[1:], rows=count,
            bytes=path.stat().st_size, columns=json.dumps(columns, ensure_ascii=False)))
    with (root / 'docs/TABLE_INDEX.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'role', 'format', 'rows', 'bytes', 'columns'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(tables)
    inventory = []
    for path in paths(root):
        if str(path.relative_to(root)) in ['FILE_INVENTORY.csv', 'DELIVERY.json']:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        inventory.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    target = root / 'FILE_INVENTORY.csv'
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'bytes', 'sha256'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(inventory)
    delivery = dict(package='GSbind current paper engineering', date='2026-10-23',
        files=len(inventory), bytes=sum(row['bytes'] for row in inventory),
        inventory_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        excludes=sorted(EXCLUDES), inventory_self_excluded=True, delivery_self_excluded=True,
        entrypoint='python main.py summarize --task all; python main.py downstream --action train; python main.py figures',
        functional_scope='Src internal five-fold best-test selected OOF; AM frozen AlloBench transfer', archive=None,
        archive_note='Existing GSbind_paper_20261020.tar.zst predates this update; bulk datasets only, install missing files without overwriting public data.')
    (root / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2) + '\n')
    print(json.dumps(dict(root=str(root), files=len(inventory), tables=len(tables), inventory_sha256=delivery['inventory_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    refresh(parser.parse_args().root.resolve())
