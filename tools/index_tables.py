"""Index current paper tables, including fields and row counts for plotting."""

import csv
import json
from pathlib import Path

import pyarrow.parquet as parquet


ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for directory in ['datasets', 'results', 'figures/source_data', 'checkpoints', 'evidence']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.suffix == '.csv':
                with path.open(newline='') as handle:
                    reader = csv.reader(handle)
                    columns = next(reader, [])
                    rows = sum(1 for record in reader)
            elif path.suffix == '.parquet':
                table = parquet.ParquetFile(path)
                columns = table.schema_arrow.names
                rows = table.metadata.num_rows
            else:
                continue
            relative = path.relative_to(ROOT).as_posix()
            role = 'authoritative_result' if relative.startswith('results/') else 'figure_copy' if relative.startswith('figures/') else 'data_or_protocol'
            records.append(dict(path=relative, role=role, format=path.suffix[1:], rows=rows,
                                bytes=path.stat().st_size, columns=json.dumps(columns, ensure_ascii=False)))
    target = ROOT / 'docs/TABLE_INDEX.csv'
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'role', 'format', 'rows', 'bytes', 'columns'])
        writer.writeheader()
        writer.writerows(records)
    print(f'Indexed {len(records)} tables in {target}')


if __name__ == '__main__':
    main()
