"""Export a detailed Parquet table as CSV or compressed CSV in bounded memory."""

import argparse
import gzip
from pathlib import Path

import pyarrow.parquet as parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == '.gz' else open
    rows = 0
    with opener(args.output, 'wt', newline='') as handle:
        for batch in parquet.ParquetFile(args.input).iter_batches(batch_size=65536):
            batch.to_pandas().to_csv(handle, index=False, header=rows == 0)
            rows += batch.num_rows
    print(f'Exported {rows} rows to {args.output}')


if __name__ == '__main__':
    main()
