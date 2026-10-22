"""Audit delivered numerical tables against a new public recomputation."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gsbind.downstream import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recomputed', type=Path, default=ROOT / 'runs/current_maintext_audit')
    arguments = parser.parse_args()
    checks = []
    for task in ['main', 'ssm220', 'downstream']:
        for source in sorted((ROOT / 'results' / task).glob('*.csv')):
            actual = arguments.recomputed / task / source.name
            if not actual.exists():
                raise FileNotFoundError(actual)
            expected = pd.read_csv(source)
            observed = pd.read_csv(actual)
            pd.testing.assert_frame_equal(expected, observed, check_dtype=False, rtol=1e-10, atol=1e-12)
            checks.append(dict(path=str(source.relative_to(ROOT)), rows=len(expected), status='passed'))
    for task in ['AlloBench', 'AM', 'Src']:
        path = Path('downstream') / task / 'OOF_predictions.csv.gz'
        expected = pd.read_csv(ROOT / 'results' / path)
        actual = pd.read_csv(arguments.recomputed / path)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, rtol=1e-10, atol=1e-12)
        checks.append(dict(path=str(path), rows=len(expected), status='passed'))
    result = dict(status='passed', checked_tables=checks,
                  classification_metric='downstream trapezoidal AUPR; binding/SSM legacy aupr fields are AP',
                  scope='Public saved predictions and frozen readouts; not full backbone retraining or manuscript editorial approval')
    write_json(ROOT / 'provenance/current_maintext_numerical_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
