"""Compare independently recomputed current application tables and intervals."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual):
    left, right = pd.read_csv(expected), pd.read_csv(actual)
    pd.testing.assert_frame_equal(left, right, atol=1e-10, rtol=1e-10)
    return dict(path=str(expected.relative_to(ROOT)), rows=len(left), columns=len(left.columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recomputed', type=Path, default=ROOT / 'runs/application_recompute')
    parser.add_argument('--statistics', type=Path, default=ROOT / 'runs/application_statistics/functional')
    parser.add_argument('--strata', type=Path, required=True)
    parser.add_argument('--unit-tests', type=int)
    args = parser.parse_args()
    checked = []
    for actual in sorted(args.recomputed.rglob('*.csv')):
        expected = ROOT / 'results' / actual.relative_to(args.recomputed)
        if expected.exists():
            checked.append(compare(expected, actual))
    interval_names = ['sequence_block_bootstrap.csv', 'budget_block_bootstrap.csv', 'partial_rank_bootstrap.csv']
    intervals = [compare(ROOT / 'results/functional' / name, args.statistics / name) for name in interval_names]
    strata = [compare(ROOT / 'results/allosteric_strata' / name, args.strata / name) for name in
        ['complete_coverage_audit.csv', 'MAIN_AUPR_BY_SCORE_STRATUM.csv', 'paired_aupr_contrasts.csv']]
    replay = json.loads((args.recomputed / 'functional/readout_verification.json').read_text())
    separation = pd.read_csv(ROOT / 'results/functional_separation/paired_block_bootstrap.csv')
    assert separation.iterations.eq(10000).all()
    assert separation.valid_draws.eq(10000).all()
    result = dict(status='passed', date='2026-10-20', model='Gaussian-direct128 E11',
        functional_scope='all_measured_positions', target_functional_training=False,
        readout_replay=replay, recomputed_tables=checked, full_10000_iteration_intervals=intervals,
        allosteric_complete_and_stratified_comparisons=strata,
        functional_separation=dict(analyses=3, bootstrap=10000, interval_rows=len(separation)),
        source_code='src/gsbind/functional.py; functional_statistics.py; allosteric_strata.py; functional_separation.py')
    (ROOT / 'provenance/application_update_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    comparison = dict(status='passed', csv_tables=len(checked), atol=1e-10,
        files=[row['path'].removeprefix('results/') for row in checked],
        application_update='all-position functional evaluation; separate complete interval/strata audit in application_update_validation.json')
    (ROOT / 'provenance/publication_result_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    publication = json.loads((ROOT / 'provenance/publication_validation.json').read_text())
    publication.update(recomputed_csv_tables=len(checked),
        current_application_update='application_update_validation.json',
        functional_scope='all_measured_positions', full_interval_reproduction=True)
    if args.unit_tests is not None:
        publication['unit_tests'] = f'{args.unit_tests} passed (separate pytest run)'
    (ROOT / 'provenance/publication_validation.json').write_text(json.dumps(publication, indent=2) + '\n')
    print(json.dumps(dict(status='passed', tables=len(checked), interval_tables=len(intervals), strata_tables=len(strata))))


if __name__ == '__main__':
    main()
