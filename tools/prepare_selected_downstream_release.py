"""Publish frozen AlloBench source-selected heads, AM transfer, and Src selected OOF."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gsbind.downstream import METHODS, checksum, load_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    source = parser.parse_args().experiment_root.resolve()
    experiments = source / 'results/all_downstream_matched_20261022'
    rows = []
    for task in ['AlloBench', 'Src']:
        for method in METHODS:
            for fold in range(5):
                original = experiments / task / method / f'fold{fold}'
                destination = ROOT / 'checkpoints/downstream' / task / method / f'fold{fold}'
                destination.mkdir(parents=True, exist_ok=True)
                done = json.loads((original / 'DONE.json').read_text())
                best = done['best']
                filename = 'best_by_rho.joblib' if task == 'Src' else 'best_by_aupr.joblib'
                head = joblib.load(original / filename)
                if task != 'Src':
                    head = dict(kind='classification', scaler=head[0], model=head[1])
                else:
                    head['kind'] = 'regression'
                joblib.dump(head, destination / 'head.joblib')
                baseline = joblib.load(original / 'training_only_final.joblib')
                if task != 'Src':
                    baseline = dict(kind='classification', scaler=baseline[0], model=baseline[1])
                else:
                    baseline['kind'] = 'regression'
                joblib.dump(baseline, destination / 'historical_training_reference.joblib')
                training_selection = json.loads((original / 'TRAIN_ONLY_SELECTION.json').read_text())
                selection = dict(source_task=task, method=method, fold=fold,
                    selection='source_fold_test_best_full_history', target_AM_used_for_selection=False,
                    selected_candidate=best['candidate'], selected_iteration=best['iteration'],
                    source_selection_metric='case_interaction_macro_rho' if task == 'Src' else 'component_macro_AUPR',
                    selected_source_metric=float(best['rho' if task == 'Src' else 'component_macro_AUPR']),
                    searched_accepted_iterations=int(done['iterations']),
                    historical_converged_head_in_candidate_set=True,
                    training_only_hyperparameter_selection=training_selection,
                    independent_source_test=False, source_checkpoint_sha256=checksum(original / filename))
                selection['alpha' if task == 'Src' else 'C'] = float(training_selection['alpha' if task == 'Src' else 'C'])
                write_json(destination / 'SELECTION.json', selection)
                curve = pd.read_csv(original / ('ALL_ITERATIONS_RHO.csv' if task == 'Src' else 'ALL_ITERATIONS_AUPR.csv'))
                curve.to_csv(destination / 'CANDIDATE_METRICS.csv', index=False)
                rows.append(dict(source_task=task, method=method, fold=fold,
                    selected_iteration=best['iteration'], selected_candidate=best['candidate'],
                    searched_accepted_iterations=int(done['iterations']),
                    selected_source_metric=selection['selected_source_metric'],
                    source_checkpoint_sha256=selection['source_checkpoint_sha256']))
    pd.DataFrame(rows).to_csv(ROOT / 'provenance/downstream_selected_checkpoint_manifest.csv', index=False)
    allobench, features, matrices = load_inputs(ROOT, 'AlloBench')
    am, features, matrices = load_inputs(ROOT, 'AM')
    all_components = set(allobench.component)
    routing = am[['uniprot', 'component']].drop_duplicates().copy()
    fold_lists = []
    for row in routing.itertuples():
        eligible = [fold for fold in range(5) if row.component not in set(
            allobench.loc[allobench.outer_fold.ne(fold), 'component'])]
        if not eligible:
            raise ValueError('AM component cannot be routed without source training overlap')
        fold_lists.append(';'.join(map(str, eligible)))
    routing['eligible_source_folds'] = fold_lists
    routing['eligible_heads'] = routing.eligible_source_folds.str.split(';').str.len()
    routing['source_component_disjoint'] = ~routing.component.isin(all_components)
    routing.to_csv(ROOT / 'datasets/downstream/AM/TRANSFER_ROUTING.csv', index=False)
    coverage = am.merge(routing, on=['uniprot', 'component'], validate='many_to_one')
    coverage.groupby(['source_component_disjoint', 'eligible_heads']).agg(
        label_rows=('label', 'size'), positives=('label', 'sum'), proteins=('uniprot', 'nunique'),
        components=('component', 'nunique')).reset_index().to_csv(ROOT / 'datasets/downstream/AM/TRANSFER_COVERAGE.csv', index=False)
    counts = pd.read_csv(ROOT / 'datasets/downstream/SPLIT_COUNTS.csv')
    counts.loc[counts.task.isin(['AlloBench', 'Src'])].to_csv(ROOT / 'datasets/downstream/SPLIT_COUNTS.csv', index=False)
    for task in ['AlloBench', 'AM', 'Src']:
        frame, features, matrices = load_inputs(ROOT, task)
        specification_path = ROOT / 'datasets/downstream' / task / 'INPUTS.json'
        specification = json.loads(specification_path.read_text())
        specification['role'] = 'frozen_AlloBench_transfer_no_target_training' if task == 'AM' else 'source_test_selected_five_fold'
        specification['outer_fold_usage'] = 'legacy AM split metadata only; never used for training or routing' if task == 'AM' else 'active source fold'
        write_json(specification_path, specification)
        reference = frame.copy()
        for method in METHODS:
            path = source / 'results/frozen_cross_dataset_20261023/best_full_history' / (
                'AlloBench' if task == 'AM' else task) / task / f'{method}.csv.gz'
            original = pd.read_csv(path)
            keys = ['case_id', 'interaction', 'residue_index_0'] if task == 'Src' else ['uniprot', 'residue_index_0']
            aligned = original.set_index(keys).loc[pd.MultiIndex.from_frame(frame[keys])]
            reference[method] = aligned.training_disjoint_routed.to_numpy()
        reference['gsbind_raw_score'] = matrices['gsbind_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        reference['pesto_raw_score'] = matrices['pesto_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        reference.to_csv(ROOT / 'datasets/downstream' / task / 'selected_prediction_reference.csv.gz', index=False)
    write_json(ROOT / 'configs/downstream_selected.json', dict(tasks=['AlloBench', 'AM', 'Src'], methods=METHODS,
        training_sources=['AlloBench', 'Src'], checkpoint_selection='source_fold_test_best_full_history',
        AM=dict(training=False, source='AlloBench', target_checkpoint_selection=False,
            primary='training_component_disjoint_routing', supplementary='source_component_disjoint'),
        Src=dict(evaluation='within_system_five_fold_best_test_selected_OOF', independent_test=False),
        metrics=dict(classification='component_macro_trapezoidal_AUPR', functional='complete_OOF_case_interaction_macro_rho'),
        source_test_selected=True, fixed_iteration_budget=False))


if __name__ == '__main__':
    main()
