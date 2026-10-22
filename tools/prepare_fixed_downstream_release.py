"""Materialize all fixed-readout inputs and checkpoints from an explicitly supplied experiment folder."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gsbind.downstream import TASKS, METHODS, FIXED_ITERATIONS, checksum, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.experiment_root.resolve()
    source_results = source / 'results/all_downstream_matched_20261022'
    manifests = []
    for task in TASKS:
        folder = ROOT / 'datasets/downstream' / task
        folder.mkdir(parents=True, exist_ok=True)
        if task != 'Src':
            frame = pd.read_parquet(source / 'datasets/allosteric/tasks' / f'{task}.parquet').reset_index(drop=True)
            saved = pd.read_csv(source_results / task / 'ALL_METHODS_MATCHED_OOF.csv.gz')
            keys = ['uniprot', 'residue_index_0']
            saved = saved.set_index(keys).loc[pd.MultiIndex.from_frame(frame[keys])].reset_index()
            frame['outer_fold'] = saved.outer_fold.to_numpy()
        else:
            frame = pd.read_csv(source_results / task / 'ALL_METHODS_MATCHED_OOF.csv.gz')
        if task != 'Src':
            hidden = np.load(source / 'datasets/allosteric/hidden_centered_conditions.npy', mmap_mode='r')
            geometry = ['relative_sasa', 'contact_density_8A', 'dssp_helix', 'dssp_strand', 'dssp_turn']
            amino = 'ARNDCQEGHILKMFPSTWYV'
            matrices = dict(gsbind_score_probe=frame[['score']].to_numpy(),
                structure_only=np.column_stack([frame[geometry].to_numpy(float),
                    np.column_stack([frame.amino_acid.eq(letter) for letter in amino]).astype(float), frame.residue_plddt.to_numpy(float)]),
                gsbind_hidden_probe=np.asarray(hidden[0, frame.matrix_row]), pesto_score_probe=saved[['pesto_raw_score']].to_numpy())
            features = frame.drop(columns=['label', 'outer_fold']).copy()
            features['source_matrix_row'] = features.matrix_row
            features['matrix_row'] = np.arange(len(features))
            frame['matrix_row'] = np.arange(len(frame))
            frame = frame[['case_id', 'uniprot', 'component', 'residue_index_0', 'label', 'matrix_row', 'outer_fold']].copy()
        else:
            metadata = pd.read_csv(source / 'results/score_context_probe_20261021/inputs/old_matrix_metadata.csv')
            features = metadata.loc[metadata.case_id.isin(frame.case_id)].copy().reset_index(drop=True)
            original_rows = features.matrix_row.to_numpy(int)
            with np.load(source / 'results/score_context_probe_20261021/inputs/old_design_matrices.npz') as arrays:
                matrices = dict(gsbind_score_probe=arrays['score_probe'][original_rows], structure_only=arrays['structure_only'][original_rows],
                                gsbind_hidden_probe=arrays['hidden_probe'][original_rows])
            pesto = np.full((len(features), 1), np.nan)
            for case, group in features.groupby('case_id', sort=False):
                with np.load(source / 'results/baseline_hidden_probe_20261021/features/pesto' / f'{case}.npz') as arrays:
                    pesto[group.index, 0] = arrays['score'][group.residue_index_0.to_numpy(int)]
            matrices['pesto_score_probe'] = pesto
            mapping = dict(zip(original_rows, range(len(original_rows))))
            features['source_matrix_row'] = original_rows
            features['matrix_row'] = np.arange(len(features))
            frame['matrix_row'] = frame.matrix_row.map(mapping)
            frame = frame[['case_id', 'interaction', 'parent_uniprot', 'family', 'residue_index_0', 'physical_position',
                           'functional_magnitude', 'stability_magnitude', 'matrix_row', 'outer_fold']].copy()
        frame.to_csv(folder / 'labels.csv.gz', index=False)
        features.to_csv(folder / 'feature_rows.csv.gz', index=False)
        parts = []
        for start in range(0, len(features), 20000):
            name = f'features_{start // 20000:03d}.npz'
            np.savez_compressed(folder / name, **{method: values[start:start + 20000] for method, values in matrices.items()})
            parts.append(name)
        write_json(folder / 'INPUTS.json', dict(task=task, rows=len(frame), feature_rows=len(features),
                   dimensions={method: values.shape[1] for method, values in matrices.items()}, feature_parts=parts,
                   full_training_features_in_git=True, raw_or_rank_targets='raw labels; ranks recomputed on training fold only'))
        write_json(folder / 'SOURCE_INPUT_AUDIT.json', json.loads((source_results / task / 'INPUT_AUDIT.json').read_text()))
        references = frame.copy()
        for method in METHODS:
            predictions = np.full(len(frame), np.nan)
            for fold in range(5):
                original = source_results / task / method / f'fold{fold}'
                destination = ROOT / 'checkpoints/downstream' / task / method / f'fold{fold}'
                destination.mkdir(parents=True, exist_ok=True)
                metric_name = 'ALL_ITERATIONS_RHO.csv' if task == 'Src' else 'ALL_ITERATIONS_AUPR.csv'
                curve = pd.read_csv(original / metric_name)
                eligible = curve.loc[curve.iteration.notna() & curve.iteration.le(FIXED_ITERATIONS[task])]
                iteration = int(eligible.iteration.max())
                head = joblib.load(original / f'iteration_{iteration:04d}.joblib')
                if task != 'Src':
                    head = dict(kind='classification', scaler=head[0], model=head[1])
                else:
                    head['kind'] = 'regression'
                joblib.dump(head, destination / 'head.joblib')
                training_selection = json.loads((original / 'TRAIN_ONLY_SELECTION.json').read_text())
                record = dict(task=task, method=method, fold=fold, requested_iterations=FIXED_ITERATIONS[task],
                              actual_iterations=iteration, selection='fixed_iteration_no_test_checkpoint_selection',
                              training_only_hyperparameter_selection=training_selection,
                              historical_budget_choice='chosen after exploratory test inspection', independent_test=False,
                              source_checkpoint_sha256=checksum(original / f'iteration_{iteration:04d}.joblib'))
                record['alpha' if task == 'Src' else 'C'] = float(training_selection['alpha' if task == 'Src' else 'C'])
                write_json(destination / 'SELECTION.json', record)
                testing = frame.outer_fold.eq(fold).to_numpy()
                values = pd.read_csv(original / f'iteration_{iteration:04d}_predictions.csv.gz').prediction.to_numpy()
                if task != 'Src':
                    source_labels = pd.read_csv(original / 'BEST_TEST_PREDICTIONS.csv.gz')
                    source_labels['prediction'] = values
                    lookup = source_labels.set_index(['uniprot', 'residue_index_0']).prediction
                    values = lookup.loc[pd.MultiIndex.from_frame(frame.loc[testing, ['uniprot', 'residue_index_0']])].to_numpy()
                predictions[testing] = values
                manifests.append(dict(task=task, method=method, fold=fold, requested_iterations=FIXED_ITERATIONS[task], actual_iterations=iteration,
                                      hyperparameter=record.get('alpha', record.get('C')), source_sha256=record['source_checkpoint_sha256']))
            references[method] = predictions
        references['gsbind_raw_score'] = matrices['gsbind_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        references['pesto_raw_score'] = matrices['pesto_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        references.to_csv(folder / 'fixed_prediction_reference.csv.gz', index=False)
    pd.DataFrame(manifests).to_csv(ROOT / 'provenance/downstream_fixed_checkpoint_manifest.csv', index=False)
    write_json(ROOT / 'configs/downstream_fixed.json', dict(tasks=TASKS, methods=METHODS, fixed_iterations=FIXED_ITERATIONS,
               early_convergence='keep last available iterate; never substitute a historical final model',
               head_selection='fixed endpoint, no per-fold test checkpoint choice',
               historical_budget_choice='budgets chosen after exploratory test inspection; not independent test',
               original_backbone='frozen E11 direct128; PeSTo scalar probability only',
               metric='classification: trapezoidal AUPR; Src: full OOF Spearman, case/interaction macro',
               removed_from_active_maintext=['KRAS', 'ASBench', 'PASSer_ASD_source', 'CASBench', 'ASD_AlloPPI']))


if __name__ == '__main__':
    main()
