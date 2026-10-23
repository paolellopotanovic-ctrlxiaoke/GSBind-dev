"""Source-selected AlloBench readouts, frozen AM transfer, and Src selected OOF."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import auc, precision_recall_curve
from threadpoolctl import threadpool_limits


TASKS = ['AlloBench', 'AM', 'Src']
METHODS = ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe', 'pesto_score_probe']
SOURCE_TASKS = ['AlloBench', 'Src']
SELECTION_POLICY = 'source_fold_test_best_full_history'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(root, task):
    directory = Path(root) / 'datasets/downstream' / task
    specification = json.loads((directory / 'INPUTS.json').read_text())
    labels = pd.read_csv(directory / 'labels.csv.gz')
    features = pd.read_csv(directory / 'feature_rows.csv.gz')
    matrices = {method: [] for method in METHODS}
    for part in specification['feature_parts']:
        with np.load(directory / part, allow_pickle=False) as arrays:
            for method in METHODS:
                matrices[method].append(arrays[method].copy())
    matrices = {method: np.concatenate(blocks, axis=0) for method, blocks in matrices.items()}
    for method, matrix in matrices.items():
        if len(matrix) != len(features) or not np.isfinite(matrix).all():
            raise ValueError('Incomplete features: ' + task + '/' + method)
        if matrix.shape[1] != specification['dimensions'][method]:
            raise ValueError('Feature dimension mismatch')
    if matrices['pesto_score_probe'].shape[1] != 1:
        raise ValueError('PeSTo must use scalar score only')
    if labels.duplicated(['case_id', 'residue_index_0'] + (['interaction'] if task == 'Src' else [])).any():
        raise ValueError('Duplicate label rows')
    if not labels.matrix_row.between(0, len(features) - 1).all():
        raise ValueError('Invalid feature mapping')
    aligned = features.iloc[labels.matrix_row.to_numpy(int)]
    np.testing.assert_array_equal(aligned.case_id, labels.case_id)
    np.testing.assert_array_equal(aligned.residue_index_0, labels.residue_index_0)
    if set(labels.outer_fold) != set(range(5)):
        raise ValueError('Require exactly five saved folds')
    return labels, features, matrices


def split_masks(frame, task, fold):
    testing = frame.outer_fold.eq(fold).to_numpy()
    training = ~testing
    if task == 'Src':
        positions = frame.physical_position.to_numpy(int)
        heldout = np.unique(positions[testing])
        training &= np.abs(positions[:, None] - heldout).min(axis=1) > 5
        if np.abs(positions[training, None] - positions[testing]).min() <= 5:
            raise ValueError('Functional neighbor leakage')
        position_folds = frame.groupby('physical_position').outer_fold.nunique()
        if position_folds.max() != 1:
            raise ValueError('Src FL/KD positions not held out jointly')
    else:
        for column in ['component', 'uniprot']:
            if set(frame.loc[training, column]) & set(frame.loc[testing, column]):
                raise ValueError('Classification group leakage: ' + column)
    if not training.any() or not testing.any():
        raise ValueError('Empty fold')
    return training, testing


def classification_weights(frame):
    lengths = frame.groupby('uniprot').size()
    proteins = frame[['component', 'uniprot']].drop_duplicates().groupby('component').size()
    weights = np.asarray([1 / (lengths[row.uniprot] * proteins[row.component]) for row in frame.itertuples()])
    return weights / weights.mean()


def regression_targets(frame):
    values = frame.copy()
    values['target'] = values.groupby(['case_id', 'interaction']).functional_magnitude.transform(
        lambda series: rankdata(series) / len(series))
    targets = values.groupby(['case_id', 'residue_index_0', 'parent_uniprot', 'matrix_row'], as_index=False).target.mean()
    cases = targets.groupby('parent_uniprot').case_id.transform('nunique').to_numpy()
    lengths = targets.groupby(['parent_uniprot', 'case_id']).case_id.transform('size').to_numpy()
    weights = 1. / (cases * lengths)
    targets['weight'] = weights * len(weights) / weights.sum()
    return targets


def aupr(labels, scores):
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        raise ValueError('AUPR requires both classes and finite predictions')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def classification_metrics(frame, prediction):
    values = frame.copy()
    values['prediction'] = prediction
    rows = [dict(uniprot=protein, component=group.component.iloc[0],
                 AUPR=aupr(group.label.to_numpy(), group.prediction.to_numpy()))
            for protein, group in values.groupby('uniprot', sort=False)]
    detailed = pd.DataFrame(rows)
    result = dict(component_macro_AUPR=float(detailed.groupby('component').AUPR.mean().mean()),
                  pooled_AUPR=aupr(values.label.to_numpy(), values.prediction.to_numpy()))
    return result, detailed


def functional_metrics(frame, prediction):
    values = frame.copy()
    values['prediction'] = prediction
    rows = []
    for (case, interaction), group in values.groupby(['case_id', 'interaction'], sort=False):
        if group.prediction.nunique() < 2 or group.functional_magnitude.nunique() < 2:
            raise ValueError('Undefined Spearman')
        value = float(spearmanr(group.prediction, group.functional_magnitude).statistic)
        rows.append(dict(case_id=case, interaction=interaction, rho=value, positions=len(group)))
    detailed = pd.DataFrame(rows)
    return dict(case_interaction_macro_rho=float(detailed.groupby('case_id').rho.mean().mean())), detailed


def predict(head, matrix):
    values = head['scaler'].transform(matrix)
    return head['model'].predict_proba(values)[:, 1] if head['kind'] == 'classification' else head['model'].predict(values)


def transfer_routing(source, target):
    eligible = np.column_stack([~target.component.isin(set(source.loc[
        source.outer_fold.ne(fold), 'component'])) for fold in range(5)])
    count = eligible.sum(axis=1)
    if not (count > 0).all():
        raise ValueError('A target component occurs in every source training fold')
    independent = ~target.component.isin(set(source.component)).to_numpy()
    return eligible, independent


def load_selected_head(root, source, method, fold, checkpoints=None):
    checkpoint_root = Path(checkpoints) if checkpoints else root / 'checkpoints/downstream'
    directory = checkpoint_root / source / method / f'fold{fold}'
    selection = json.loads((directory / 'SELECTION.json').read_text())
    if selection['selection'] != SELECTION_POLICY or selection['source_task'] != source:
        raise ValueError('Wrong source checkpoint selection protocol')
    if selection['target_AM_used_for_selection']:
        raise ValueError('AM must never select source checkpoints')
    curve = pd.read_csv(directory / 'CANDIDATE_METRICS.csv')
    metric = 'rho' if source == 'Src' else 'component_macro_AUPR'
    np.testing.assert_allclose(curve[metric].max(), selection['selected_source_metric'], rtol=0, atol=1e-12)
    iteration = selection['selected_iteration']
    candidates = curve.loc[curve.candidate.eq(selection['selected_candidate'])]
    candidates = candidates.loc[candidates.iteration.isna() if iteration is None else candidates.iteration.eq(iteration)]
    if len(candidates) != 1:
        raise ValueError('Selected source candidate missing from search history')
    np.testing.assert_allclose(candidates.iloc[0][metric], selection['selected_source_metric'], rtol=0, atol=1e-12)
    return joblib.load(directory / 'head.joblib'), selection


@threadpool_limits.wrap(limits=2)
def replay(root, output, tasks=None, checkpoints=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requested = TASKS if tasks is None else tasks
    if not requested or not set(requested).issubset(TASKS):
        raise ValueError('Unknown or empty downstream task list')
    summaries, fold_rows, detailed_rows, validations = [], [], [], []
    source_frame = load_inputs(root, 'AlloBench')[0] if 'AM' in requested else None
    for task in requested:
        frame, features, matrices = load_inputs(root, task)
        source = 'AlloBench' if task == 'AM' else task
        predictions = frame.copy()
        if task == 'AM':
            eligible, independent = transfer_routing(source_frame, frame)
            predictions['eligible_source_heads'] = eligible.sum(axis=1)
            predictions['source_component_disjoint'] = independent
            routing = pd.read_csv(root / 'datasets/downstream/AM/TRANSFER_ROUTING.csv').set_index('uniprot')
            np.testing.assert_array_equal(routing.loc[frame.uniprot, 'eligible_heads'], eligible.sum(axis=1))
            np.testing.assert_array_equal(routing.loc[frame.uniprot, 'source_component_disjoint'], independent)
            recorded = routing.loc[frame.uniprot, 'eligible_source_folds'].astype(str)
            expected_folds = [';'.join(map(str, np.flatnonzero(row))) for row in eligible]
            np.testing.assert_array_equal(recorded, expected_folds)
        for method in METHODS:
            values = np.full(len(frame), np.nan)
            target_matrix = matrices[method][frame.matrix_row.to_numpy(int)]
            if task == 'AM':
                per_head = np.column_stack([predict(load_selected_head(root, source, method, fold, checkpoints)[0],
                    target_matrix) for fold in range(5)])
                values = (per_head * eligible).sum(axis=1) / eligible.sum(axis=1)
            else:
                for fold in range(5):
                    training, testing = split_masks(frame, task, fold)
                    head, selection = load_selected_head(root, source, method, fold, checkpoints)
                    values[testing] = predict(head, target_matrix[testing])
                    evaluator = functional_metrics if task == 'Src' else classification_metrics
                    measured, detailed = evaluator(frame.loc[testing], values[testing])
                    selected_metric = measured['case_interaction_macro_rho' if task == 'Src' else 'component_macro_AUPR']
                    np.testing.assert_allclose(selected_metric, selection['selected_source_metric'], rtol=0, atol=1e-12)
                    fold_rows.append(dict(task=task, method=method, fold=fold, source_task=source,
                        selected_iteration=selection['selected_iteration'], selected_candidate=selection['selected_candidate'],
                        **measured))
            if not np.isfinite(values).all():
                raise ValueError('Incomplete predictions')
            predictions[method] = values
        predictions['gsbind_raw_score'] = matrices['gsbind_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        predictions['pesto_raw_score'] = matrices['pesto_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        evaluator = functional_metrics if task == 'Src' else classification_metrics
        scopes = [('all_positions', np.ones(len(frame), bool))]
        if task == 'AM':
            scopes.append(('source_component_disjoint', independent))
        for scope, mask in scopes:
            for method in METHODS + ['gsbind_raw_score', 'pesto_raw_score']:
                measured, detailed = evaluator(frame.loc[mask], predictions.loc[mask, method].to_numpy())
                summaries.append(dict(task=task, source_task=source, method=method, scope=scope,
                    evaluation='frozen_source_transfer' if task == 'AM' else 'source_test_selected_OOF',
                    label_rows=int(mask.sum()), **measured))
                detailed_rows.append(detailed.assign(task=task, source_task=source, method=method, scope=scope))
        target = output / task
        target.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(target / 'predictions.csv.gz', index=False)
        expected = pd.read_csv(root / 'datasets/downstream' / task / 'selected_prediction_reference.csv.gz')
        pd.testing.assert_frame_equal(predictions[expected.columns], expected, rtol=1e-10, atol=1e-12, check_dtype=False)
        validations.append(dict(task=task, source_task=source, rows=len(predictions), selected_checkpoint_reference='passed',
            target_training=False if task == 'AM' else True,
            source_component_disjoint_rows=int(independent.sum()) if task == 'AM' else None))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'METRICS.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output / 'FOLD_METRICS.csv', index=False)
    pd.concat(detailed_rows, ignore_index=True).to_csv(output / 'DETAILED_METRICS.csv', index=False)
    write_json(output / 'VALIDATION.json', dict(status='passed', tasks=validations, protocol=SELECTION_POLICY,
        checkpoint_selection_uses_source_test=True, target_AM_training=False,
        target_AM_checkpoint_selection=False, independent_source_test=False,
        AM_all_positions_fully_source_component_independent=False))
    return summary


def build_figure(root, output=None):
    from .downstream_figures import build_figure as render
    return render(root, output)
