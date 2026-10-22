"""Fixed-budget frozen readouts for the current biological application section."""

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg
from scipy.stats import rankdata, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


TASKS = ['AlloBench', 'AM', 'Src']
METHODS = ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe', 'pesto_score_probe']
FIXED_ITERATIONS = {'AlloBench': 50, 'AM': 50, 'Src': 10}


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


@threadpool_limits.wrap(limits=2)
def fit_fixed(frame, matrix, task, selection, parameters):
    cap = FIXED_ITERATIONS[task]
    if task != 'Src':
        indices = frame.matrix_row.to_numpy(int)
        values = matrix[indices]
        scaler = StandardScaler().fit(values)
        from sklearn.linear_model import LogisticRegression
        options = dict(parameters)
        options['max_iter'] = cap
        estimator = LogisticRegression(**options)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(scaler.transform(values), frame.label.to_numpy(), sample_weight=classification_weights(frame))
        expected = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        return dict(kind='classification', scaler=scaler, model=estimator), dict(
            requested_iterations=cap, actual_iterations=int(estimator.n_iter_[0]),
            iteration_limit_warning=bool(expected), training_only_C=selection['C'])
    targets = regression_targets(frame)
    indices = targets.matrix_row.to_numpy(int)
    values = matrix[indices]
    weights = targets.weight.to_numpy()
    target = targets.target.to_numpy()
    scaler = StandardScaler().fit(values, sample_weight=weights)
    transformed = scaler.transform(values)
    mean = np.average(transformed, axis=0, weights=weights)
    center = float(np.average(target, weights=weights))
    centered = transformed - mean
    alpha = float(selection['alpha'])
    gram = centered.T @ (weights[:, None] * centered) + alpha * np.eye(values.shape[1])
    right = centered.T @ (weights * (target - center))
    accepted = []
    def record(coefficients):
        accepted.append(coefficients.copy())
    coefficients, status = cg(gram, right, x0=np.zeros(values.shape[1]), rtol=1e-10, atol=0., maxiter=cap, callback=record)
    if status < 0 or not accepted:
        raise ValueError('Ridge iterative solver failed')
    estimator = Ridge(alpha=alpha)
    estimator.coef_ = coefficients.copy()
    estimator.intercept_ = center - mean @ coefficients
    estimator.n_features_in_ = values.shape[1]
    return dict(kind='regression', scaler=scaler, model=estimator), dict(
        requested_iterations=cap, actual_iterations=len(accepted), converged_early=status == 0,
        training_only_alpha=alpha, cg_status=int(status))


def replay(root, output, tasks=None, checkpoints=None):
    root, output = Path(root), Path(output)
    checkpoint_root = Path(checkpoints) if checkpoints else root / 'checkpoints/downstream'
    output.mkdir(parents=True, exist_ok=True)
    summaries, fold_rows, detailed_rows, validations = [], [], [], []
    for task in tasks or TASKS:
        frame, features, matrices = load_inputs(root, task)
        oof = frame.copy()
        for method in METHODS:
            values = np.full(len(frame), np.nan)
            for fold in range(5):
                training, testing = split_masks(frame, task, fold)
                directory = checkpoint_root / task / method / f'fold{fold}'
                head = joblib.load(directory / 'head.joblib')
                selection = json.loads((directory / 'SELECTION.json').read_text())
                if selection['requested_iterations'] != FIXED_ITERATIONS[task] or selection['selection'] != 'fixed_iteration_no_test_checkpoint_selection':
                    raise ValueError('Wrong checkpoint selection protocol')
                values[testing] = predict(head, matrices[method][frame.loc[testing, 'matrix_row'].to_numpy(int)])
                evaluator = functional_metrics if task == 'Src' else classification_metrics
                measured, detailed = evaluator(frame.loc[testing], values[testing])
                fold_rows.append(dict(task=task, method=method, fold=fold, **measured,
                                      actual_iterations=selection['actual_iterations']))
            if not np.isfinite(values).all():
                raise ValueError('Incomplete OOF predictions')
            oof[method] = values
        oof['gsbind_raw_score'] = matrices['gsbind_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        oof['pesto_raw_score'] = matrices['pesto_score_probe'][frame.matrix_row.to_numpy(int), 0].astype(float)
        evaluator = functional_metrics if task == 'Src' else classification_metrics
        for method in METHODS + ['gsbind_raw_score', 'pesto_raw_score']:
            measured, detailed = evaluator(frame, oof[method].to_numpy())
            summaries.append(dict(task=task, method=method, fixed_iterations=FIXED_ITERATIONS[task], **measured))
            detailed_rows.append(detailed.assign(task=task, method=method))
        target = output / task
        target.mkdir(parents=True, exist_ok=True)
        oof.to_csv(target / 'OOF_predictions.csv.gz', index=False)
        reference = root / 'datasets/downstream' / task / 'fixed_prediction_reference.csv.gz'
        if reference.exists():
            expected = pd.read_csv(reference)
            pd.testing.assert_frame_equal(oof[expected.columns], expected, rtol=1e-10, atol=1e-12, check_dtype=False)
            validations.append(dict(task=task, rows=len(oof), checkpoint_replay='passed', fixed_iteration_reference='passed'))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'METRICS.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output / 'FOLD_METRICS.csv', index=False)
    pd.concat(detailed_rows, ignore_index=True).to_csv(output / 'DETAILED_METRICS.csv', index=False)
    write_json(output / 'VALIDATION.json', dict(status='passed', tasks=validations,
               protocol='fixed50_AlloBench_AM_fixed10_Src', no_test_checkpoint_selection=True,
               budgets_chosen_after_exploratory_test_inspection=True, independent_test=False))
    return summary


def train(root, output, tasks=None):
    root, output = Path(root), Path(output)
    checkpoint_root = output / 'checkpoints'
    for task in tasks or TASKS:
        frame, features, matrices = load_inputs(root, task)
        for method in METHODS:
            for fold in range(5):
                training, testing = split_masks(frame, task, fold)
                saved = root / 'checkpoints/downstream' / task / method / f'fold{fold}'
                selection = json.loads((saved / 'SELECTION.json').read_text())
                reference = joblib.load(saved / 'head.joblib')
                parameters = reference['model'].get_params() if task != 'Src' else {}
                head, information = fit_fixed(frame.loc[training], matrices[method], task, selection, parameters)
                destination = checkpoint_root / task / method / f'fold{fold}'
                destination.mkdir(parents=True, exist_ok=True)
                joblib.dump(head, destination / 'head.joblib')
                record = dict(selection)
                record.update(information)
                record['refit'] = True
                write_json(destination / 'SELECTION.json', record)
                print(json.dumps(dict(task=task, method=method, fold=fold, **information)), flush=True)
    return replay(root, output / 'results', tasks, checkpoint_root)


def build_figure(root, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    directory = Path(output) if output else root / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / 'results/downstream/METRICS.csv')
    labels = ['GSBind score', 'Structure', 'GSBind hidden', 'PeSTo score']
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for task, axis in zip(TASKS, axes):
        metric = 'case_interaction_macro_rho' if task == 'Src' else 'component_macro_AUPR'
        values = table.loc[table.task.eq(task)].set_index('method').loc[METHODS, metric]
        axis.bar(labels, values, color=['#8294a4', '#a4b2b9', '#bb5f48', '#56788b'])
        axis.set(title=f'{task}: fixed {FIXED_ITERATIONS[task]} iterations',
                 ylabel='OOF Spearman rho' if task == 'Src' else 'Component-macro trapezoidal AUPR')
        axis.tick_params(axis='x', rotation=30)
    figure.savefig(directory / 'downstream_fixed_iterations.pdf')
    figure.savefig(directory / 'downstream_fixed_iterations.png', dpi=180)
    plt.close(figure)
    table.to_csv(directory / 'downstream_fixed_iterations_source.csv', index=False)
