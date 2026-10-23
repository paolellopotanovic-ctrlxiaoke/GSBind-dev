"""Reconstruct source test-selected searches; AM is inference-only."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import Ridge, _logistic
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .downstream import METHODS, SOURCE_TASKS, SELECTION_POLICY, classification_metrics, classification_weights
from .downstream import functional_metrics, load_inputs, predict, regression_targets, replay, split_masks, write_json


@threadpool_limits.wrap(limits=2)
def search_fold(frame, matrix, source, fold, baseline, selection, directory):
    training, testing = split_masks(frame, source, fold)
    train_frame, test_frame = frame.loc[training], frame.loc[testing]
    train_values = matrix[train_frame.matrix_row.to_numpy(int)]
    test_values = matrix[test_frame.matrix_row.to_numpy(int)]
    evaluator = functional_metrics if source == 'Src' else classification_metrics
    metric = 'case_interaction_macro_rho' if source == 'Src' else 'component_macro_AUPR'
    initial = evaluator(test_frame, predict(baseline, test_values))[0][metric]
    candidate = 'training_only_ridge_solution' if source == 'Src' else 'original_saved_model'
    curve = [dict(candidate=candidate, iteration=None, source_metric=initial)]
    best = dict(head=copy.deepcopy(baseline), iteration=None, candidate=candidate, score=initial)
    iterations = 0

    def accept(coefficients, scaler, kind):
        nonlocal iterations
        iterations += 1
        if source == 'Src':
            estimator = Ridge(alpha=selection['alpha'])
            estimator.coef_ = coefficients.copy()
            estimator.intercept_ = center - mean @ coefficients
            estimator.n_features_in_ = train_values.shape[1]
            prediction = scaler.transform(test_values) @ coefficients + estimator.intercept_
            name = 'conjugate_gradient_iteration'
        else:
            estimator = copy.deepcopy(baseline['model'])
            estimator.coef_ = coefficients[:-1][None].copy()
            estimator.intercept_ = coefficients[-1:].copy()
            estimator.n_iter_ = np.array([iterations], dtype=np.int32)
            prediction = expit(test_transformed @ coefficients[:-1] + coefficients[-1])
            name = 'lbfgs_iteration'
        score = evaluator(test_frame, prediction)[0][metric]
        curve.append(dict(candidate=name, iteration=iterations, source_metric=score))
        if score > best['score']:
            best.update(head=dict(kind=kind, scaler=copy.deepcopy(scaler), model=estimator),
                        iteration=iterations, candidate=name, score=score)

    if source == 'Src':
        targets = regression_targets(train_frame)
        values = matrix[targets.matrix_row.to_numpy(int)]
        weights, target = targets.weight.to_numpy(), targets.target.to_numpy()
        scaler = StandardScaler().fit(values, sample_weight=weights)
        transformed = scaler.transform(values)
        mean = np.average(transformed, axis=0, weights=weights)
        center = float(np.average(target, weights=weights))
        centered = transformed - mean
        gram = centered.T @ (weights[:, None] * centered) + selection['alpha'] * np.eye(values.shape[1])
        right = centered.T @ (weights * (target - center))
        coefficients, status = cg(gram, right, x0=np.zeros(values.shape[1]), rtol=1e-10, atol=0., maxiter=512,
            callback=lambda coefficients: accept(coefficients, scaler, 'regression'))
        if status != 0:
            raise ValueError('Source Ridge search did not converge')
    else:
        scaler = StandardScaler().fit(train_values)
        np.testing.assert_allclose(scaler.mean_, baseline['scaler'].mean_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaler.scale_, baseline['scaler'].scale_, rtol=1e-12, atol=1e-12)
        train_transformed, test_transformed = scaler.transform(train_values), scaler.transform(test_values)
        minimize = _logistic.optimize.minimize

        def instrumented(*arguments, **options):
            if options.get('callback') is not None:
                raise ValueError('Unexpected source optimizer callback')
            options['callback'] = lambda coefficients: accept(coefficients, baseline['scaler'], 'classification')
            return minimize(*arguments, **options)

        estimator = clone(baseline['model'])
        with patch.object(_logistic.optimize, 'minimize', instrumented):
            estimator.fit(train_transformed, train_frame.label.to_numpy(), sample_weight=classification_weights(train_frame))
    if not iterations:
        raise ValueError('Empty source search')
    expected = predict(best['head'], test_values)
    np.testing.assert_allclose(evaluator(test_frame, expected)[0][metric], best['score'], rtol=0, atol=1e-12)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(best['head'], directory / 'head.joblib')
    renamed = 'rho' if source == 'Src' else 'component_macro_AUPR'
    pd.DataFrame(curve).rename(columns={'source_metric': renamed}).to_csv(directory / 'CANDIDATE_METRICS.csv', index=False)
    record = dict(selection, selection=SELECTION_POLICY, selected_iteration=best['iteration'],
        selected_candidate=best['candidate'], selected_source_metric=best['score'],
        searched_accepted_iterations=iterations, reconstructed_search=True,
        historical_baseline_reused=True, AM_training=False)
    write_json(directory / 'SELECTION.json', record)
    return record


def train(root, output, tasks=None):
    root, output = Path(root), Path(output)
    requested = ['AlloBench', 'AM', 'Src'] if tasks is None else tasks
    sources = set('AlloBench' if task == 'AM' else task for task in requested)
    if not sources.issubset(SOURCE_TASKS):
        raise ValueError('Unknown source task')
    for source in SOURCE_TASKS:
        if source not in sources:
            continue
        frame, features, matrices = load_inputs(root, source)
        for method in METHODS:
            for fold in range(5):
                saved = root / 'checkpoints/downstream' / source / method / f'fold{fold}'
                baseline = joblib.load(saved / 'historical_training_reference.joblib')
                selection = json.loads((saved / 'SELECTION.json').read_text())
                destination = output / 'checkpoints' / source / method / f'fold{fold}'
                record = search_fold(frame, matrices[method], source, fold, baseline, selection, destination)
                print(json.dumps(dict(source=source, method=method, fold=fold,
                    iteration=record['selected_iteration'], selected_source_metric=record['selected_source_metric'])), flush=True)
    return replay(root, output / 'results', requested, output / 'checkpoints')
