import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from gsbind.downstream import SELECTION_POLICY, classification_weights, regression_targets, split_masks
from gsbind.downstream_training import search_fold


def test_classification_source_search_records_best_without_am(tmp_path):
    generator = np.random.default_rng(17)
    frame = pd.DataFrame(dict(case_id=np.repeat(['a', 'b', 'c', 'd', 'e'], 12),
        uniprot=np.repeat(['a', 'b', 'c', 'd', 'e'], 12), component=np.repeat(['a', 'b', 'c', 'd', 'e'], 12),
        outer_fold=np.repeat(range(5), 12), label=np.tile([0, 1], 30), matrix_row=np.arange(60)))
    matrix = generator.normal(size=(60, 4))
    training, testing = split_masks(frame, 'AlloBench', 0)
    scaler = StandardScaler().fit(matrix[training])
    model = LogisticRegression(C=.1, max_iter=1500).fit(scaler.transform(matrix[training]),
        frame.loc[training, 'label'], sample_weight=classification_weights(frame.loc[training]))
    baseline = dict(kind='classification', scaler=scaler, model=model)
    selection = dict(source_task='AlloBench', C=.1, target_AM_used_for_selection=False)
    record = search_fold(frame, matrix, 'AlloBench', 0, baseline, selection, tmp_path)
    candidates = pd.read_csv(tmp_path / 'CANDIDATE_METRICS.csv')
    assert record['selection'] == SELECTION_POLICY
    assert record['selected_source_metric'] == pytest.approx(candidates.component_macro_AUPR.max())
    assert candidates.iteration.isna().sum() == 1
    assert record['historical_baseline_reused']
    assert not record['AM_training']


def test_src_source_search_preserves_purged_split_and_best(tmp_path):
    generator = np.random.default_rng(21)
    positions = np.tile(np.arange(1, 161), 2)
    matrix = generator.normal(size=(160, 5))
    frame = pd.DataFrame(dict(case_id=np.repeat(['FL', 'KD'], 160), interaction=np.repeat(['SRC_FL', 'SRC_KD'], 160),
        parent_uniprot='P12931', residue_index_0=positions - 1, physical_position=positions,
        matrix_row=positions - 1, functional_magnitude=np.tile(matrix[:, 0], 2), outer_fold=((positions - 1) // 10) % 5))
    training, testing = split_masks(frame, 'Src', 0)
    targets = regression_targets(frame.loc[training])
    values = matrix[targets.matrix_row.to_numpy(int)]
    scaler = StandardScaler().fit(values, sample_weight=targets.weight)
    model = Ridge(alpha=10.).fit(scaler.transform(values), targets.target, sample_weight=targets.weight)
    baseline = dict(kind='regression', scaler=scaler, model=model)
    selection = dict(source_task='Src', alpha=10., target_AM_used_for_selection=False)
    record = search_fold(frame, matrix, 'Src', 0, baseline, selection, tmp_path)
    candidates = pd.read_csv(tmp_path / 'CANDIDATE_METRICS.csv')
    assert record['selected_source_metric'] == pytest.approx(candidates.rho.max())
    assert candidates.iteration.isna().sum() == 1
    assert (tmp_path / 'head.joblib').is_file()
    assert json.loads((tmp_path / 'SELECTION.json').read_text())['reconstructed_search']
