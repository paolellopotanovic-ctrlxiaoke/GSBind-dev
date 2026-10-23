import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from gsbind.downstream import TASKS, METHODS, SOURCE_TASKS, SELECTION_POLICY, aupr
from gsbind.downstream import load_inputs, load_selected_head, replay, split_masks, transfer_routing


ROOT = Path(__file__).resolve().parents[1]


def test_selected_config_matches_runtime():
    configuration = json.loads((ROOT / 'configs/downstream_selected.json').read_text())
    assert configuration['tasks'] == TASKS
    assert configuration['methods'] == METHODS
    assert configuration['training_sources'] == SOURCE_TASKS
    assert configuration['checkpoint_selection'] == SELECTION_POLICY
    assert not configuration['AM']['training']
    assert not configuration['AM']['target_checkpoint_selection']
    assert not (ROOT / 'checkpoints/downstream/AM').exists()


def test_aupr_is_not_average_precision():
    labels, scores = [1, 0, 1], [.1, .2, .3]
    assert aupr(labels, scores) != pytest.approx(average_precision_score(labels, scores))


@pytest.mark.parametrize('task,rows', [('AlloBench', 216556), ('AM', 1801), ('Src', 538)])
def test_public_inputs_and_source_splits(task, rows):
    frame, features, matrices = load_inputs(ROOT, task)
    assert len(frame) == rows
    assert matrices['gsbind_hidden_probe'].shape[1] == 128
    assert matrices['pesto_score_probe'].shape[1] == 1
    assert matrices['structure_only'].shape[1] == (29 if task == 'Src' else 26)
    if task == 'AM':
        return
    coverage = np.zeros(len(frame), int)
    for fold in range(5):
        training, testing = split_masks(frame, task, fold)
        assert not (training & testing).any()
        coverage += testing
        for method in METHODS:
            head, selected = load_selected_head(ROOT, task, method, fold)
            assert selected['selection'] == SELECTION_POLICY
            assert not selected['target_AM_used_for_selection']
            assert not selected['independent_source_test']
    np.testing.assert_array_equal(coverage, 1)


def test_am_routing_never_uses_target_labels_or_legacy_folds():
    source = pd.DataFrame(dict(component=['a', 'b', 'c', 'd', 'e'], outer_fold=range(5)))
    target = pd.DataFrame(dict(component=['a', 'e', 'new'], label=[0, 1, 0], outer_fold=[4, 0, 3]))
    eligible, independent = transfer_routing(source, target)
    np.testing.assert_array_equal(eligible.sum(axis=1), [1, 1, 5])
    np.testing.assert_array_equal(independent, [False, False, True])
    altered = target.assign(label=1 - target.label, outer_fold=-99)
    changed, changed_independent = transfer_routing(source, altered)
    np.testing.assert_array_equal(changed, eligible)
    np.testing.assert_array_equal(changed_independent, independent)
    for fold in range(5):
        trained = set(source.loc[source.outer_fold.ne(fold), 'component'])
        assert not set(target.loc[eligible[:, fold], 'component']) & trained


def test_source_selection_history_and_hidden_iterations():
    for source, iterations in [('AlloBench', [28, 49, 69, 44, 74]), ('Src', [1, 7, 2, 4, 7])]:
        for fold, iteration in enumerate(iterations):
            head, selection = load_selected_head(ROOT, source, 'gsbind_hidden_probe', fold)
            assert selection['selected_iteration'] == iteration


def test_all_selected_heads_and_transfer_replay(tmp_path):
    actual = replay(ROOT, tmp_path)
    expected = pd.read_csv(ROOT / 'results/downstream/METRICS.csv')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    validation = json.loads((tmp_path / 'VALIDATION.json').read_text())
    assert len(validation['tasks']) == len(TASKS)
    assert validation['checkpoint_selection_uses_source_test']
    assert not validation['target_AM_training']
    assert not validation['target_AM_checkpoint_selection']
    assert not validation['independent_source_test']
    am = pd.read_csv(tmp_path / 'AM/predictions.csv.gz')
    assert len(am) == 1801
    assert am.source_component_disjoint.sum() == 371
    assert am.loc[am.source_component_disjoint, 'eligible_source_heads'].eq(5).all()
