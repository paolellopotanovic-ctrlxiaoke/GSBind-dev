import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from gsbind.downstream import TASKS, METHODS, FIXED_ITERATIONS, aupr, load_inputs, replay, split_masks


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_config_matches_runtime():
    configuration = json.loads((ROOT / 'configs/downstream_fixed.json').read_text())
    assert configuration['tasks'] == TASKS
    assert configuration['methods'] == METHODS
    assert configuration['fixed_iterations'] == FIXED_ITERATIONS


def test_aupr_is_not_average_precision():
    labels, scores = [1, 0, 1], [.1, .2, .3]
    assert aupr(labels, scores) != pytest.approx(average_precision_score(labels, scores))


@pytest.mark.parametrize('task,rows', [('AlloBench', 216556), ('AM', 1801), ('Src', 538)])
def test_public_inputs_and_splits(task, rows):
    frame, features, matrices = load_inputs(ROOT, task)
    assert len(frame) == rows
    assert matrices['gsbind_hidden_probe'].shape[1] == 128
    assert matrices['pesto_score_probe'].shape[1] == 1
    assert matrices['structure_only'].shape[1] == (29 if task == 'Src' else 26)
    coverage = np.zeros(len(frame), int)
    for fold in range(5):
        training, testing = split_masks(frame, task, fold)
        assert not (training & testing).any()
        coverage += testing
        for method in METHODS:
            selected = json.loads((ROOT / 'checkpoints/downstream' / task / method / f'fold{fold}/SELECTION.json').read_text())
            assert selected['requested_iterations'] == FIXED_ITERATIONS[task]
            assert 1 <= selected['actual_iterations'] <= FIXED_ITERATIONS[task]
            assert selected['selection'] == 'fixed_iteration_no_test_checkpoint_selection'
    np.testing.assert_array_equal(coverage, 1)


def test_all_fixed_heads_replay(tmp_path):
    actual = replay(ROOT, tmp_path)
    expected = pd.read_csv(ROOT / 'results/downstream/METRICS.csv')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    validation = json.loads((tmp_path / 'VALIDATION.json').read_text())
    assert len(validation['tasks']) == len(TASKS)
    assert validation['no_test_checkpoint_selection']
    assert not validation['independent_test']
