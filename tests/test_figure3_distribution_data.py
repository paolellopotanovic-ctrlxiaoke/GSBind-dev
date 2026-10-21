import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
SPEC = importlib.util.spec_from_file_location('distribution_data', ROOT/'tools/build_figure3_distribution_data.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_cost_curve_includes_boundary_ties():
    values = study.recall_cost_curve(np.array([1, 0, 1, 0]), np.array([.9, .9, .2, .1]), [.5, .8])
    np.testing.assert_allclose(values, [.5, .75])


def test_cost_curve_is_nondecreasing_and_uses_no_score_threshold_tuning():
    labels = np.array([1, 0, 1, 0, 1])
    values = study.recall_cost_curve(labels, np.array([.2, .5, .6, .1, .9]), np.arange(1, 10)/10)
    assert np.all(np.diff(values)>=0)
    np.testing.assert_allclose(values, study.recall_cost_curve(labels, np.array([.2, .5, .6, .1, .9])*2+3, np.arange(1, 10)/10))
