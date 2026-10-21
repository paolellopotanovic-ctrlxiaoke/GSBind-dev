import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('joint_benefit', ROOT/'tools/build_figure3_joint_benefit.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def example_metrics():
    rows = []
    for system, count, score, cost in [('one', 3, .8, .2), ('two', 1, .4, .6)]:
        for case in range(count):
            for model in ['gaussian_direct128']+study.BASELINES:
                rows.append(dict(system_id=system, case_id=f'{system}-{case}', input_state='af2',
                    model=model, state_ap=score if model=='gaussian_direct128' else .5,
                    recall80_cost=cost if model=='gaussian_direct128' else .4))
    return pd.DataFrame(rows)


def test_joint_benefit_uses_equal_systems_not_case_counts():
    table = study.joint_table(example_metrics())
    summary = study.summarize(table, iterations=100)
    np.testing.assert_allclose(summary.loc[summary.quadrant.eq('both_better'), 'estimate'], .5)
    assert len(table)==6


def test_exact_boundary_is_not_a_strict_joint_win():
    metrics = example_metrics()
    metrics['state_ap'] = .5
    table = study.joint_table(metrics)
    assert table.quadrant.eq('boundary_tie').all()


def test_bootstrap_is_reproducible_and_quadrants_partition():
    table = study.joint_table(example_metrics())
    first = study.summarize(table, iterations=100)
    pd.testing.assert_frame_equal(first, study.summarize(table, iterations=100))
    assert first.groupby(['input_state', 'baseline'])['count'].sum().eq(2).all()
    assert ((first.lower<=first.estimate)&(first.estimate<=first.upper)).all()


def test_missing_comparator_system_is_rejected():
    metrics = example_metrics()
    metrics = metrics[~(metrics.system_id.eq('two')&metrics.model.eq('pesto_official'))]
    with pytest.raises(ValueError, match='all four methods'):
        study.joint_table(metrics)
