from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gsbind.functional import region_select, run, weighted_spearman
from gsbind.functional_statistics import weighted_top_effect


ROOT = Path(__file__).resolve().parents[1]


def test_region_all_retains_high_score_positions():
    frame = pd.DataFrame(dict(score=[.1, .8, .9], noncontact_eligible=[True, True, False]))
    assert list(region_select(frame, 'all').index) == [0, 1, 2]
    assert list(region_select(frame, 'high_noncontact').index) == [1]
    assert list(region_select(frame, 'low_noncontact').index) == [0]


def test_weighted_spearman_exact_with_ties():
    prediction = np.array([.2, .2, .8, .3])
    target = np.array([3., 2., 2., 1.])
    weights = np.array([[2, 1, 3, 1], [0, 3, 1, 2]])
    expected = [spearmanr(np.repeat(prediction, row), np.repeat(target, row)).statistic for row in weights]
    np.testing.assert_allclose(weighted_spearman(prediction, target, weights), expected, atol=1e-12)


def test_weighted_budget_exact_with_explicit_repeats():
    prediction = np.array([.9, .3, .6])
    target = np.array([2., 1., 4.])
    weights = np.array([[1, 2, 3], [3, 0, 1]])
    expected = []
    for row in weights:
        repeated_prediction, repeated_target = np.repeat(prediction, row), np.repeat(target, row)
        budget = int(np.ceil(.2 * row.sum()))
        order = np.argsort(-repeated_prediction, kind='stable')[:budget]
        expected.append(repeated_target[order].mean())
    np.testing.assert_allclose(weighted_top_effect(prediction, target, weights, fraction=.2), expected)


def test_allosteric_complete_coverage_and_current_main_table():
    directory = ROOT / 'results/allosteric_strata'
    coverage = pd.read_csv(directory / 'complete_coverage_audit.csv')
    assert len(coverage) == 7
    assert coverage.identical_complete_position_set.all()
    np.testing.assert_array_equal(coverage.complete_positions, coverage.task_positions)
    np.testing.assert_array_equal(coverage.complete_positions, coverage.low_positions + coverage.high_positions)
    assert coverage.high_positions.gt(0).all()
    main = pd.read_csv(ROOT / 'results/allosteric/MAIN_AUPR_TABLE.csv')
    main = main[main.subset.eq('all_evaluable')].set_index('experiment').sort_index()
    strata = pd.read_csv(directory / 'MAIN_AUPR_BY_SCORE_STRATUM.csv')
    strata = strata[strata.region.eq('all')].set_index('experiment').sort_index()
    np.testing.assert_allclose(main[['geometry', 'amino', 'hidden', 'full']],
        strata[['geometry', 'amino', 'hidden', 'full']], atol=1e-12)


def test_all_position_readout_replays_official_results(tmp_path):
    run(ROOT, tmp_path)
    keys = ['version', 'condition', 'region', 'target', 'method', 'dataset']
    expected = pd.read_csv(ROOT / 'results/functional/parent_metrics.csv').set_index(keys).sort_index()
    actual = pd.read_csv(tmp_path / 'parent_metrics.csv').set_index(keys).sort_index()
    pd.testing.assert_frame_equal(expected, actual, atol=1e-12, rtol=1e-12)
    positions = pd.read_csv(tmp_path / 'native_experimental_positions.csv')
    assert len(positions) == 1656
    assert positions.score.ge(.5).any()
    assert positions.groupby('dataset').size().to_dict() == {'KRAS': 1118, 'SRC': 538}
    for region in ['all', 'low_all', 'high_all', 'noncontact']:
        assert region in actual.index.get_level_values('region')


def test_experimental_source_does_not_mix_old_predictions():
    labels = pd.read_csv(ROOT / 'datasets/functional/experimental_residues.csv')
    assert not {'hidden_G', 'G_original', 'full_probe', 'raw_score', 'score'}.intersection(labels.columns)
    features = pd.read_parquet(ROOT / 'datasets/functional/complete_features.parquet')
    predictions = pd.read_parquet(ROOT / 'datasets/functional/readout_predictions.parquet')
    native = predictions[predictions.condition.eq('baseline')]
    aligned = features.merge(native, on=['case_id', 'residue_index_0'], validate='one_to_one')
    np.testing.assert_array_equal(aligned.score_x, aligned.score_y)


def test_score_quintile_membership_uses_all_positions():
    from gsbind.functional_separation import groups
    frame = pd.DataFrame(dict(score=np.linspace(.01, .99, 50), position=np.arange(50),
        functional_magnitude=np.arange(50), stability_magnitude=np.arange(50)))
    parts = groups(frame, 'within_score_quintiles')
    assert len(parts) == 5
    assert sorted(position for _, group, _ in parts for position in group.position) == list(range(50))
    assert all(len(group) == 10 for _, group, _ in parts)


def test_separation_intervals_match_frozen_main_comparison():
    frame = pd.read_csv(ROOT / 'results/functional_separation/paired_block_bootstrap.csv')
    selected = frame[frame.analysis.eq('within_score_quintiles')
        & frame.comparison.eq('full_probe_minus_raw_score')].set_index('dataset')
    assert set(selected.index) == {'KRAS', 'SRC'}
    assert selected.lower.gt(0).all()
    assert selected.iterations.eq(10000).all()
    assert selected.valid_draws.eq(10000).all()
