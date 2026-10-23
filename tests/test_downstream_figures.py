from pathlib import Path

import numpy as np
import pandas as pd

from gsbind.downstream import METHODS
from gsbind.downstream_figures import SCOPES, build_figure, build_pr_figure, source_tables


ROOT = Path(__file__).resolve().parents[1]


def test_figure_component_means_and_pairs_match_report():
    tables = source_tables(ROOT)
    metrics = pd.read_csv(ROOT / 'results/downstream/METRICS.csv')
    for (task, scope), count in zip(SCOPES, [282, 112, 25]):
        components = tables['component_AUPR']
        rows = components.loc[components.task.eq(task) & components.scope.eq(scope)]
        assert rows.groupby('method').component.nunique().eq(count).all()
        for method in METHODS:
            expected = metrics.loc[metrics.task.eq(task) & metrics.scope.eq(scope) & metrics.method.eq(method), 'component_macro_AUPR'].iloc[0]
            np.testing.assert_allclose(rows.loc[rows.method.eq(method), 'AUPR'].mean(), expected)
        deltas = tables['paired_component_deltas']
        paired = deltas.loc[deltas.task.eq(task) & deltas.scope.eq(scope) & deltas.comparator.eq('structure_only')]
        assert len(paired) == count
        values = rows.pivot(index='component', columns='method', values='AUPR')
        np.testing.assert_allclose(paired.set_index('component').delta_AUPR.sort_index(),
                                   (values.gsbind_hidden_probe - values.structure_only).sort_index())


def test_src_figure_uses_two_complete_constructs():
    tables = source_tables(ROOT)
    constructs = tables['src_construct_rho']
    assert len(constructs) == 8
    assert constructs.positions.eq(269).all()
    positions = tables['src_position_rank_scatter']
    assert len(positions) == 538
    for interaction, group in positions.groupby('interaction'):
        expected = constructs.loc[constructs.interaction.eq(interaction) & constructs.method.eq('gsbind_hidden_probe'), 'rho'].iloc[0]
        np.testing.assert_allclose(group.functional_rank.corr(group.hidden_prediction_rank), expected)
    gains = tables['src_functional_gains']
    structural = gains.loc[gains.comparator.eq('structure_only')]
    assert len(structural) == 2
    assert structural.delta_rho.gt(0).all()
    np.testing.assert_allclose(structural.delta_rho.mean(), .11222820223541515)


def test_rich_figure_exports_all_source_tables(tmp_path):
    tables = build_figure(ROOT, tmp_path)
    for name in tables:
        assert (tmp_path / 'source_data/downstream_selected' / (name + '.csv')).is_file()
    for suffix in ['pdf', 'png']:
        assert (tmp_path / ('downstream_selected_transfer.' + suffix)).stat().st_size > 10000
        for name in ['downstream_results_four_panel', 'downstream_A_AlloBench',
                     'downstream_B_AM', 'downstream_C_Src', 'downstream_D_pooled_AUPR']:
            assert (tmp_path / (name + '.' + suffix)).stat().st_size > 10000
    bars = pd.read_csv(tmp_path / 'source_data/downstream_selected/pooled_AUPR_bars.csv')
    metrics = pd.read_csv(ROOT / 'results/downstream/METRICS.csv')
    assert len(bars) == 8
    for row in bars.itertuples():
        expected = metrics.loc[metrics.task.eq(row.task) & metrics.scope.eq('all_positions')
                               & metrics.method.eq(row.method), 'pooled_AUPR'].iloc[0]
        np.testing.assert_allclose(row.pooled_AUPR, expected)


def test_publication_panels_have_no_scatter_and_preserve_bar_values():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from gsbind.downstream_publication import draw_panel

    tables = source_tables(ROOT)
    metrics = pd.read_csv(ROOT / 'results/downstream/METRICS.csv')
    for panel in ['AlloBench', 'AM', 'Src', 'pooled_AUPR']:
        figure, axis = plt.subplots()
        draw_panel(axis, panel, tables, metrics)
        assert not any(isinstance(collection, PathCollection) for collection in axis.collections)
        if panel in ['AlloBench', 'AM']:
            assert len(axis.collections) == len(METHODS)
            assert len(axis.patches) == len(METHODS)
            for patch in axis.patches:
                np.testing.assert_allclose(patch.get_facecolor(), [1, 1, 1, 1])
        elif panel == 'Src':
            expected = tables['src_construct_rho'].groupby('method').rho.mean().loc[METHODS]
            np.testing.assert_allclose([patch.get_height() for patch in axis.patches], expected)
        else:
            expected = [metrics.loc[metrics.task.eq(task) & metrics.scope.eq('all_positions')
                                    & metrics.method.eq(method), 'pooled_AUPR'].iloc[0]
                        for method in METHODS for task in ['AlloBench', 'AM']]
            np.testing.assert_allclose([patch.get_height() for patch in axis.patches], expected)
            assert len(axis.lines) == 2
            for line, task in zip(axis.lines, ['AlloBench', 'AM']):
                labels = pd.read_csv(ROOT / 'results/downstream' / task / 'predictions.csv.gz',
                                     usecols=['label']).label
                np.testing.assert_allclose(line.get_ydata(), labels.mean())
            assert len(axis.texts) == 8
            assert all(text.get_rotation() == 0 for text in axis.texts)
        plt.close(figure)


def test_pooled_pr_curves_match_saved_areas(tmp_path):
    from sklearn.metrics import auc

    curves, summary = build_pr_figure(ROOT, tmp_path)
    assert len(summary) == 8
    for row in summary.itertuples():
        current = curves.loc[curves.task.eq(row.task) & curves.method.eq(row.method)]
        np.testing.assert_allclose(auc(current.recall, current.precision), row.pooled_AUPR)
        assert current.recall.iloc[0] == 1
        assert current.recall.iloc[-1] == 0
        np.testing.assert_allclose(row.prevalence, row.positives / row.positions)
    assert (tmp_path / 'downstream_pooled_pr.png').is_file()
    assert (tmp_path / 'source_data/downstream_pooled_pr/PR_CURVES.csv.gz').is_file()
