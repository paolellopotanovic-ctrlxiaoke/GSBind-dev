import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'figures/structural_cases'


def test_abl1_case_mapping_matches_original_annotation():
    mapped = pd.read_csv(CASES / 'ABL1_RESIDUE_MAPPING.csv')
    original = pd.read_csv(ROOT / 'datasets/allosteric/allobench_record_mapping.csv')
    positions = original.loc[original.uniprot.eq('P00519') & original.source_pdb.eq('5MO4'), 'positions_0'].iloc[0]
    np.testing.assert_array_equal(sorted(mapped.loc[mapped.site_5mo4, 'canonical_position'] - 1),
                                   sorted(map(int, positions.split(';'))))
    assert len(mapped) == 429
    assert set(mapped.loc[~mapped.identity_match, 'pdb_resi']) == {334, 382}
    predictions = pd.read_csv(CASES / 'ABL1_ALL_EVALUATED_PREDICTIONS.csv').set_index('residue_index_0')
    np.testing.assert_allclose(mapped.hidden_percentile,
                               predictions.loc[mapped.canonical_position - 1, 'gsbind_hidden_probe_percentile'])


def test_src_case_overlay_matches_saved_oof():
    overlay = pd.read_csv(CASES / 'SRC_POSITION_OVERLAYS.csv')
    original = pd.read_csv(ROOT / 'results/downstream/Src/predictions.csv.gz')
    assert len(overlay) == 538
    for column in ['case_id', 'physical_position', 'functional_magnitude', 'gsbind_hidden_probe']:
        pd.testing.assert_series_equal(overlay[column], original[column])
    np.testing.assert_array_equal(overlay.activation_segment, overlay.physical_position.between(407, 435))
    annotation = json.loads((CASES / 'SRC_ANNOTATIONS.json').read_text())
    assert annotation['no_activation_state_assignment']


def test_case_rendering_is_open_source_and_has_sessions():
    rendering = json.loads((CASES / 'PYMOL_RENDER.json').read_text())
    assert rendering['ray_traced']
    assert rendering['identical_view_within_comparisons']
    assert rendering['renderer_distribution'] == 'pymol-open-source'
    for name in ['abl1_annotation', 'abl1_hidden', 'src_activation_segment', 'src_hidden']:
        assert (CASES / (name + '.pse')).is_file()
        assert (CASES / (name + '.png')).stat().st_size > 10000


def test_abl1_topk_is_fixed_full_protein_selection():
    summary = pd.read_csv(CASES / 'ABL1_TOPK_SUMMARY.csv')
    rows = pd.read_csv(CASES / 'ABL1_TOPK_RESIDUES.csv')
    original = pd.read_csv(CASES / 'ABL1_ALL_EVALUATED_PREDICTIONS.csv')
    assert len(summary) == 6
    for row in summary.itertuples():
        ranked = original.sort_values([row.method, 'residue_index_0'], ascending=[False, True]).head(row.top_k)
        expected = set(ranked.residue_index_0 + 1)
        current = rows.loc[rows.fraction.eq(row.fraction) & rows.method.eq(row.method)]
        np.testing.assert_array_equal(current.selected, current.canonical_position.isin(expected))
        assert current.category.eq('hit').sum() == row.mapped_hits
        assert current.category.eq('off_annotation_high').sum() == row.mapped_off_annotation
        assert current.loc[~current.identity_match, 'category'].eq('sequence_mismatch').all()
        assert row.full_protein_hits == int(ranked.label.sum())
        np.testing.assert_allclose(row.full_protein_precision, row.full_protein_hits/row.top_k)
    hidden = summary.loc[summary.method.eq('gsbind_hidden_probe')].sort_values('fraction')
    assert hidden.full_protein_hits.tolist() == [14, 18]


def test_src_case_figure_highlights_actual_complete_oof_gain():
    from scipy.stats import spearmanr

    metadata = json.loads((CASES / 'SRC_CASE_FIGURE.json').read_text())
    assert metadata['no_retraining']
    assert metadata['no_site_selection_or_color_remapping']
    original = pd.read_csv(ROOT / 'results/downstream/Src/predictions.csv.gz')
    summary = pd.read_csv(CASES / 'SRC_CASE_AGREEMENT.csv')
    assert len(summary) == 4
    for row in summary.itertuples():
        group = original.loc[original.case_id.eq(row.case_id)]
        assert len(group) == row.positions == 269
        np.testing.assert_allclose(row.rho, spearmanr(group.functional_magnitude, group[row.method]).statistic)
    np.testing.assert_allclose(metadata['absolute_delta_rho'],
                               metadata['hidden_rho'] - metadata['structure_rho'])
    np.testing.assert_allclose(metadata['absolute_delta_rho'], .1561005382011872)
    assert (CASES / 'src_functional_comparison.pdf').stat().st_size > 10000


def test_src_spatial_gain_uses_all_positions_and_preserves_regressions():
    original = pd.read_csv(ROOT / 'results/downstream/Src/predictions.csv.gz')
    original = original.loc[original.case_id.eq('src_kd_af2')].reset_index(drop=True)
    mapped = pd.read_csv(CASES / 'SRC_SPATIAL_GAIN.csv')
    metadata = json.loads((CASES / 'SRC_SPATIAL_GAIN.json').read_text())
    assert len(mapped) == len(original) == metadata['positions'] == 269
    np.testing.assert_array_equal(mapped.physical_position, original.physical_position)
    experimental_rank = original.functional_magnitude.rank(pct=True)
    structure_error = (original.structure_only.rank(pct=True) - experimental_rank).abs()
    hidden_error = (original.gsbind_hidden_probe.rank(pct=True) - experimental_rank).abs()
    expected = structure_error - hidden_error
    np.testing.assert_allclose(mapped.rank_error_reduction, expected, atol=1e-14)
    np.testing.assert_allclose(metadata['symmetric_color_limit'], expected.abs().max())
    np.testing.assert_allclose(metadata['structure_MAE'], structure_error.mean())
    np.testing.assert_allclose(metadata['hidden_MAE'], hidden_error.mean())
    assert (metadata['improved'], metadata['worsened'], metadata['unchanged']) == (157, 111, 1)
    assert metadata['zero_centered_symmetric_scale']
    assert metadata['source_test_selected']
    assert metadata['spatial_map_is_not_a_spearman_decomposition']
    for filename in ['src_spatial_gain.pdf', 'src_spatial_gain.png', 'src_spatial_gain_render.pse']:
        assert (CASES / filename).stat().st_size > 10000
