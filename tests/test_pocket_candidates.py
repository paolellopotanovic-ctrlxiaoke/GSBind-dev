import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases/pocket_screen'


def test_candidate_screen_is_traceable():
    screen = pd.read_csv(DIRECTORY / 'ALL_SOURCE_POCKET_SCREEN.csv')
    assert len(screen) == 1506
    assert screen.uniprot.nunique() == 357
    provenance = json.loads((DIRECTORY / 'CANDIDATE_PROVENANCE.json').read_text())
    assert len(provenance['cases']) == 3
    for case in provenance['cases']:
        assert case['source_annotation_mapping_matches']
        assert case['ligand_heavy_atoms'] > 0


def test_candidate_selection_and_spatial_counts():
    summary = pd.read_csv(DIRECTORY / 'CANDIDATE_SPATIAL_SUMMARY.csv')
    for pdb in ['2XPU', '1ZA2', '1H9G']:
        rows = pd.read_csv(DIRECTORY / pdb / 'TOPK_RESIDUES.csv')
        original = pd.read_csv(DIRECTORY / pdb / 'ALL_PROTEIN_PREDICTIONS.csv')
        for record in summary.loc[summary.pdb.eq(pdb)].itertuples():
            current = rows.loc[rows.fraction.eq(record.fraction) & rows.method.eq(record.method)]
            ordered = original.sort_values([record.method,'residue_index_0'], ascending=[False,True]).head(record.top_k)
            selected = set(ordered.residue_index_0+1)
            np.testing.assert_array_equal(current.selected,current.canonical_position.isin(selected))
            assert current.category.eq('hit').sum()==record.hits
            mapped = current.loc[current.selected & current.identity_match]
            assert mapped.ligand_min_distance.le(6).sum()==record.selected_within_6A
            np.testing.assert_allclose(record.precision,record.hits/record.top_k)
    hidden = summary.loc[summary.method.eq('gsbind_hidden_probe') & summary.fraction.eq(.05)].set_index('pdb')
    assert hidden.loc['2XPU','hits']==8
    assert hidden.loc['1ZA2','hits']==6


def test_candidate_render_metadata_and_files():
    rendering=json.loads((DIRECTORY / 'PYMOL_RENDER.json').read_text())
    assert rendering['same_camera_across_methods_and_thresholds']
    assert rendering['renderer']=='pymol-open-source'
    for pdb in ['2XPU','1ZA2','1H9G']:
        assert (DIRECTORY / (pdb+'_pocket_comparison.png')).stat().st_size>10000
        assert (DIRECTORY / pdb / 'top5_gsbind_hidden_probe.pse').is_file()


def test_atcase_extra_top5_positions_contact_regulatory_partner():
    contacts=pd.read_csv(DIRECTORY / '1ZA2/TOP5_BIOLOGICAL_CONTACTS.csv')
    off=contacts.loc[~contacts.source_pocket].set_index('pdb_resi')
    assert set(off.index)=={24,27}
    assert off.contact_4A.all()
    assert off.nearest_chain.eq('D').all()
    assert off.nearest_operation.eq(1).all()
    assert off.ligand_distance_A.gt(15).all()
    assert not contacts.loc[contacts.source_pocket,'contact_5A'].any()
    provenance=json.loads((DIRECTORY / '1ZA2/CONTACT_PROVENANCE.json').read_text())
    assert 'biological assembly' in provenance['source_of_partners']


def test_interface_aware_colors_preserve_prediction_and_pocket_labels():
    original=pd.read_csv(DIRECTORY / '1ZA2/TOPK_RESIDUES.csv')
    colored=pd.read_csv(DIRECTORY / '1ZA2/INTERFACE_AWARE_TOPK.csv')
    for column in ['pdb_resi','source_site','selected','method','fraction','top_k']:
        pd.testing.assert_series_equal(original[column],colored[column])
    hidden=colored.loc[colored.method.eq('gsbind_hidden_probe') & colored.fraction.eq(.05)]
    assert hidden.display_category.eq('pocket_hit').sum()==6
    assert set(hidden.loc[hidden.display_category.eq('interface_hit'),'pdb_resi'])=={24,27}
    assert hidden.display_category.eq('other_high').sum()==0
    assert colored.loc[colored.display_category.eq('interface_context'),'selected'].eq(False).all()
    definition=json.loads((DIRECTORY / '1ZA2/INTERFACE_FIGURE.json').read_text())
    assert definition['independent_of_prediction']
    assert definition['allosteric_labels_unchanged']
