import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('figure3_binding', ROOT/'tools/build_figure3_binding_reliability.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_sequence_mapping_does_not_depend_on_pdb_numbers():
    first, second, report = study.sequence_mapping('ACDEFGHIKLMN', 'XXACDEFGHIKLMNYY')
    np.testing.assert_array_equal(first, np.arange(12))
    np.testing.assert_array_equal(second, np.arange(2, 14))
    assert report['identity']==1.


def test_kabsch_removes_rigid_rotation_and_translation():
    generator = np.random.default_rng(77)
    geometry = generator.normal(size=(20, 3))
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    transformed = geometry@rotation+[12., -3., 5.]
    aligned, shifts = study.superpose(transformed, geometry)
    np.testing.assert_allclose(aligned, geometry, atol=1e-12)
    np.testing.assert_allclose(shifts, 0., atol=1e-12)


def test_recall_cost_includes_all_boundary_ties():
    labels = np.array([1, 0, 1, 0])
    score = np.array([.9, .9, .2, .1])
    assert study.recall_cost(labels, score, .5)==.5
    assert study.recall_cost(labels, score, .8)==.75


def test_consistency_does_not_count_shared_wrong_candidates_as_correct():
    labels = np.array([0, 0, 1, 1])
    score = np.array([.9, .8, .2, .1])
    values = study.stable_candidates(labels, score, score, 2)
    assert values['jaccard']==1.
    assert values['retained_correct_per_budget']==0.


def test_packing_auc_uses_only_within_stratum_pairs():
    labels = np.array([0, 1, 0, 1])
    score = np.array([.1, .2, .8, .9])
    auc, support = study.packing_auc(labels, score, np.array([2, 2, 10, 10]))
    assert auc==1.
    assert sum(row['negative_positive_pairs'] for row in support)==2


def test_region_selection_does_not_depend_on_labels():
    distances = np.abs(np.arange(10)[:, None]-np.arange(10)[None, :])*5.
    first = study.candidate_regions(np.array([1, 0]*5), np.linspace(0., 1., 10), distances)
    second = study.candidate_regions(np.array([0, 1]*5), np.linspace(0., 1., 10), distances)
    assert [row['centers'] for row in first]==[row['centers'] for row in second]


def test_system_mean_is_not_directed_case_weighted():
    import pandas as pd
    table = pd.DataFrame(dict(system_id=['one', 'one', 'two'], effect=[0., 0., 1.]))
    result = study.mean_ci(table, 'effect', 100)
    assert result['estimate']==.5
    assert result['case_mean']==1/3


def test_modified_amino_acid_is_read_from_hetatm(tmp_path, monkeypatch):
    pdb = tmp_path/'mse.pdb'
    pdb.write_text('HETATM    1  CA  MSE A 121       1.000   2.000   3.000  1.00 90.00           C  \n')
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    sequence, geometry, confidence, numbers = study.read_structure(dict(receptor_pdb_path='mse.pdb',
        receptor_chain_id='A', input_residue_ids='121', num_residues=1))
    assert sequence=='M' and numbers==['121']
    np.testing.assert_allclose(geometry, [[1., 2., 3.]])
    np.testing.assert_allclose(confidence, [90.])
