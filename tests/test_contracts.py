import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score, matthews_corrcoef

from gsbind.metrics import binary_metrics, component_difference
from gsbind.readouts import centered_hidden
from gsbind.surface import SurfaceContextCutoff


def test_binary_metrics():
    labels = np.array([1, 0, 1, 0, 1])
    scores = np.array([.9, .5, .8, .1, .2])
    result = binary_metrics(labels, scores)
    assert result['mcc'] == pytest.approx(matthews_corrcoef(labels, scores >= .5))
    assert result['f1'] == pytest.approx(f1_score(labels, scores >= .5))
    assert result['tp'] == 2
    assert result['fp'] == 1


def test_single_class_is_explicit():
    result = binary_metrics([0, 0], [.1, .2])
    assert np.isnan(result['aupr_mixed'])
    assert result['mcc'] == 0
    assert not result['mcc_defined']


def test_unknown_is_not_negative():
    result = binary_metrics([1, np.nan, 0], [.9, .95, .1])
    assert result['positions'] == 2
    assert result['f1'] == 1


def test_complete_protein_centering():
    frame = pd.DataFrame(dict(case_id=['protein']*3, residue_index_0=[0, 1, 2], protein_length=[3]*3, score=[.1, .2, .8]))
    hidden = np.arange(384, dtype=np.float32).reshape(3, 128)
    result = centered_hidden(frame, hidden)
    np.testing.assert_array_equal(result, hidden-hidden[:2].mean(axis=0))
    with pytest.raises(ValueError):
        centered_hidden(frame.iloc[:2], hidden[:2])


def test_surface_cut_preserves_residue_half():
    original = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    controller = SurfaceContextCutoff(None, 'S_retain00')
    result = controller.replace((original,), 0)[0]
    assert torch.equal(result[:, :4], original[:, :4])
    assert torch.count_nonzero(result[:, 4:]) == 0
    identity = SurfaceContextCutoff(None, 'identity').replace((original,), 0)[0]
    assert identity is original


def test_surface_noise_is_norm_matched():
    original = torch.randn(5, 256)
    result = SurfaceContextCutoff(None, 'S_noise101').replace((original,), 1)[0]
    torch.testing.assert_close((result[:, 128:]-original[:, 128:]).norm(dim=1), original[:, 128:].norm(dim=1))
    assert torch.equal(result[:, :128], original[:, :128])


def test_bootstrap_units_are_components():
    frame = pd.DataFrame(dict(component=['one', 'one', 'two'], basic=[0, 0, 0], hidden=[1, 1, 0]))
    result = component_difference(frame, 'basic', 'hidden', iterations=100)
    assert result['components'] == 2
    assert result['mean'] == .5


def test_four_allosteric_designs_are_exact():
    from gsbind.allostery import feature_matrices, METHODS
    frame = pd.DataFrame(dict(score=[.1,.9],amino_acid=['A','R'],relative_sasa=[.1,.4],residue_plddt=[90,95],contact_density_8A=[4,9],dssp_helix=[1,0],dssp_strand=[0,1],dssp_turn=[0,0]))
    hidden = np.arange(256,dtype=float).reshape(2,128)
    features = feature_matrices(frame,hidden)
    assert list(features)==METHODS==['geometry','amino','hidden','full']
    for method,width in [('geometry',5),('amino',20),('hidden',128),('full',155)]:
        assert features[method].shape==(2,width)
    np.testing.assert_array_equal(features['hidden'],hidden)
    np.testing.assert_array_equal(features['full'][:,27:],hidden)
    modified = feature_matrices(frame.assign(score=[.4,.3],residue_plddt=[5,9]),hidden)
    for method in ['geometry','amino','hidden']:
        np.testing.assert_array_equal(features[method],modified[method])
    modified = feature_matrices(frame.assign(amino_acid=['W','W']),hidden)
    np.testing.assert_array_equal(features['geometry'],modified['geometry'])


def test_saved_allosteric_summary_never_loads_hidden(tmp_path, monkeypatch):
    from gsbind import allostery

    monkeypatch.setattr(allostery, 'TARGETS', [])

    def reject_replay(*args, **kwargs):
        raise AssertionError('Saved-prediction summary must not load hidden or refit heads')

    def summarize(predicted, output):
        records = [dict(method=method, condition='baseline', region='all', proteins=1,
                        aupr_mixed=.5, mcc=0., mcc_at05=0., f1=0., f1_at05=0., auroc=.5)
                   for method in allostery.METHODS]
        return pd.DataFrame(records), pd.DataFrame([dict(comparison='fixture')])

    monkeypatch.setattr(allostery, 'predictions', reject_replay)
    monkeypatch.setattr(allostery, 'analyze', summarize)
    for task in ['AlloBench', 'AM']:
        records = [dict(uniprot='fixture', residue_index_0=position, method=method,
                        prediction=.2 + .6 * position, condition='baseline',
                        transfer_role='task_specific_oof', component='fixture', outer_fold=0)
                   for method in allostery.METHODS for position in [0, 1]]
        frame = pd.DataFrame(records)
        saved = tmp_path / 'saved' / task
        saved.mkdir(parents=True)
        frame.to_parquet(saved / 'predictions.parquet', index=False)
        references = tmp_path / 'datasets/allosteric/reference_predictions'
        references.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(references / (task + '.parquet'), index=False)
        models = tmp_path / 'checkpoints/allosteric' / task
        models.mkdir(parents=True)
        pd.DataFrame([dict(uniprot='fixture', component='fixture', outer_fold=0)]).to_csv(models / 'fold_membership.csv', index=False)
    result = allostery.run(tmp_path, tmp_path / 'output', saved_predictions=tmp_path / 'saved')
    assert result['status'] == 'passed'
    assert result['prediction_source'] == 'saved_predictions'
    assert set(result['experiments']) == {'AlloBench', 'AM'}


def test_frozen_four_method_model_dimensions():
    from pathlib import Path
    import joblib
    from gsbind.allostery import METHODS
    root = Path(__file__).resolve().parents[1]
    for task in ['AlloBench','AM']:
        membership = pd.read_csv(root/'checkpoints/allosteric'/task/'fold_membership.csv')
        assert membership.groupby('component').outer_fold.nunique().eq(1).all()
        assert membership.outer_fold.nunique()==5
        for method,width in zip(METHODS,[5,20,128,155]):
            model = joblib.load(root/'checkpoints/allosteric'/task/('0_'+method+'.joblib'))
            assert model.n_features_in_==width
