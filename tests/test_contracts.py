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
