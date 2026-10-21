import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('surface_figure',ROOT/'tools/build_figure3_gaussian_surface.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_identity_preserves_ranking_and_correct_candidates():
    labels=np.array([1,0,1,0,0,1,0,0,0,0])
    score=np.linspace(.95,.05,10)
    result=study.case_readouts(labels,score,score)
    assert result['correct_retained_per_budget']==1
    assert result['correct_lost_per_budget']==0
    assert result['interface_priority_loss']==0
    assert result['background_priority_loss']==0


def test_monotone_score_shift_is_not_rank_damage():
    labels=np.array([1,0,1,0,0,1,0,0,0,0])
    score=np.linspace(.9,.1,10)
    result=study.case_readouts(labels,score,score*.5)
    native=study.case_readouts(labels,score,score)
    assert result['ap']==native['ap']
    assert result['correct_lost_per_budget']==0
    assert result['interface_priority_loss']==0
    assert result['native_candidate_score_drop']>0


def test_rank_loss_has_correct_interface_and_background_direction():
    labels=np.array([1,1,0,0])
    result=study.case_readouts(labels,np.array([.9,.8,.2,.1]),np.array([.1,.2,.8,.9]))
    assert result['interface_priority_loss']>0
    assert result['background_priority_loss']<0
    assert result['correct_lost_per_budget']==1


def test_lost_correct_candidates_differs_from_precision_drop():
    labels=np.array([1,1,0,0,0,0,0,0,0,0])
    result=study.case_readouts(labels,np.array([.9,.8,.7,.6,.5,.4,.3,.2,.1,0]),np.array([.8,.9,.7,.6,.5,.4,.3,.2,.1,0]))
    assert result['top_fraction_precision']==1
    assert result['correct_lost_per_budget']==1


def test_system_bootstrap_keeps_unequal_cases_equal_by_system():
    table=pd.DataFrame(dict(input_state=['af2']*4,case_id=['a1','a2','a3','b1'],
        system_id=['a','a','a','b'],condition=['identity']*4,damage=[0,0,0,1]))
    summary=study.bootstrap_summary(table,['input_state','condition'],['damage']).iloc[0]
    assert summary.estimate==.5
    assert summary.systems==2
    assert summary.cases==4


def test_top_budget_is_prediction_defined_and_deterministic_under_ties():
    labels=np.array([1,0,1,0,1,0,1,0,1,0])
    result=study.case_readouts(labels,np.ones(10),np.ones(10))
    assert result['top_fraction_precision']==1
    assert result['correct_lost_per_budget']==0
