"""Independently check saved intervention masks and the new figure metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figure3_gaussian_surface_20261021'
SOURCE=ROOT.parent/'gaussian-direct128/results/official_mmseq_gaussian_cutoff_e11_20261019'


def validate_geometry(folder,source):
    roster=pd.read_csv(folder/'intervention_roster.csv')
    rows=[]
    for record in roster.itertuples():
        stem=hashlib.sha256(record.case_id.encode()).hexdigest()[:20]
        path=source/'cases'/stem/'actual_outputs.npz'
        with np.load(path) as packet:
            centers=packet['centers']
            surface=packet['surface_positions']
            top=packet['top_indices']
            target=packet['P_top'].astype(bool)
            expected=cKDTree(centers[top]).query(surface)[0]<=8.
            np.testing.assert_array_equal(target,expected)
            assert len(top)==int(np.ceil(.1*len(centers)))
            for seed in [101,211,307]:
                assert packet['P_random'+str(seed)].sum()==target.sum()
            rows.append(dict(case_id=record.case_id,input_state=record.input_state,
                input_residues=len(centers),target_residues=len(top),surface_points=len(surface),
                target_surface_points=int(target.sum()),target_surface_fraction=float(target.mean()),
                all_three_random_masks_equal_count=True,target_equals_native_top10pct_8A_neighborhood=True))
    pd.DataFrame(rows).to_csv(folder/'local_intervention_mask_audit.csv',index=False)
    return len(rows)


def validate_readouts(folder):
    metrics=pd.read_csv(folder/'intervention_case_metrics.csv')
    pairs=pd.read_csv(folder/'pair_model_metrics.csv')
    stability=pd.read_csv(folder/'stability_case_metrics.csv')
    assert metrics.groupby('case_id').condition.nunique().eq(12).all()
    assert metrics[metrics.condition.eq('identity')].ap_loss.abs().max()<1e-15
    assert pairs.groupby('case_id').model.nunique().eq(4).all()
    assert pairs[['case_id','input_state']].drop_duplicates().input_state.value_counts().to_dict()=={'af2':718,'apo':70}
    top=stability[stability.requested_k.eq(0)]
    np.testing.assert_allclose(top.actual_k,np.ceil(.1*top.common_positions))
    assert top.retained_correct_count.le(top.actual_k).all()
    np.testing.assert_allclose(top.retained_correct_per_budget,top.retained_correct_count/top.actual_k,atol=1e-15)
    summary=pd.read_csv(folder/'intervention_summary.csv')
    for row in summary.itertuples():
        if row.condition=='P_random_mean':
            current=metrics[metrics.condition.isin(['P_random101','P_random211','P_random307'])]
            current=current.groupby(['case_id','system_id','input_state'],as_index=False).mean(numeric_only=True)
        else:
            current=metrics[metrics.condition.eq(row.condition)]
        expected=current[current.input_state.eq(row.input_state)].groupby('system_id')[row.metric].mean().mean()
        assert abs(expected-row.estimate)<1e-14
    selection=json.loads((folder/'F_EXAMPLE_SELECTION.json').read_text())
    residues=pd.read_csv(folder/'F_example_residues.csv')
    residues=residues[residues.label_mask.astype(bool)]
    example=metrics[metrics.case_id.eq(selection['case_id'])].set_index('condition')
    native=residues.baseline.to_numpy()
    for condition in ['baseline','P_top','surface_zero']:
        score=residues[condition].to_numpy()
        priorities=rankdata(score,method='average')/len(score)
        np.testing.assert_allclose(residues[condition+'_rank'],priorities,atol=1e-15)
        lost=(rankdata(native)-rankdata(score))/len(score)
        assert abs(lost[residues.label.eq(1)].mean()-example.loc[condition,'interface_priority_loss'])<1e-15
    assert len(summary)==3*13*8
    residues=pd.read_parquet(folder/'raw/paired_residue_predictions.parquet')
    paired_lookup=pairs.set_index(['case_id','model'])
    retention_lookup=top.set_index(['case_id','model'])
    replays=0
    for case,group in residues.groupby('case_id',sort=False):
        labels=group.reference_label.to_numpy(int)
        budget=int(np.ceil(.1*len(group)))
        for model in ['gaussian_direct128','pesto_official','gpsfun_gpsite_pdb','scannet_ppi_no_msa']:
            state=group[model+'_state_score'].to_numpy()
            holo=group[model+'_holo_score'].to_numpy()
            expected=average_precision_score(labels,state)
            assert abs(expected-paired_lookup.loc[(case,model),'state_ap'])<1e-14
            chosen_state=np.argsort(-state,kind='stable')[:budget]
            chosen_holo=np.argsort(-holo,kind='stable')[:budget]
            common=np.intersect1d(chosen_state,chosen_holo)
            assert int(labels[common].sum())==retention_lookup.loc[(case,model),'retained_correct_count']
            replays+=1
    return dict(summary_rows_replayed=len(summary),paired_cases=pairs.case_id.nunique(),
        paired_four_models_same_cases=True,paired_residue_metric_replays=replays,
        correct_candidate_counts_replayed=True,example_priorities_replayed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=OUT)
    parser.add_argument('--source-dir',type=Path,default=SOURCE)
    arguments=parser.parse_args()
    report=validate_readouts(arguments.data_dir)
    report['geometry_cases_audited']=validate_geometry(arguments.data_dir,arguments.source_dir)
    report['status']='passed'
    (arguments.data_dir/'INDEPENDENT_FIGURE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
