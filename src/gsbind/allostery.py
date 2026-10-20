"""Four frozen linear readouts, grouped evaluation, and fixed-source transfer."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .metrics import binary_metrics, component_difference, bh_adjust


METHODS = ['geometry','amino','hidden','full']
GEOMETRY = ['relative_sasa','contact_density_8A','dssp_helix','dssp_strand','dssp_turn']
AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'
TARGETS = ['AM','ASBench','PASSer_ASD_source','CASBench','ASD_AlloPPI']


def feature_matrices(frame,hidden):
    geometry = frame[GEOMETRY].to_numpy(float)
    amino = np.column_stack([frame.amino_acid.eq(letter) for letter in AMINO_ACIDS]).astype(float)
    hidden = np.asarray(hidden)
    if hidden.shape!=(len(frame),128) or not np.isfinite(hidden).all():
        raise ValueError('Require aligned finite 128-dimensional hidden')
    full = np.column_stack([frame.score.to_numpy(float),geometry,amino,frame.residue_plddt.to_numpy(float),hidden])
    return dict(geometry=geometry,amino=amino,hidden=hidden,full=full)


def load_task(root,task):
    return pd.read_parquet(root/'datasets/allosteric/tasks'/(task+'.parquet')).reset_index(drop=True)


def predictions(root,experiment,models=None,surface=True):
    data = root/'datasets/allosteric'
    model_root = Path(models) if models else root/'checkpoints/allosteric'
    task = experiment.removeprefix('AlloBench_to_')
    source = 'AlloBench' if experiment.startswith('AlloBench_to_') else task
    frame = load_task(root,task)
    membership = pd.read_csv(model_root/source/'fold_membership.csv')
    folds = membership.drop_duplicates('component').set_index('component').outer_fold
    frame['outer_fold'] = frame.component.map(folds).fillna(-1).astype(int)
    if source==task:
        assert frame.outer_fold.ge(0).all()
    frame['transfer_role'] = np.where(frame.outer_fold.eq(-1),'source_component_disjoint_external','source_related_group_held_out') if source!=task else 'task_specific_oof'
    conditions = json.loads((data/'conditions.json').read_text())
    hidden = np.load(data/'hidden_centered_conditions.npy',mmap_mode='r')
    selections = pd.read_csv(model_root/source/'model_selections.csv')
    columns = ['uniprot','component','case_id','residue_index_0','matrix_row','score','label','e11_both80_hit','e11_either80_hit','outer_fold','transfer_role']
    blocks = []
    for fold,group in frame.groupby('outer_fold'):
        native = feature_matrices(group,hidden[0,group.matrix_row])
        for method in METHODS:
            model = joblib.load(model_root/source/((str(fold) if fold>=0 else 'all')+'_'+method+'.joblib'))
            chosen = selections[selections.design.eq(method)]
            threshold = float(chosen[chosen.fold.eq(fold)].threshold.iloc[0]) if fold>=0 else float(chosen.threshold.median())
            block = group[columns].copy().assign(method=method,condition='baseline',threshold=threshold)
            block['prediction'] = model.predict_proba(native[method])[:,1]
            blocks.append(block)
            if method=='full' and surface:
                for index,condition in enumerate(conditions[1:],1):
                    changed = block.copy()
                    changed['condition'] = condition
                    altered = feature_matrices(group,hidden[index,group.matrix_row])['full']
                    changed['prediction'] = model.predict_proba(altered)[:,1]
                    if condition=='identity':
                        np.testing.assert_allclose(changed.prediction,block.prediction,atol=1e-12,rtol=0)
                    blocks.append(changed)
    return pd.concat(blocks,ignore_index=True)


def analyze(predicted,output):
    output.mkdir(parents=True,exist_ok=True)
    records = []
    for (method,condition),current in predicted.groupby(['method','condition'],sort=False):
        for region in ['all','baseline_low']:
            selected = current if region=='all' else current[current.score.lt(.5)]
            for identifier,group in selected.groupby('uniprot',sort=False):
                metrics = binary_metrics(group.label,group.prediction,float(group.threshold.iloc[0]))
                fixed = binary_metrics(group.label,group.prediction,.5)
                records.append(dict(method=method,condition=condition,region=region,uniprot=identifier,component=group.component.iloc[0],
                    e11_both80_hit=bool(group.e11_both80_hit.iloc[0]),e11_either80_hit=bool(group.e11_either80_hit.iloc[0]),
                    mcc_at05=fixed['mcc'],f1_at05=fixed['f1'],**metrics))
    cases = pd.DataFrame(records)
    cases.to_csv(output/'case_metrics.csv',index=False)
    metrics = ['aupr_mixed','auroc','mcc','f1','mcc_at05','f1_at05','top10_hits','top10_precision','top10_recall']
    components = cases.groupby(['method','condition','region','component'])[metrics].mean().reset_index()
    components.to_csv(output/'component_metrics.csv',index=False)
    summary = components.groupby(['method','condition','region'])[metrics].mean().reset_index()
    counts = cases.groupby(['method','condition','region']).agg(proteins=('uniprot','nunique'),mixed_proteins=('aupr_mixed','count'),positions=('positions','sum'),positives=('positives','sum')).reset_index()
    summary = summary.merge(counts,on=['method','condition','region'])
    summary.to_csv(output/'summary.csv',index=False)
    contrasts = []
    for region in ['all','baseline_low']:
        for clean,column in [('all',None),('both_clean','e11_both80_hit'),('either_clean','e11_either80_hit')]:
            selected = cases[cases.region.eq(region)]
            if column:
                selected = selected[~selected[column]]
            if not len(selected):
                continue
            native = selected[selected.condition.eq('baseline')]
            wide = native.pivot(index=['uniprot','component'],columns='method',values='aupr_mixed').reset_index()
            for baseline in ['geometry','amino','hidden']:
                contrasts.append(dict(region=region,e11_overlap_filter=clean,comparison='full_minus_'+baseline,**component_difference(wide,baseline,'full',seed=20261020)))
            surface = selected[selected.method.eq('full')].pivot(index=['uniprot','component'],columns='condition',values='aupr_mixed').reset_index()
            for condition in ['S_retain50','S_retain00','S_noise101','S_noise211','S_noise307']:
                if condition in surface:
                    contrasts.append(dict(region=region,e11_overlap_filter=clean,comparison=condition+'_minus_baseline',**component_difference(surface,'baseline',condition,seed=20261020)))
            noise = ['S_noise101','S_noise211','S_noise307']
            if set(noise+['S_retain00']).issubset(surface.columns):
                surface['mean_noise'] = surface[noise].mean(axis=1)
                contrasts.append(dict(region=region,e11_overlap_filter=clean,comparison='S_retain00_minus_mean_noise',**component_difference(surface,'mean_noise','S_retain00',seed=20261020)))
    statistics = pd.DataFrame(contrasts)
    statistics['q_bh'] = bh_adjust(statistics.p)
    statistics.to_csv(output/'paired_aupr_statistics.csv',index=False)
    budgets = []
    native = predicted[predicted.condition.eq('baseline')]
    for identifier,group in native.groupby('uniprot'):
        for baseline in ['geometry','amino','hidden']:
            enhanced = group[group.method.eq('full')].sort_values(['prediction','residue_index_0'],ascending=[False,True]).head(10)
            reference = group[group.method.eq(baseline)].sort_values(['prediction','residue_index_0'],ascending=[False,True]).head(10)
            labels = group.drop_duplicates('residue_index_0').set_index('residue_index_0').label
            new,old = set(enhanced.residue_index_0),set(reference.residue_index_0)
            added = sorted(index for index in new-old if labels[index])
            lost = sorted(index for index in old-new if labels[index])
            budgets.append(dict(uniprot=identifier,component=group.component.iloc[0],baseline=baseline,budget=min(10,len(labels)),
                full_hits=int(enhanced.label.sum()),baseline_hits=int(reference.label.sum()),added=len(added),lost=len(lost),
                added_positions_0=';'.join(map(str,added)),lost_positions_0=';'.join(map(str,lost))))
    pd.DataFrame(budgets).to_csv(output/'top10_added_lost.csv',index=False)
    return summary,statistics


def run(root,output,models=None,saved_predictions=None):
    output.mkdir(parents=True,exist_ok=True)
    checks = {}
    summaries,contrasts = [],[]
    for experiment in ['AlloBench','AM']+['AlloBench_to_'+target for target in TARGETS]:
        folder = output/experiment
        folder.mkdir(exist_ok=True)
        if saved_predictions is None:
            predicted = predictions(root,experiment,models)
        else:
            predicted = pd.read_parquet(Path(saved_predictions)/experiment/'predictions.parquet')
        predicted.to_parquet(folder/'predictions.parquet',index=False)
        expected = pd.read_parquet(root/'datasets/allosteric/reference_predictions'/(experiment+'.parquet'))
        keys = ['uniprot','residue_index_0','method']
        actual = predicted[predicted.condition.eq('baseline')].set_index(keys).sort_index()
        saved = expected.set_index(keys).sort_index()
        if models is None:
            np.testing.assert_allclose(actual.prediction,saved.prediction,atol=1e-12,rtol=0)
        task = experiment.removeprefix('AlloBench_to_')
        source = 'AlloBench' if experiment.startswith('AlloBench_to_') else task
        model_root = Path(models) if models else root/'checkpoints/allosteric'
        membership = pd.read_csv(model_root/source/'fold_membership.csv')
        assert membership.groupby('component').outer_fold.nunique().eq(1).all()
        independent = predicted[predicted.transfer_role.eq('source_component_disjoint_external')]
        assert not set(independent.component)&set(membership.component)
        for fold,group in predicted[predicted.outer_fold.ge(0)].groupby('outer_fold'):
            assert not set(group.component)&set(membership[membership.outer_fold.ne(fold)].component)
        checks[experiment] = dict(max_native_replay_error=float(abs(actual.prediction-saved.prediction).max()),proteins=int(predicted.uniprot.nunique()),
            source_component_disjoint_proteins=int(independent.uniprot.nunique()),target_labels_used_for_fitting=not experiment.startswith('AlloBench_to_'))
        for subset,frame,current in [('all_evaluable',predicted,folder),('source_component_disjoint',independent,folder/'source_component_disjoint')]:
            if not len(frame):
                continue
            summary,statistics = analyze(frame,current)
            summaries.append(summary.assign(experiment=experiment,subset=subset))
            contrasts.append(statistics.assign(experiment=experiment,subset=subset))
        print(experiment,'complete',flush=True)
    summary = pd.concat(summaries,ignore_index=True)
    statistics = pd.concat(contrasts,ignore_index=True)
    summary.to_csv(output/'ALL_METRICS.csv',index=False)
    statistics.to_csv(output/'ALL_AUPR_CONTRASTS.csv',index=False)
    native = summary[summary.condition.eq('baseline')&summary.region.eq('all')]
    main = native.pivot(index=['experiment','subset','proteins'],columns='method',values='aupr_mixed').reset_index()
    main.columns.name = None
    main[['experiment','subset','proteins']+METHODS].to_csv(output/'MAIN_AUPR_TABLE.csv',index=False)
    native[['experiment','subset','method','proteins','mcc','mcc_at05','f1','f1_at05','auroc']].to_csv(output/'SUPPLEMENTARY_CLASSIFICATION_METRICS.csv',index=False)
    verification = dict(status='passed',methods=METHODS,experiments=checks,full_surface_readout_dimension=155,source_target_group_protection=True,
        prediction_source='saved_predictions' if saved_predictions is not None else 'frozen_head_replay')
    (output/'readout_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return verification
