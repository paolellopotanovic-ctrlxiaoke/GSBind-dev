"""Nested component-grouped logistic probes; never retrains the GSbind model."""

from collections import Counter
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .allostery import METHODS, feature_matrices, load_task
from .readouts import training_weights


def fit_head(matrix,frame,indices,regularization):
    head = make_pipeline(StandardScaler(),LogisticRegression(C=regularization,max_iter=1500,random_state=20261020))
    with threadpool_limits(limits=2):
        head.fit(matrix[indices],frame.label.to_numpy()[indices],logisticregression__sample_weight=training_weights(frame.iloc[indices]))
    return head


def component_ap(frame,scores):
    selected = frame[['uniprot','component','label']].assign(prediction=scores)
    rows = [dict(component=group.component.iloc[0],ap=average_precision_score(group.label,group.prediction))
            for _,group in selected.groupby('uniprot',sort=False) if group.label.nunique()==2]
    return float(pd.DataFrame(rows).groupby('component').ap.mean().mean())


def fit(root,output,task='both'):
    tasks = ['AlloBench','AM'] if task=='both' else [task]
    hidden = np.load(root/'datasets/allosteric/hidden_centered_conditions.npy',mmap_mode='r')
    for name in tasks:
        directory = output/name
        directory.mkdir(parents=True,exist_ok=True)
        frame = load_task(root,name)
        features = feature_matrices(frame,hidden[0,frame.matrix_row])
        labels,groups = frame.label.to_numpy(int),frame.component.to_numpy()
        splits = list(GroupKFold(5).split(frame,labels,groups))
        selections,tuning,audits = [],[],[]
        frame['outer_fold'] = -1
        for fold,(training,evaluation) in enumerate(splits):
            frame.loc[evaluation,'outer_fold'] = fold
            assert not set(groups[training])&set(groups[evaluation])
            audits.append(dict(fold=fold,training_proteins=int(frame.iloc[training].uniprot.nunique()),evaluation_proteins=int(frame.iloc[evaluation].uniprot.nunique()),overlapping_components=0))
            inner = list(GroupKFold(3).split(training,labels[training],groups[training]))
            for method in METHODS:
                best = None
                for regularization in [.1,1.,10.]:
                    values = np.zeros(len(training))
                    scores = []
                    for train_rows,valid_rows in inner:
                        fit_indices,valid_indices = training[train_rows],training[valid_rows]
                        assert not set(groups[fit_indices])&set(groups[valid_indices])
                        head = fit_head(features[method],frame,fit_indices,regularization)
                        prediction = head.predict_proba(features[method][valid_indices])[:,1]
                        values[valid_rows] = prediction
                        scores.append(component_ap(frame.iloc[valid_indices],prediction))
                    value = float(np.mean(scores))
                    tuning.append(dict(task=name,fold=fold,design=method,C=regularization,inner_component_ap=value))
                    if best is None or value>best[0]:
                        best = (value,regularization,values.copy())
                candidates = np.unique(np.r_[.5,np.quantile(best[2],np.linspace(.01,.99,99))])
                threshold = float(max(candidates,key=lambda value:matthews_corrcoef(labels[training],best[2]>=value)))
                head = fit_head(features[method],frame,training,best[1])
                joblib.dump(head,directory/(str(fold)+'_'+method+'.joblib'))
                selection = dict(fold=fold,design=method,C=best[1],threshold=threshold,inner_component_ap=best[0],features=features[method].shape[1])
                selections.append(selection)
                (directory/(str(fold)+'_'+method+'.json')).write_text(json.dumps(selection,indent=2)+'\n')
                print(name,fold,method,selection,flush=True)
        frame[['uniprot','component','outer_fold']].drop_duplicates().to_csv(directory/'fold_membership.csv',index=False)
        pd.DataFrame(selections).to_csv(directory/'model_selections.csv',index=False)
        pd.DataFrame(tuning).to_csv(directory/'tuning.csv',index=False)
        pd.DataFrame(audits).to_csv(directory/'split_audit.csv',index=False)
        for method in METHODS:
            regularization = Counter(row['C'] for row in selections if row['design']==method).most_common(1)[0][0]
            joblib.dump(fit_head(features[method],frame,np.arange(len(frame)),regularization),directory/('all_'+method+'.joblib'))
