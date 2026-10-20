"""Final paper endpoints: binding, allostery, score-only SSM, and KRAS/Src."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from .metrics import binary_metrics, component_difference, group_metrics, bh_adjust


def write_table(frame, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / (name + '.csv'), index=False)


def summarize(cases, groups):
    columns = ['aupr', 'aupr_mixed', 'auroc', 'mcc', 'f1', 'precision', 'recall', 'accuracy', 'top10_hits', 'top10_precision', 'top10_recall']
    summary = cases.groupby(groups, dropna=False)[columns].mean().reset_index()
    counts = cases.groupby(groups, dropna=False).agg(cases=('case_id', 'count'), mixed_cases=('aupr_mixed', 'count'))
    return summary.merge(counts.reset_index(), on=groups)


def contrasts(cases, groups, condition='condition', baseline='baseline', component='component'):
    rows = []
    for identity, frame in cases.groupby(groups, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        keys = dict(zip(groups, identity))
        for metric in ['aupr_mixed', 'mcc', 'f1', 'top10_hits', 'top10_precision']:
            wide = frame.pivot(index=['case_id', component], columns=condition, values=metric)
            if baseline not in wide:
                continue
            for name in wide.columns:
                if name == baseline:
                    continue
                data = wide[[baseline, name]].reset_index()
                rows.append(keys | dict(metric=metric, comparison=name, direction='comparison_minus_baseline',
                                        **component_difference(data, baseline, name, component)))
            noise = ['S_noise101', 'S_noise211', 'S_noise307']
            if all(name in wide for name in noise + ['S_retain00']):
                wide['noise_mean'] = wide[noise].mean(axis=1)
                rows.append(keys | dict(metric=metric, comparison='S_zero_minus_noise', direction='surface_zero_minus_noise_mean',
                                        **component_difference(wide.reset_index(), 'noise_mean', 'S_retain00', component)))
    result = pd.DataFrame(rows)
    result['q'] = np.nan
    for _, indices in result.groupby('metric').groups.items():
        result.loc[indices, 'q'] = bh_adjust(result.loc[indices, 'p'])
    return result


def main_task(root, output):
    predictions = pd.read_parquet(root / 'datasets/main/four_model_predictions.parquet')
    predictions = predictions[predictions.label_mask.gt(.5)].sort_values(['case_id', 'residue_index_0'])
    cases = group_metrics(predictions, ['panel', 'model', 'case_id'], 'label', 'score')
    write_table(cases, output, 'case_metrics')
    write_table(summarize(cases, ['panel', 'model']), output, 'case_macro_metrics')
    write_table(group_metrics(predictions, ['panel', 'model'], 'label', 'score'), output, 'pooled_metrics')
    cases['component'] = cases.case_id.map(lambda identity: identity.split('::')[1])
    write_table(contrasts(cases, ['panel'], 'model', 'gaussian_direct128'), output, 'four_model_paired_statistics')
    surface = pd.read_parquet(root / 'datasets/main/surface_predictions.parquet')
    surface = surface[surface.label_mask.gt(.5)].sort_values(['case_id', 'residue_index_0'])
    cut = group_metrics(surface, ['panel', 'condition', 'case_id'], 'label', 'probability')
    cut['component'] = cut.case_id.map(lambda identity: identity.split('::')[1])
    write_table(cut, output, 'surface_case_metrics')
    write_table(summarize(cut, ['panel', 'condition']), output, 'surface_case_macro_metrics')
    write_table(group_metrics(surface, ['panel', 'condition'], 'label', 'probability'), output, 'surface_pooled_metrics')
    write_table(contrasts(cut, ['panel']), output, 'surface_paired_statistics')


def allosteric(root, output):
    from .allostery import run as run_allostery
    return run_allostery(root, output)


def ssm(root, output):
    frame = pd.read_parquet(root / 'datasets/ssm220/score_predictions.parquet').sort_values('residue_index_0')
    records = []
    pooled = []
    for endpoint in ['is_direct_interface', 'sensitive_ge_1', 'sensitive_ge_2', 'primary18_label', 'relaxed1_label']:
        for region in ['all', 'noninterface', 'baseline_high', 'baseline_low']:
            domain = frame[frame[endpoint].notna()]
            if region == 'noninterface':
                domain = domain[domain.is_direct_interface.eq(0)]
            elif region == 'baseline_high':
                domain = domain[domain.protein_interface_probability.ge(.5)]
            elif region == 'baseline_low':
                domain = domain[domain.protein_interface_probability.lt(.5)]
            current = group_metrics(domain, ['case_id', 'cluster_rep', 'split', 'condition'], endpoint, 'binding_score')
            records.append(current.rename(columns={'cluster_rep': 'component'}).assign(endpoint=endpoint, region=region))
            pooled.append(group_metrics(domain, ['condition'], endpoint, 'binding_score').assign(endpoint=endpoint, region=region))
    cases = pd.concat(records, ignore_index=True)
    write_table(cases, output, 'case_metrics')
    write_table(pd.concat(pooled, ignore_index=True), output, 'pooled_metrics')
    write_table(summarize(cases, ['endpoint', 'region', 'condition']), output, 'case_macro_metrics')
    write_table(contrasts(cases, ['endpoint', 'region']), output, 'surface_paired_statistics')
    coverage = cases[cases.condition.eq('baseline') & cases.region.eq('all')].groupby('endpoint').agg(cases=('case_id', 'nunique'), positions=('positions', 'sum'), positives=('positives', 'sum'), mixed_cases=('aupr_mixed', 'count')).reset_index()
    write_table(coverage, output, 'endpoint_coverage')


def functional_readout(root, features, hidden):
    weights = json.loads((root / 'checkpoints/functional_probe/weights.json').read_text())['folds']
    contract = json.loads((root / 'checkpoints/functional_probe/manifest.json').read_text())['input_contract']
    audit = pd.read_csv(root / 'datasets/functional/fold_policy.csv').set_index('case_id')
    low = features.score.lt(.5).to_numpy()
    selected = features.loc[low]
    centered = hidden[low] - pd.DataFrame(hidden[low]).groupby(selected.case_id.to_numpy()).transform('mean').to_numpy()
    basic = np.column_stack([selected[contract['basic_scalar_order']].to_numpy(float), np.column_stack([selected.amino_acid.eq(letter) for letter in contract['amino_acid_order']])])
    matrices = dict(full_probe=np.column_stack([basic, centered]), basic_probe=basic, score_probe=selected[['score']].to_numpy(float))
    heads = dict(full_probe='basic_center128', basic_probe='basic_linear', score_probe='score_linear')
    result = features[['case_id', 'residue_index_0']].copy()
    for method in matrices:
        result[method] = np.nan
    for identity, group in selected.groupby('case_id'):
        heldout = audit.loc[identity, 'source_heldout_folds']
        folds = weights if pd.isna(heldout) else [entry for entry in weights if entry['fold'] == int(heldout)]
        mask = selected.case_id.eq(identity).to_numpy()
        for method, matrix in matrices.items():
            predicted = [expit(matrix[mask] @ np.asarray(fold[heads[method]]['raw_coefficients']) + fold[heads[method]]['raw_intercept']) for fold in folds]
            result.loc[group.index, method] = np.mean(predicted, axis=0)
    return result


def functional(root, output):
    features = pd.read_parquet(root / 'datasets/functional/complete_features.parquet')
    original = np.load(root / 'datasets/functional/hidden.npy')
    conditions = ['baseline', 'identity', 'S_retain50', 'S_retain00', 'S_noise101', 'S_noise211', 'S_noise307']
    predictions = []
    for condition in conditions:
        changed = original.copy()
        for identity, group in features.groupby('case_id'):
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            with np.load(root / 'datasets/functional/surface' / (digest + '.npz')) as saved:
                names = saved['conditions'].tolist()
                delta = saved['fused'][names.index(condition)] - saved['fused'][names.index('baseline')]
            changed[group.index] = original[group.index] + delta
        predictions.append(functional_readout(root, features, changed).assign(condition=condition))
    predicted = pd.concat(predictions, ignore_index=True)
    expected = pd.read_parquet(root / 'datasets/functional/readout_predictions.parquet')
    merged = predicted.merge(expected, on=['case_id', 'residue_index_0', 'condition'], suffixes=('', '_saved'), validate='one_to_one')
    maximum = max(float(abs(merged[method]-merged[method + '_saved']).max()) for method in ['full_probe', 'basic_probe', 'score_probe'])
    if maximum > 1e-6:
        raise ValueError(f'Functional frozen readout mismatch: {maximum}')
    predicted.to_parquet(output / 'frozen_predictions.parquet', index=False)
    labels = pd.read_csv(root / 'datasets/functional/experimental_residues.csv', float_precision='round_trip')
    labels = labels.drop(columns=['full_probe', 'basic_probe', 'score_probe'])
    joined = labels.merge(predicted, on=['case_id', 'residue_index_0'], validate='many_to_many')
    records = []
    keys = ['case_id', 'dataset', 'context', 'parent_uniprot', 'interaction', 'condition']
    for region in ['low_all', 'low_noncontact']:
        domain = joined[joined.score.lt(.5)]
        if region == 'low_noncontact':
            domain = domain[domain.noncontact_eligible.eq(True)]
        for identity, group in domain.groupby(keys, dropna=False):
            for target in ['functional_magnitude', 'stability_magnitude']:
                valid = group.dropna(subset=[target, 'full_probe', 'basic_probe', 'score_probe'])
                if len(valid) < 8 or valid[target].nunique() < 2:
                    continue
                methods = ['full_probe', 'basic_probe', 'score_probe', 'raw_score', 'reverse_score', 'burial', 'packing'] if identity[-1] == 'baseline' else ['full_probe']
                for method in methods:
                    budget = int(np.ceil(.2*len(valid)))
                    top = valid.sort_values([method, 'position'], ascending=[False, True]).head(budget)
                    records.append(dict(zip(keys, identity)) | dict(region=region, target=target, method=method, positions=len(valid), budget=budget,
                                   rho=float(spearmanr(valid[method], valid[target]).statistic) if valid[method].nunique() > 1 else np.nan,
                                   top20_effect_mean=float(top[target].mean()), all_effect_mean=float(valid[target].mean())))
    table = pd.DataFrame(records)
    write_table(table, output, 'interaction_metrics')
    case_keys = ['case_id', 'dataset', 'context', 'parent_uniprot', 'region', 'target', 'condition', 'method']
    case = table.groupby(case_keys, dropna=False)[['rho', 'top20_effect_mean']].mean().reset_index()
    parents = case.groupby([name for name in case_keys if name not in ['case_id', 'context']], dropna=False)[['rho', 'top20_effect_mean']].mean().reset_index()
    write_table(case, output, 'case_metrics')
    write_table(parents, output, 'parent_metrics')
    effects = []
    for identity, group in parents[parents.method.eq('full_probe')].groupby(['dataset', 'parent_uniprot', 'region', 'target']):
        values = group.set_index('condition')
        for metric in ['rho', 'top20_effect_mean']:
            noise = float(values.loc[['S_noise101', 'S_noise211', 'S_noise307'], metric].mean())
            original_value = float(values.loc['baseline', metric])
            for condition in ['S_retain50', 'S_retain00']:
                changed_value = float(values.loc[condition, metric])
                effects.append(dict(zip(['dataset', 'parent_uniprot', 'region', 'target'], identity)) | dict(metric=metric, condition=condition,
                                baseline=original_value, perturbed=changed_value, loss=original_value-changed_value,
                                noise_mean=noise, full_cut_excess_over_noise=noise-float(values.loc['S_retain00', metric])))
    write_table(pd.DataFrame(effects), output, 'surface_effects')
    (output / 'readout_verification.json').write_text(json.dumps(dict(max_absolute_error=maximum, parents=2, model_inputs=3), indent=2))


TASKS = dict(main=main_task, allosteric=allosteric, ssm220=ssm, functional=functional)


def run(root, task='all', output_root=None, saved_allosteric=False):
    for name, function in TASKS.items():
        if task not in ['all', name]:
            continue
        output = (Path(output_root) if output_root is not None else root / 'results') / name
        output.mkdir(parents=True, exist_ok=True)
        print('Evaluating', name, flush=True)
        if name == 'allosteric' and saved_allosteric:
            from .allostery import run as run_allostery
            run_allostery(root, output, saved_predictions=root / 'results/allosteric')
        else:
            function(root, output)
        print('Saved', output, flush=True)
