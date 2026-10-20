"""Final paper endpoints: binding, allostery, score-only SSM, and KRAS/Src."""

from pathlib import Path

import numpy as np
import pandas as pd

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


def functional(root, output):
    from .functional import run as run_functional
    return run_functional(root, output)


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
