"""Audit complete allosteric coverage and add native-score-high evaluation."""


import numpy as np
import pandas as pd

from .metrics import binary_metrics, component_difference


def run(package, output, saved_predictions=None):
    output.mkdir(parents=True, exist_ok=True)
    saved_predictions = saved_predictions or package / 'results/allosteric'
    audits, records = [], []
    for folder in sorted(saved_predictions.iterdir()):
        if not folder.is_dir() or not (folder / 'predictions.parquet').exists():
            continue
        predicted = pd.read_parquet(folder / 'predictions.parquet')
        native = predicted[predicted.condition.eq('baseline')]
        unique = native[native.method.eq('hidden')].copy()
        task = folder.name.removeprefix('AlloBench_to_')
        labels = pd.read_parquet(package / 'datasets/allosteric/tasks' / (task + '.parquet'))
        keys = ['uniprot', 'residue_index_0']
        assert not unique.duplicated(keys).any()
        merged = unique.merge(labels[keys + ['label']], on=keys, suffixes=('_prediction', '_task'), validate='one_to_one')
        assert len(merged) == len(labels) == len(unique)
        np.testing.assert_array_equal(merged.label_prediction, merged.label_task)
        audits.append(dict(experiment=folder.name, task=task, proteins=unique.uniprot.nunique(),
            complete_positions=len(unique), low_positions=int(unique.score.lt(.5).sum()), high_positions=int(unique.score.ge(.5).sum()),
            task_positions=len(labels), identical_complete_position_set=True, positives=int(unique.label.sum())))
        raw = unique.copy()
        raw['method'] = 'binding_score'
        raw['prediction'] = raw.score
        predicted = pd.concat([predicted, raw], ignore_index=True)
        for region in ['all', 'baseline_low', 'baseline_high']:
            domain = predicted if region == 'all' else predicted[predicted.score.lt(.5) if region == 'baseline_low' else predicted.score.ge(.5)]
            for (method, condition, identifier), group in domain.groupby(['method', 'condition', 'uniprot'], sort=False):
                value = binary_metrics(group.label, group.prediction)
                records.append(dict(experiment=folder.name, region=region, method=method, condition=condition,
                    uniprot=identifier, component=group.component.iloc[0], **value))
        print('allosteric coverage/high-score', folder.name, 'complete', flush=True)
    cases = pd.DataFrame(records)
    cases.to_csv(output / 'case_metrics.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'complete_coverage_audit.csv', index=False)
    keys = ['experiment', 'region', 'method', 'condition']
    components = cases.groupby(keys + ['component'])[['aupr_mixed', 'top10_hits', 'top10_precision']].mean().reset_index()
    summary = components.groupby(keys)[['aupr_mixed', 'top10_hits', 'top10_precision']].mean().reset_index()
    counts = cases.groupby(keys).agg(proteins=('uniprot', 'nunique'), mixed_proteins=('aupr_mixed', 'count'),
        positions=('positions', 'sum'), positives=('positives', 'sum')).reset_index()
    summary = summary.merge(counts, on=keys)
    summary.to_csv(output / 'summary.csv', index=False)
    summary[summary.condition.eq('baseline')].pivot(index=['experiment', 'region', 'proteins', 'mixed_proteins'],
        columns='method', values='aupr_mixed').reset_index().to_csv(output / 'MAIN_AUPR_BY_SCORE_STRATUM.csv', index=False)
    paper = summary[summary.condition.eq('baseline')].pivot(
        index=['experiment', 'region', 'proteins', 'mixed_proteins'], columns='method', values='aupr_mixed').reset_index()
    fields = ['experiment', 'geometry', 'amino', 'hidden', 'full', 'binding_score']
    paper[paper.region.eq('all')][fields].to_csv(output / 'PAPER_allosteric_all_positions.csv', index=False)
    paper[paper.region.eq('baseline_high')][fields + ['proteins', 'mixed_proteins']].to_csv(
        output / 'PAPER_allosteric_high_score.csv', index=False)
    contrasts = []
    for (experiment, region), group in cases.groupby(['experiment', 'region']):
        native = group[group.condition.eq('baseline')].pivot(index=['uniprot', 'component'], columns='method', values='aupr_mixed').reset_index()
        for method, reference in [('hidden', 'geometry'), ('full', 'geometry'), ('full', 'hidden'),
                                  ('hidden', 'binding_score'), ('full', 'binding_score')]:
            contrasts.append(dict(experiment=experiment, region=region, comparison=method + '_minus_' + reference,
                **component_difference(native, reference, method, seed=20261020)))
        surface = group[group.method.eq('full')].pivot(index=['uniprot', 'component'], columns='condition', values='aupr_mixed').reset_index()
        for condition in ['S_retain50', 'S_retain00']:
            contrasts.append(dict(experiment=experiment, region=region, comparison=condition + '_minus_baseline',
                **component_difference(surface, 'baseline', condition, seed=20261020)))
    pd.DataFrame(contrasts).to_csv(output / 'paired_aupr_contrasts.csv', index=False)
    contrasts = pd.DataFrame(contrasts)
    contrasts[contrasts.region.eq('baseline_high') & contrasts.comparison.isin(
        ['full_minus_geometry', 'hidden_minus_geometry', 'full_minus_binding_score'])].to_csv(
            output / 'PAPER_high_score_contrasts.csv', index=False)
    reference = pd.read_csv(package / 'results/allosteric/MAIN_AUPR_TABLE.csv')
    reference = reference[reference.subset.eq('all_evaluable')]
    native = summary[summary.region.eq('all') & summary.condition.eq('baseline')].pivot(index='experiment', columns='method', values='aupr_mixed')
    for row in reference.itertuples():
        for method in ['geometry', 'amino', 'hidden', 'full']:
            np.testing.assert_allclose(native.loc[row.experiment, method], getattr(row, method), atol=1e-12)
    print('Existing all-position main table exactly reproduced', flush=True)

    return summary
