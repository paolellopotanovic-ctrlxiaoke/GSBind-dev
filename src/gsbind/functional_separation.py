"""Frozen-readout functional separation within native-score environments."""

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .functional import KEYS, weighted_spearman


METHODS = ['raw_score', 'reverse_score', 'burial', 'packing', 'basic_probe',
           'full_probe', 'current_hidden', 'current_full']
ANALYSES = ['within_score_quintiles', 'lower_observed_folding_half', 'functional_minus_folding_rank']


def groups(frame, analysis):
    if analysis == 'within_score_quintiles':
        labels = pd.qcut(frame.score, 5, labels=False, duplicates='drop')
        return [(str(label), part, part.functional_magnitude.to_numpy())
            for label, part in frame.assign(score_bin=labels).groupby('score_bin') if len(part) >= 8]
    if analysis == 'lower_observed_folding_half':
        part = frame[frame.stability_magnitude.rank(method='average', pct=True).le(.5)]
        return [('lower_half', part, part.functional_magnitude.to_numpy())] if len(part) >= 8 else []
    if analysis == 'functional_minus_folding_rank':
        target = (rankdata(frame.functional_magnitude) - rankdata(frame.stability_magnitude)) / len(frame)
        return [('all', frame, target)]
    raise ValueError('Unknown functional separation analysis: ' + analysis)


def run(root, output, iterations=10000):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / 'results/functional/native_experimental_positions.csv')
    frame['context'] = frame.context.fillna('')
    records, coverage, positions, intervals = [], [], [], []
    for dataset, protein in frame.groupby('dataset'):
        block_ids = np.unique((protein.position.to_numpy() - 1) // 10)
        generator = np.random.default_rng(20261020 + (dataset == 'SRC'))
        draws = generator.multinomial(len(block_ids), np.full(len(block_ids), 1 / len(block_ids)), size=iterations)
        for analysis in ANALYSES:
            assay_distributions = {}
            for identity, assay in protein.groupby(KEYS):
                assay = assay.sort_values('position')
                key = dict(zip(KEYS, identity), analysis=analysis)
                bin_distributions = {method: [] for method in METHODS}
                bin_points = {method: [] for method in METHODS}
                for label, part, target in groups(assay, analysis):
                    coverage.append(dict(**key, group=label, positions=len(part),
                        score_min=part.score.min(), score_max=part.score.max()))
                    for position in part.position:
                        positions.append(dict(**key, group=label, position=position))
                    mapping = np.searchsorted(block_ids, (part.position.to_numpy() - 1) // 10)
                    for method in METHODS:
                        prediction = part[method].to_numpy()
                        point = float(spearmanr(prediction, target).statistic)
                        chunks = [weighted_spearman(prediction, target, draws[start:start + 256, mapping])
                            for start in range(0, iterations, 256)]
                        bin_points[method].append(point)
                        bin_distributions[method].append(np.concatenate(chunks))
                for method in METHODS:
                    if not bin_points[method]:
                        continue
                    records.append(dict(**key, method=method, groups=len(bin_points[method]),
                        rho=float(np.nanmean(bin_points[method]))))
                    assay_distributions.setdefault((identity[2], method), []).append(
                        np.nanmean(bin_distributions[method], axis=0))
            case_distributions = {key: np.nanmean(values, axis=0) for key, values in assay_distributions.items()}
            parent_distributions = {method: np.nanmean([values for key, values in case_distributions.items()
                if key[1] == method], axis=0) for method in METHODS}
            points = pd.DataFrame(records)
            points = points[points.dataset.eq(dataset) & points.analysis.eq(analysis)]
            points = points.groupby(['case_id', 'method']).rho.mean().groupby('method').mean()
            for method, values in parent_distributions.items():
                lower, upper = np.nanquantile(values, [.025, .975])
                intervals.append(dict(dataset=dataset, analysis=analysis, method=method, comparison='rho',
                    estimate=float(points[method]), lower=lower, upper=upper, iterations=iterations,
                    valid_draws=int(np.isfinite(values).sum()), block_length=10))
            comparisons = [('full_probe', method) for method in ['raw_score', 'reverse_score', 'burial', 'packing', 'basic_probe']]
            comparisons += [('current_hidden', method) for method in ['raw_score', 'burial', 'packing']]
            for method, reference in comparisons:
                values = parent_distributions[method] - parent_distributions[reference]
                lower, upper = np.nanquantile(values, [.025, .975])
                intervals.append(dict(dataset=dataset, analysis=analysis, method=method,
                    comparison=method + '_minus_' + reference, estimate=float(points[method] - points[reference]),
                    lower=lower, upper=upper, iterations=iterations,
                    valid_draws=int(np.isfinite(values).sum()), block_length=10))
            print('functional separation', dataset, analysis, 'complete', flush=True)
    assays = pd.DataFrame(records)
    assays.to_csv(output / 'assay_metrics.csv', index=False)
    parents = assays.groupby(['dataset', 'analysis', 'case_id', 'method']).rho.mean().reset_index()
    parents = parents.groupby(['dataset', 'analysis', 'method']).rho.mean().reset_index()
    parents.to_csv(output / 'parent_metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(output / 'paired_block_bootstrap.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    pd.DataFrame(positions).to_csv(output / 'position_membership.csv', index=False)
    (output / 'PROTOCOL.json').write_text(json.dumps(dict(
        model='Gaussian-direct128 E11', target_training=False, iterations=iterations,
        score_groups='five native-score quantiles; raw-score ties retained; duplicates dropped',
        score_bin_aggregation='equal bins within assay, then assays within case, then cases within parent',
        folding_group='observed folding-effect lower half by average percentile rank; not zero folding effect',
        selectivity='(rank(functional magnitude)-rank(folding magnitude))/N; descriptive experimental index',
        analysis_status='follow-up analyses on already inspected experimental systems',
        block_length=10, fixed_observed_membership=True, multiple_comparison_adjustment=False,
        all_methods=METHODS), indent=2) + '\n')
    return parents
