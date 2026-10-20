"""Paired block-bootstrap uncertainty for all-position functional applications."""

import numpy as np
import pandas as pd

from .functional import KEYS, region_select, weighted_rank, bootstrap


def weighted_top_effect(prediction, target, weights, fraction=None, count=None):
    order = np.argsort(-prediction, kind='stable')
    ordered_weights = weights[:, order]
    before = np.cumsum(ordered_weights, axis=1) - ordered_weights
    total = weights.sum(axis=1)
    budget = np.ceil(fraction * total) if fraction is not None else np.minimum(count, total)
    selected = np.minimum(np.maximum(budget[:, None] - before, 0), ordered_weights)
    return (selected * target[order]).sum(axis=1) / budget


def budget_intervals(output, iterations=10000):
    frame = pd.read_parquet(output / 'experimental_predictions.parquet')
    frame = frame[frame.version.eq('fresh_E11') & frame.condition.eq('baseline')]
    methods = ['full_probe', 'basic_probe', 'raw_score', 'reverse_score', 'burial', 'packing',
               'current_geometry', 'current_hidden', 'current_full', 'joint_hidden_logit']
    records = []
    for dataset, protein in frame.groupby('dataset'):
        blocks = np.unique((protein.position.to_numpy() - 1) // 10)
        generator = np.random.default_rng(20261020 + (dataset == 'SRC'))
        draws = generator.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=iterations)
        for region in ['all', 'noncontact']:
            assay_values = {}
            for identity, group in region_select(protein, region).groupby(KEYS):
                group = group.sort_values('position')
                mapping = np.searchsorted(blocks, (group.position.to_numpy() - 1) // 10)
                weights = draws[:, mapping]
                for method in methods:
                    for budget, options in [('top10', dict(count=10)), ('top20', dict(fraction=.2))]:
                        value = weighted_top_effect(group[method].to_numpy(), group.functional_magnitude.to_numpy(), weights, **options)
                        assay_values.setdefault((identity[2], method, budget), []).append(value)
            cases = {key: np.mean(values, axis=0) for key, values in assay_values.items()}
            parent_values = {}
            for key, values in cases.items():
                parent_values.setdefault(key[1:], []).append(values)
            parent_values = {key: np.mean(values, axis=0) for key, values in parent_values.items()}
            points = pd.read_csv(output / 'parent_metrics.csv')
            points = points[points.version.eq('fresh_E11') & points.condition.eq('baseline') & points.region.eq(region)
                            & points.dataset.eq(dataset) & points.target.eq('functional_magnitude')].set_index('method')
            for (method, budget), values in parent_values.items():
                lower, upper = np.quantile(values, [.025, .975])
                records.append(dict(dataset=dataset, region=region, budget=budget, method=method, comparison='effect_mean',
                    estimate=points.loc[method, budget + '_effect_mean'], lower=lower, upper=upper, bootstrap=iterations, sequence_block_length=10))
            comparisons = [('full_probe', reference) for reference in ['basic_probe', 'raw_score', 'reverse_score', 'burial', 'packing']]
            comparisons += [('current_hidden', 'current_geometry'), ('current_full', 'current_hidden'), ('joint_hidden_logit', 'full_probe')]
            for method, reference in comparisons:
                for budget in ['top10', 'top20']:
                    delta = parent_values[(method, budget)] - parent_values[(reference, budget)]
                    lower, upper = np.quantile(delta, [.025, .975])
                    records.append(dict(dataset=dataset, region=region, budget=budget, method=method, comparison=method + '_minus_' + reference,
                        estimate=points.loc[method, budget + '_effect_mean'] - points.loc[reference, budget + '_effect_mean'],
                        lower=lower, upper=upper, bootstrap=iterations, sequence_block_length=10))
            print('budget bootstrap', dataset, region, 'complete', flush=True)
    pd.DataFrame(records).to_csv(output / 'budget_block_bootstrap.csv', index=False)


def partial_correlation(group, method, controls, weights, include_amino):
    total = weights.sum(axis=1)
    numeric = np.stack([weighted_rank(group[column].to_numpy(), weights) / total[:, None]
                        for column in controls], axis=-1)
    design = np.concatenate([np.ones((*weights.shape, 1)), numeric], axis=-1)
    if include_amino:
        amino = np.column_stack([group.amino_acid.eq(letter) for letter in 'ARNDCQEGHILKMFPSTWYV'[:-1]]).astype(float)
        design = np.concatenate([design, np.broadcast_to(amino, (len(weights), *amino.shape))], axis=-1)
    response = np.stack([weighted_rank(group[column].to_numpy(), weights) / total[:, None]
                         for column in [method, 'functional_magnitude']], axis=-1)
    weighted_design = design * np.sqrt(weights[:, :, None])
    weighted_response = response * np.sqrt(weights[:, :, None])
    gram = np.einsum('bni,bnj->bij', weighted_design, weighted_design)
    right = np.einsum('bni,bnj->bij', weighted_design, weighted_response)
    coefficient = np.linalg.pinv(gram, rcond=1e-10, hermitian=True) @ right
    residual = response - design @ coefficient
    cross = np.einsum('bni,bnj,bn->bij', residual, residual, weights)
    return cross[:, 0, 1] / np.sqrt(cross[:, 0, 0] * cross[:, 1, 1])


def partial_intervals(output, iterations=10000):
    frame = pd.read_parquet(output / 'experimental_predictions.parquet')
    frame = frame[frame.version.eq('fresh_E11') & frame.condition.eq('baseline')]
    point = pd.read_csv(output / 'partial_rank_parent.csv')
    records = []
    for dataset, protein in frame.groupby('dataset'):
        blocks = np.unique((protein.position.to_numpy() - 1) // 10)
        generator = np.random.default_rng(20261020 + (dataset == 'SRC'))
        draws = generator.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=iterations)
        for region in ['all', 'noncontact']:
            for label, controls in [('score_geometry_AA_plddt', ['score', 'relative_sasa', 'contact_density_8A', 'dssp_helix', 'dssp_strand', 'dssp_turn', 'residue_plddt']),
                                    ('score_geometry_AA_plddt_folding', ['score', 'relative_sasa', 'contact_density_8A', 'dssp_helix', 'dssp_strand', 'dssp_turn', 'residue_plddt', 'stability_magnitude'])]:
                assay_values = {}
                for identity, group in region_select(protein, region).groupby(KEYS):
                    group = group.sort_values('position')
                    mapping = np.searchsorted(blocks, (group.position.to_numpy() - 1) // 10)
                    for method in ['full_probe', 'current_hidden']:
                        chunks = [partial_correlation(group, method, controls, draws[start:start + 128, mapping], True)
                                  for start in range(0, iterations, 128)]
                        assay_values.setdefault((identity[2], method), []).append(np.concatenate(chunks))
                cases = {key: np.mean(values, axis=0) for key, values in assay_values.items()}
                for method in ['full_probe', 'current_hidden']:
                    values = np.mean([value for key, value in cases.items() if key[1] == method], axis=0)
                    lower, upper = np.quantile(values, [.025, .975])
                    match = point[point.dataset.eq(dataset) & point.region.eq(region) & point.method.eq(method)
                                  & point.target.eq('functional_magnitude') & point.controls.eq(label)]
                    records.append(dict(dataset=dataset, region=region, method=method, controls=label,
                        estimate=float(match.partial_rank_r.iloc[0]), lower=lower, upper=upper,
                        bootstrap=iterations, block_length=10, refit_controls_each_draw=True))
                print('partial bootstrap', dataset, region, label, 'complete', flush=True)
    pd.DataFrame(records).to_csv(output / 'partial_rank_bootstrap.csv', index=False)
    frame = pd.DataFrame(records)
    frame[frame.region.eq('all') & frame.method.eq('full_probe')].to_csv(output / 'PAPER_functional_partial.csv', index=False)


def run(output, iterations=10000):
    frame = pd.read_parquet(output / 'experimental_predictions.parquet')
    parents = pd.read_csv(output / 'parent_metrics.csv')
    bootstrap(frame, parents, output, iterations, 10)
    budget_intervals(output, iterations)
    partial_intervals(output, iterations)
