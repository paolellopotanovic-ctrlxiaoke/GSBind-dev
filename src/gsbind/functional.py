"""Frozen all-position functional analysis; no target-label model fitting."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr


METHODS = ['full_probe', 'basic_probe', 'score_probe', 'raw_score', 'reverse_score',
           'burial', 'packing', 'joint_hidden_logit', 'joint_basic_logit',
           'current_geometry', 'current_amino', 'current_hidden', 'current_full']
REGIONS = ['all', 'noncontact', 'not_noncontact', 'low_all', 'high_all',
           'low_noncontact', 'high_noncontact']
KEYS = ['dataset', 'context', 'case_id', 'interaction']
TARGETS = ['functional_magnitude', 'stability_magnitude']


def safe_rho(left, right):
    if len(left) < 8 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan
    return float(spearmanr(left, right).statistic)


def region_select(frame, region):
    mask = np.ones(len(frame), dtype=bool)
    if region.startswith('low_'):
        mask &= frame.score.lt(.5)
    if region.startswith('high_'):
        mask &= frame.score.ge(.5)
    if region == 'not_noncontact':
        mask &= frame.noncontact_eligible.eq(False)
    elif region.endswith('noncontact'):
        mask &= frame.noncontact_eligible.eq(True)
    return frame.loc[mask].copy()


def current_fold_policy(package, output):
    hits = pd.read_csv(package / 'datasets/functional/source_homology_hits.tsv', sep='\t',
                       names=['query', 'target', 'identity', 'qcov', 'tcov', 'evalue', 'bits'])
    membership = pd.read_csv(package / 'datasets/allosteric/homology/membership.csv')
    folds = pd.read_csv(package / 'checkpoints/allosteric/AlloBench/fold_membership.csv')
    hits = hits.merge(membership, left_on='target', right_on='uniprot').merge(
        folds[['component', 'outer_fold']].drop_duplicates(), on='component')
    hits['both80'] = hits.identity.ge(.3) & hits.qcov.ge(.8) & hits.tcov.ge(.8)
    hits['either80'] = hits.identity.ge(.3) & (hits.qcov.ge(.8) | hits.tcov.ge(.8))
    hits.to_csv(output / 'current_source_homology_audit.csv', index=False)
    selected = {}
    for case, parent in [('kras4b_af2_full', 'kras4b_af2_full'), ('src_fl_af2', 'src_fl_af2'), ('src_kd_af2', 'src_fl_af2')]:
        heldout = hits[hits['query'].eq(parent) & hits.both80].outer_fold.unique()
        if len(heldout) != 1:
            raise ValueError(f'No unique parent full-length source-heldout fold: {case}')
        selected[case] = int(heldout[0])
    (output / 'current_source_fold_policy.json').write_text(json.dumps(dict(
        folds=selected, rule='parent full-length both80 source component held out',
        domain_caveat='Src KD has either80 hits across all source folds; not globally domain-homology-disjoint'), indent=2) + '\n')
    return selected


def predict(package, features, hidden, current_policy):
    from gsbind.allostery import feature_matrices

    manifest = json.loads((package / 'checkpoints/functional_probe/manifest.json').read_text())
    weights = json.loads((package / 'checkpoints/functional_probe/weights.json').read_text())['folds']
    policy = pd.read_csv(package / 'datasets/functional/fold_policy.csv').set_index('case_id')
    result = features.copy()
    contract = manifest['input_contract']
    for case, group in features.groupby('case_id', sort=False):
        reference = group.index[group.score.lt(.5)]
        if not len(reference):
            reference = group.index
        center = pd.DataFrame(hidden[reference]).mean().to_numpy(dtype=np.float32)
        centered = hidden[group.index] - center
        basic = np.column_stack([group[contract['basic_scalar_order']].to_numpy(float),
            np.column_stack([group.amino_acid.eq(letter) for letter in contract['amino_acid_order']])])
        matrix = np.column_stack([basic, centered])
        heldout = policy.loc[case, 'source_heldout_folds']
        selected = weights if pd.isna(heldout) else [fold for fold in weights if fold['fold'] == int(heldout)]
        for method, name, values in [('full_probe', 'basic_center128', matrix),
                                    ('basic_probe', 'basic_linear', basic),
                                    ('score_probe', 'score_linear', group[['score']].to_numpy())]:
            result.loc[group.index, method] = np.mean([
                expit(values @ np.asarray(fold[name]['raw_coefficients']) + fold[name]['raw_intercept'])
                for fold in selected], axis=0)
        result.loc[group.index, 'joint_hidden_logit'] = np.mean([
            centered @ np.asarray(fold['basic_center128']['raw_coefficients'])[27:] for fold in selected], axis=0)
        result.loc[group.index, 'joint_basic_logit'] = np.mean([
            basic @ np.asarray(fold['basic_center128']['raw_coefficients'])[:27] + fold['basic_center128']['raw_intercept']
            for fold in selected], axis=0)
        current = feature_matrices(group, centered)
        for method in ['geometry', 'amino', 'hidden', 'full']:
            head = joblib.load(package / 'checkpoints/allosteric/AlloBench' / f'{current_policy[case]}_{method}.joblib')
            result.loc[group.index, 'current_' + method] = head.predict_proba(current[method])[:, 1]
    result['raw_score'] = features.score
    result['reverse_score'] = -features.score
    result['burial'] = 1 - features.relative_sasa
    result['packing'] = features.contact_density_8A
    return result


def build_predictions(package, output):
    features = pd.read_parquet(package / 'datasets/functional/complete_features.parquet').reset_index(drop=True)
    policy = current_fold_policy(package, output)
    conditions = ['baseline', 'identity', 'S_retain50', 'S_retain00', 'S_noise101', 'S_noise211', 'S_noise307']
    fresh = {}
    native_features = features.copy()
    for case, group in features.groupby('case_id'):
        digest = hashlib.sha256(case.encode()).hexdigest()[:20] + '.npz'
        with np.load(package / 'datasets/functional/surface' / digest) as values:
            if values['conditions'].tolist() != conditions:
                raise ValueError(f'Unexpected functional condition order: {case}')
            fresh[case] = {key: values[key].copy() for key in ['fused', 'logits']}
        np.testing.assert_array_equal(fresh[case]['fused'][0], fresh[case]['fused'][1])
        if fresh[case]['fused'].shape != (len(conditions), len(group), 128):
            raise ValueError(f'Functional hidden alignment mismatch: {case}')
        native_features.loc[group.index, 'score'] = expit(fresh[case]['logits'][0, :, 0])
    blocks = []
    for index, condition in enumerate(conditions):
        hidden = np.empty((len(features), 128), dtype=np.float32)
        for case, group in features.groupby('case_id'):
            hidden[group.index] = fresh[case]['fused'][index]
        predicted = predict(package, native_features, hidden, policy)
        predicted['version'] = 'fresh_E11'
        predicted['condition'] = condition
        blocks.append(predicted)
    predicted = pd.concat(blocks, ignore_index=True)
    predicted.to_parquet(output / 'complete_predictions.parquet', index=False)
    expected = pd.read_parquet(package / 'datasets/functional/readout_predictions.parquet')
    merged = predicted.merge(expected, on=['case_id', 'residue_index_0', 'condition'],
        suffixes=('', '_saved'), validate='one_to_one')
    if len(merged) != len(predicted) or len(expected) != len(predicted):
        raise ValueError('Functional reference position set mismatch')
    maximum = max(float(abs(merged[method] - merged[method + '_saved']).max()) for method in METHODS)
    if maximum > 1e-6:
        raise ValueError(f'Frozen all-position functional readout mismatch: {maximum}')
    (output / 'readout_verification.json').write_text(json.dumps(dict(status='passed',
        max_absolute_error=maximum, parents=2, model_inputs=3, scope='all_measured_positions',
        forward_backend='fresh_E11_gpu'), indent=2) + '\n')
    labels = pd.read_csv(package / 'datasets/functional/experimental_residues.csv')
    feature_columns = set(features.columns) | set(METHODS)
    labels = labels.drop(columns=[column for column in feature_columns if column in labels
        and column not in ['case_id', 'residue_index_0']])
    if labels.duplicated(KEYS + ['position']).any():
        raise ValueError('Duplicate functional experimental positions')
    joined = labels.merge(predicted, on=['case_id', 'residue_index_0'], validate='many_to_many')
    if len(joined) != len(labels) * len(conditions):
        raise ValueError('Missing functional labels in condition expansion')
    joined['context'] = joined.context.fillna('')
    joined.to_parquet(output / 'experimental_predictions.parquet', index=False)
    return joined


def evaluate(joined, output):
    records = []
    candidates = []
    coverage = []
    comparisons = []
    for (version, condition), frame in joined.groupby(['version', 'condition'], sort=False):
        methods = METHODS if condition == 'baseline' else ['full_probe', 'joint_hidden_logit', 'current_hidden', 'current_full']
        for region in REGIONS:
            domain = region_select(frame, region)
            for identity, group in domain.groupby(KEYS, dropna=False):
                keys = dict(zip(KEYS, identity), version=version, condition=condition, region=region)
                coverage.append(dict(**keys, positions=len(group), unique_positions=group.position.nunique(),
                                     missing_function=int(group.functional_magnitude.isna().sum()), missing_folding=int(group.stability_magnitude.isna().sum())))
                for target in TARGETS:
                    valid = group.dropna(subset=[target] + methods)
                    if len(valid) < 8 or valid[target].nunique() < 2:
                        continue
                    for method in methods:
                        ordered = valid.sort_values([method, 'position'], ascending=[False, True])
                        record = dict(**keys, target=target, method=method, positions=len(valid),
                            rho=safe_rho(valid[method], valid[target]), all_effect_mean=float(valid[target].mean()))
                        for name, budget in [('top10', min(10, len(valid))), ('top20', int(np.ceil(.2 * len(valid))))]:
                            top = ordered.head(budget)
                            truth = valid.sort_values([target, 'position'], ascending=[False, True]).head(budget)
                            selected_positions, truth_positions = set(top.position), set(truth.position)
                            record[name + '_budget'] = budget
                            record[name + '_effect_mean'] = float(top[target].mean())
                            record[name + '_fold_enrichment'] = float(top[target].mean() / valid[target].mean())
                            record[name + '_target_top_hits'] = len(selected_positions & truth_positions)
                            record[name + '_target_top_precision'] = len(selected_positions & truth_positions) / budget
                            if version == 'fresh_E11' and condition == 'baseline' and target == 'functional_magnitude' and region in ['all', 'noncontact']:
                                for rank, row in enumerate(top.itertuples(), 1):
                                    candidates.append(dict(**keys, target=target, method=method, budget_kind=name,
                                        rank=rank, position=row.position, prediction=getattr(row, method),
                                        functional_magnitude=row.functional_magnitude, stability_magnitude=row.stability_magnitude,
                                        score=row.score, amino_acid=row.amino_acid, noncontact_eligible=row.noncontact_eligible))
                        records.append(record)
                    if version == 'fresh_E11' and condition == 'baseline' and target == 'functional_magnitude':
                        budget = min(10, len(valid))
                        truth = set(valid.sort_values([target, 'position'], ascending=[False, True]).head(budget).position)
                        full = set(valid.sort_values(['full_probe', 'position'], ascending=[False, True]).head(budget).position)
                        for reference in ['raw_score', 'reverse_score', 'basic_probe', 'burial', 'packing']:
                            baseline = set(valid.sort_values([reference, 'position'], ascending=[False, True]).head(budget).position)
                            added, lost = (full - baseline) & truth, (baseline - full) & truth
                            comparisons.append(dict(**keys, baseline=reference, budget=budget, full_hits=len(full & truth),
                                baseline_hits=len(baseline & truth), added=len(added), lost=len(lost),
                                added_positions=';'.join(map(str, sorted(added))), lost_positions=';'.join(map(str, sorted(lost)))))
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / 'assay_metrics.csv', index=False)
    metric_columns = [column for column in metrics if column.startswith('top') or column in ['rho', 'all_effect_mean']]
    case_keys = ['version', 'condition', 'region', 'target', 'method', 'dataset', 'context', 'case_id']
    cases = metrics.groupby(case_keys, dropna=False)[metric_columns].mean().reset_index()
    cases.to_csv(output / 'case_metrics.csv', index=False)
    parent_keys = [column for column in case_keys if column not in ['context', 'case_id']]
    parents = cases.groupby(parent_keys, dropna=False)[metric_columns].mean().reset_index()
    parents.to_csv(output / 'parent_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    pd.DataFrame(candidates).to_csv(output / 'candidate_positions.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'top10_added_lost.csv', index=False)
    baseline = parents[parents.version.eq('fresh_E11') & parents.condition.eq('baseline')]
    baseline.to_csv(output / 'MAIN_ALLPOSITION_RESULTS.csv', index=False)
    return parents


def partial_rank_associations(joined, output):
    records = []
    frame = joined[joined.version.eq('fresh_E11') & joined.condition.eq('baseline')]
    amino_order = 'ARNDCQEGHILKMFPSTWYV'
    for region in ['all', 'noncontact', 'low_noncontact']:
        for identity, group in region_select(frame, region).groupby(KEYS):
            for target in TARGETS:
                group = group.dropna(subset=[target, 'full_probe', 'current_hidden'])
                for controls, columns in [('score', ['score']), ('score_geometry', ['score', 'relative_sasa', 'contact_density_8A', 'dssp_helix', 'dssp_strand', 'dssp_turn']),
                    ('score_geometry_AA_plddt', ['score', 'relative_sasa', 'contact_density_8A', 'dssp_helix', 'dssp_strand', 'dssp_turn', 'residue_plddt']),
                    ('score_geometry_AA_plddt_folding', ['score', 'relative_sasa', 'contact_density_8A', 'dssp_helix', 'dssp_strand', 'dssp_turn', 'residue_plddt', 'stability_magnitude'])]:
                    if target == 'stability_magnitude' and controls.endswith('_folding'):
                        continue
                    matrix = np.column_stack([np.ones(len(group)), rankdata(group[columns].to_numpy(), axis=0)])
                    if 'AA' in controls:
                        matrix = np.column_stack([matrix, np.column_stack([group.amino_acid.eq(amino) for amino in amino_order[:-1]])])
                    for method in ['full_probe', 'joint_hidden_logit', 'current_hidden', 'current_full']:
                        ranks = rankdata(group[[method, target]].to_numpy(), axis=0)
                        residual = ranks - matrix @ np.linalg.lstsq(matrix, ranks, rcond=None)[0]
                        value = float(np.corrcoef(residual.T)[0, 1]) if len(group) > np.linalg.matrix_rank(matrix) + 3 else np.nan
                        records.append(dict(zip(KEYS, identity), region=region, target=target, method=method, controls=controls,
                            positions=len(group), control_rank=np.linalg.matrix_rank(matrix), partial_rank_r=value))
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'partial_rank_assay.csv', index=False)
    keys = ['dataset', 'case_id', 'context', 'region', 'target', 'method', 'controls']
    cases = frame.groupby(keys, dropna=False).partial_rank_r.mean().reset_index()
    cases.groupby([column for column in keys if column not in ['case_id', 'context']], dropna=False).partial_rank_r.mean().reset_index().to_csv(output / 'partial_rank_parent.csv', index=False)


def weighted_rank(values, weights):
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    cumulative = np.cumsum(weights[:, order], axis=1)
    before = np.column_stack([np.zeros(len(weights)), cumulative[:, ends[:-1]]])
    group_weight = cumulative[:, ends] - before
    ranks = before + (group_weight + 1) / 2
    mapped = np.repeat(ranks, np.diff(np.r_[starts, len(values)]), axis=1)
    result = np.empty_like(mapped)
    result[:, order] = mapped
    return result


def weighted_spearman(left, right, weights):
    left_ranks = weighted_rank(np.asarray(left), weights)
    right_ranks = weighted_rank(np.asarray(right), weights)
    total = weights.sum(axis=1)
    left_ranks -= (left_ranks * weights).sum(axis=1)[:, None] / total[:, None]
    right_ranks -= (right_ranks * weights).sum(axis=1)[:, None] / total[:, None]
    denominator = np.sqrt((weights * left_ranks ** 2).sum(axis=1) * (weights * right_ranks ** 2).sum(axis=1))
    return np.divide((weights * left_ranks * right_ranks).sum(axis=1), denominator,
                     out=np.full(len(weights), np.nan), where=denominator > 0)


def bootstrap(joined, parents, output, iterations, block_length):
    frame = joined[joined.version.eq('fresh_E11')]
    names = ['full_probe', 'basic_probe', 'raw_score', 'reverse_score', 'burial', 'packing',
             'current_geometry', 'current_hidden', 'current_full']
    records = []
    for dataset, protein in frame.groupby('dataset', sort=True):
        base = protein[protein.condition.eq('baseline')]
        blocks = np.sort(((base.position.unique() - 1) // block_length))
        blocks = np.unique(blocks)
        generator = np.random.default_rng(20261020 + (dataset == 'SRC'))
        draws = generator.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=iterations)
        for region in ['all', 'noncontact']:
            distributions = {}
            for identity, group in region_select(base, region).groupby(KEYS):
                group = group.sort_values('position')
                mapping = np.searchsorted(blocks, (group.position.to_numpy() - 1) // block_length)
                assay = {}
                for method in names:
                    assay[(method, 'baseline')] = group[method].to_numpy()
                for condition in ['S_retain50', 'S_retain00', 'S_noise101', 'S_noise211', 'S_noise307']:
                    changed = protein[protein.case_id.eq(identity[2]) & protein.interaction.eq(identity[3]) & protein.condition.eq(condition)].set_index('position')
                    for method in ['full_probe', 'current_full']:
                        assay[(method, condition)] = changed.loc[group.position, method].to_numpy()
                for target in TARGETS:
                    valid = np.isfinite(group[target].to_numpy())
                    target_values = group[target].to_numpy()[valid]
                    weights = draws[:, mapping[valid]]
                    for (method, condition), values in assay.items():
                        chunks = [weighted_spearman(values[valid], target_values, weights[start:start + 256])
                                  for start in range(0, iterations, 256)]
                        distributions.setdefault((identity[2], target, method, condition), []).append(np.concatenate(chunks))
            case_values = {key: np.nanmean(values, axis=0) for key, values in distributions.items()}
            aggregate = {}
            for key, values in case_values.items():
                aggregate.setdefault(key[1:], []).append(values)
            aggregate = {key: np.nanmean(values, axis=0) for key, values in aggregate.items()}
            point = parents[parents.version.eq('fresh_E11') & parents.dataset.eq(dataset) & parents.region.eq(region)].set_index(['target', 'method', 'condition']).rho
            for key, values in aggregate.items():
                lower, upper = np.nanquantile(values, [.025, .975])
                records.append(dict(dataset=dataset, region=region, target=key[0], method=key[1], condition=key[2],
                    comparison='rho', estimate=float(point[key]), lower=lower, upper=upper, iterations=iterations, sequence_block_length=block_length))
            comparisons = [('full_probe', 'baseline', method, 'baseline') for method in ['basic_probe', 'raw_score', 'reverse_score', 'burial', 'packing']]
            comparisons += [('current_hidden', 'baseline', 'current_geometry', 'baseline'), ('current_full', 'baseline', 'current_hidden', 'baseline')]
            comparisons += [(method, condition, method, 'baseline') for method in ['full_probe', 'current_full'] for condition in ['S_retain50', 'S_retain00']]
            for target in TARGETS:
                for method, condition, reference, reference_condition in comparisons:
                    left, right = (target, method, condition), (target, reference, reference_condition)
                    delta = aggregate[left] - aggregate[right]
                    lower, upper = np.nanquantile(delta, [.025, .975])
                    records.append(dict(dataset=dataset, region=region, target=target, method=method, condition=condition,
                        comparison=f'{method}:{condition}_minus_{reference}:{reference_condition}', estimate=float(point[left] - point[right]),
                        lower=lower, upper=upper, iterations=iterations, sequence_block_length=block_length))
                for method in ['full_probe', 'current_full']:
                    zero = (target, method, 'S_retain00')
                    noise = [(target, method, condition) for condition in ['S_noise101', 'S_noise211', 'S_noise307']]
                    delta = aggregate[zero] - np.mean([aggregate[key] for key in noise], axis=0)
                    lower, upper = np.nanquantile(delta, [.025, .975])
                    records.append(dict(dataset=dataset, region=region, target=target, method=method, condition='S_retain00',
                        comparison='zero_minus_mean_noise', estimate=float(point[zero] - np.mean([point[key] for key in noise])),
                        lower=lower, upper=upper, iterations=iterations, sequence_block_length=block_length))
            print('bootstrap', dataset, region, 'complete', flush=True)
    pd.DataFrame(records).to_csv(output / 'sequence_block_bootstrap.csv', index=False)



def export_positions(joined, output):
    columns = ['dataset', 'context', 'case_id', 'interaction', 'position', 'amino_acid',
        'residue_index_0', 'score', 'noncontact_eligible', 'distance_definition', 'space_distance',
        'functional_magnitude', 'stability_magnitude', 'raw_function_magnitude',
        'raw_abundance_magnitude', 'relative_sasa', 'contact_density_8A', 'residue_plddt'] + METHODS
    joined[joined.condition.eq('baseline')][columns].to_csv(output / 'native_experimental_positions.csv', index=False)
    joined[['version', 'condition'] + columns].to_csv(output / 'all_conditions_experimental_positions.csv.gz', index=False)
    joined[joined.condition.eq('baseline')].groupby(['dataset', 'case_id', 'interaction'], dropna=False).agg(
        measured_positions=('position', 'size'), low=('score', lambda values: int(values.lt(.5).sum())),
        high=('score', lambda values: int(values.ge(.5).sum())),
        noncontact=('noncontact_eligible', 'sum')).reset_index().to_csv(output / 'experimental_coverage_summary.csv', index=False)


def export_paper_tables(parents, output):
    native = parents[parents.condition.eq('baseline') & parents.region.eq('all')
        & parents.target.eq('functional_magnitude')]
    selected = ['raw_score', 'burial', 'packing', 'basic_probe', 'full_probe', 'current_hidden', 'current_full']
    native[native.method.isin(selected)].pivot(index='method', columns='dataset', values='rho').reset_index().to_csv(
        output / 'PAPER_functional_all_positions.csv', index=False)
    native[native.method.isin(['raw_score', 'burial', 'packing', 'basic_probe', 'full_probe'])][
        ['dataset', 'method', 'top10_effect_mean', 'top10_fold_enrichment', 'top20_effect_mean', 'top20_fold_enrichment']].to_csv(
        output / 'PAPER_functional_budget.csv', index=False)
    parents[parents.region.eq('all') & parents.target.eq('functional_magnitude') & parents.method.eq('full_probe')
        & parents.condition.isin(['baseline', 'S_retain50', 'S_retain00'])].pivot(
            index='dataset', columns='condition', values='rho').reset_index().to_csv(
                output / 'PAPER_functional_surface.csv', index=False)


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    joined = build_predictions(root, output)
    parents = evaluate(joined, output)
    partial_rank_associations(joined, output)
    export_positions(joined, output)
    export_paper_tables(parents, output)
    return parents
