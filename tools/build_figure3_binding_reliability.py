"""Sequence-audited paired binding-site reliability from saved four-model scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figure3_binding_reliability_20261021'
PROTOCOL = ROOT/'docs/FIGURE3_BINDING_RELIABILITY_PROTOCOL_CN_20261021.md'
MODELS = ['gaussian_direct128', 'pesto_official', 'gpsfun_gpsite_pdb', 'scannet_ppi_no_msa']
NAMES = dict(zip(MODELS, ['GSBind', 'PeSTo', 'GPSFun', 'ScanNet']))
KS = [5, 10, 20, 40]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_structure(row):
    path = ROOT/row['receptor_pdb_path']
    atoms = {}
    for line in path.read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if line.startswith(('ATOM  ', 'HETATM')) and line[12:16].strip()=='CA' and line[21]==row['receptor_chain_id'] and line[16] in ' A':
            number = line[22:26].strip()+line[26].strip()
            atoms.setdefault(number, (seq1(line[17:20], custom_map={'MSE':'M'}), [float(line[30:38]), float(line[38:46]), float(line[46:54])], float(line[60:66])))
    numbers = str(row['input_residue_ids']).split(';')
    values = [atoms[number] for number in numbers]
    assert len(values)==int(row['num_residues'])
    return ''.join(value[0] for value in values), np.asarray([value[1] for value in values]), np.asarray([value[2] for value in values]), numbers


def sequence_mapping(state, holo):
    aligner = PairwiseAligner(mode='global', match_score=2., mismatch_score=-1., open_gap_score=-5., extend_gap_score=-.5)
    aligner.end_insertion_score = 0.
    aligner.end_deletion_score = 0.
    alignments = aligner.align(state, holo)
    alignment = alignments[0]
    try:
        count = len(alignments)
    except OverflowError:
        count = -1
    state_indices, holo_indices = [], []
    for first, second in zip(*alignment.aligned):
        state_indices.extend(range(first[0], first[1]))
        holo_indices.extend(range(second[0], second[1]))
    state_indices, holo_indices = np.asarray(state_indices), np.asarray(holo_indices)
    equal = np.asarray([state[first]==holo[second] for first, second in zip(state_indices, holo_indices)])
    identity = float(equal.mean()) if len(equal) else 0.
    return state_indices[equal], holo_indices[equal], dict(optimal_alignments=count, aligned_pairs=len(equal),
        identity=identity, state_coverage=float(equal.sum()/len(state)), holo_coverage=float(equal.sum()/len(holo)))


def superpose(state, holo):
    state_center, holo_center = state.mean(axis=0), holo.mean(axis=0)
    left, _, right = np.linalg.svd((state-state_center).T@(holo-holo_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left@right)
    rotation = left@correction@right
    aligned = (state-state_center)@rotation+holo_center
    return aligned, np.linalg.norm(aligned-holo, axis=1)


def ordering(score):
    return np.argsort(-np.asarray(score), kind='stable')


def recall_cost(label, score, recall):
    selected = ordering(score)
    required = int(np.ceil(label.sum()*recall))
    boundary = np.flatnonzero(np.cumsum(label[selected])>=required)[0]
    return float(np.count_nonzero(score>=score[selected[boundary]])/len(score))


def stable_candidates(label, first, second, budget):
    first_top, second_top = set(ordering(first)[:budget]), set(ordering(second)[:budget])
    common = first_top&second_top
    correct = {index for index in common if label[index]==1}
    return dict(jaccard=len(common)/len(first_top|second_top), retained_correct_per_budget=len(correct)/budget,
        retained_correct_count=len(correct), holo_precision=float(label[list(first_top)].mean()),
        state_precision=float(label[list(second_top)].mean()))


def packing_auc(label, score, density):
    strata = np.digitize(density, [8, 16, 24])
    numerator, denominator, support = 0., 0, []
    for stratum in range(4):
        selected = strata==stratum
        positives = int(label[selected].sum())
        negatives = int(selected.sum()-positives)
        pairs = positives*negatives
        auc = roc_auc_score(label[selected], score[selected]) if pairs else np.nan
        if pairs:
            numerator += auc*pairs
            denominator += pairs
        support.append(dict(stratum=stratum, stratum_positions=int(selected.sum()), stratum_positives=positives, negative_positive_pairs=pairs, auc=auc))
    return numerator/denominator if denominator else np.nan, support


def candidate_regions(label, score, distances):
    membership = distances<=8.
    priority = np.asarray([np.sort(score[region])[-min(5, int(region.sum())):].mean() for region in membership])
    kept, regions = [], []
    for center in ordering(priority):
        region = membership[center]
        if any(np.count_nonzero(region & other)/min(region.sum(), other.sum())>.5 for other in regions):
            continue
        kept.append(int(center))
        regions.append(region)
        if len(kept)==5:
            break
    values = []
    for number in [1, 3, 5]:
        selected = np.any(regions[:number], axis=0)
        values.append(dict(requested_regions=number, actual_regions=min(number, len(regions)),
            inspected_fraction=float(selected.mean()), interface_recall=float(label[selected].sum()/label.sum()),
            precision=float(label[selected].mean()), centers=';'.join(map(str, kept[:number]))))
    return values


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(ROOT/'datasets/main/apo_af2_test.csv', keep_default_na=False)
    holo = pd.read_csv(ROOT/'datasets/main/holo_test.csv', keep_default_na=False).set_index(['system_id', 'receptor_side'])
    assert holo.index.is_unique
    predictions = pd.read_parquet(ROOT/'datasets/main/four_model_predictions.parquet')
    lookup = {}
    for (case, model), group in predictions.groupby(['case_id', 'model'], sort=False):
        group = group.sort_values('residue_index_0')
        assert group.residue_index_0.tolist()==list(range(len(group)))
        lookup[(case, model)] = (group.score.to_numpy(float), group.label.to_numpy(int), group.label_mask.to_numpy(int))
    return states, holo, lookup


def run_pairs():
    states, holo, lookup = prepare()
    audit, residue_tables, metrics, budgets, regions, confidence, packings, stability = [], [], [], [], [], [], [], []
    for number, row in enumerate(states.to_dict('records')):
        reference = holo.loc[(row['system_id'], row['receptor_side'])].to_dict()
        record = dict(case_id=row['apex_case_id'], holo_case_id=reference['apex_case_id'], system_id=row['system_id'],
            input_state=row['input_state'], receptor_uniprot=row['receptor_uniprot'], receptor_side=row['receptor_side'])
        try:
            state_sequence, state_xyz, plddt, state_numbers = read_structure(row)
            holo_sequence, holo_xyz, _, holo_numbers = read_structure(reference)
            first, second, alignment = sequence_mapping(state_sequence, holo_sequence)
            record.update(alignment, state_length=len(state_sequence), holo_length=len(holo_sequence))
            state_score, state_label, state_mask = lookup[(row['apex_case_id'], MODELS[0])]
            holo_score, holo_label, holo_mask = lookup[(reference['apex_case_id'], MODELS[0])]
            assert len(state_score)==len(state_sequence) and len(holo_score)==len(holo_sequence)
            valid = state_mask[first].astype(bool)&holo_mask[second].astype(bool)
            first, second = first[valid], second[valid]
            label = holo_label[second]
            reasons = []
            if alignment['optimal_alignments']!=1:
                reasons.append('nonunique_alignment')
            if alignment['identity']<.95:
                reasons.append('identity_below_095')
            if min(alignment['state_coverage'], alignment['holo_coverage'])<.8:
                reasons.append('coverage_below_080')
            if len(first)<10:
                reasons.append('fewer_than_10_common_positions')
            if len(np.unique(label))<2:
                reasons.append('reference_not_mixed')
            record.update(included=not reasons, exclusion_reason=';'.join(reasons), common_positions=len(first),
                reference_positives=int(label.sum()), native_state_label_disagreements=int(np.count_nonzero(state_label[first]!=label)),
                state_pdb_sha256=digest(ROOT/row['receptor_pdb_path']), holo_pdb_sha256=digest(ROOT/reference['receptor_pdb_path']))
            audit.append(record)
            if any(reason!='nonunique_alignment' for reason in reasons):
                continue
            aligned, displacement = superpose(state_xyz[first], holo_xyz[second])
            distances = cdist(state_xyz[first], state_xyz[first])
            density = (distances<=8.).sum(axis=1)-1
            metadata = {key:record[key] for key in ['case_id', 'holo_case_id', 'system_id', 'input_state', 'receptor_uniprot']}
            metadata.update(mapping_unique=alignment['optimal_alignments']==1, common_positions=len(first), positives=int(label.sum()), positive_fraction=float(label.mean()),
                mean_ca_shift=float(displacement.mean()), interface_ca_shift=float(displacement[label==1].mean()),
                ca_rmsd=float(np.sqrt(np.mean(displacement**2))), interface_rmsd=float(np.sqrt(np.mean(displacement[label==1]**2))))
            residues = pd.DataFrame(dict(case_id=row['apex_case_id'], holo_case_id=reference['apex_case_id'],
                system_id=row['system_id'], input_state=row['input_state'], state_index=first, holo_index=second,
                state_residue_id=np.asarray(state_numbers)[first], holo_residue_id=np.asarray(holo_numbers)[second],
                amino_acid=[state_sequence[index] for index in first], reference_label=label, native_state_label=state_label[first],
                state_x=aligned[:, 0], state_y=aligned[:, 1], state_z=aligned[:, 2],
                holo_x=holo_xyz[second, 0], holo_y=holo_xyz[second, 1], holo_z=holo_xyz[second, 2],
                ca_shift=displacement, packing_8A=density, af2_plddt=plddt[first] if row['input_state']=='af2' else np.nan))
            residues['mapping_unique'] = alignment['optimal_alignments']==1
            for model in MODELS:
                state_values, state_truth, state_valid = lookup[(row['apex_case_id'], model)]
                holo_values, holo_truth, holo_valid = lookup[(reference['apex_case_id'], model)]
                np.testing.assert_array_equal(state_truth, state_label)
                np.testing.assert_array_equal(holo_truth, holo_label)
                np.testing.assert_array_equal(state_valid, state_mask)
                np.testing.assert_array_equal(holo_valid, holo_mask)
                state_values, holo_values = state_values[first], holo_values[second]
                assert np.isfinite(state_values).all() and np.isfinite(holo_values).all()
                residues[model+'_state_score'], residues[model+'_holo_score'] = state_values, holo_values
                ap = average_precision_score(label, state_values)
                holo_ap = average_precision_score(label, holo_values)
                conditional_auc, support = packing_auc(label, state_values, density)
                metrics.append(dict(**metadata, model=model, state_ap=ap, holo_ap=holo_ap, ap_change=ap-holo_ap,
                    score_spearman=spearmanr(state_values, holo_values).statistic, packing_conditional_auc=conditional_auc,
                    recall50_cost=recall_cost(label, state_values, .5), recall80_cost=recall_cost(label, state_values, .8),
                    native_state_ap=average_precision_score(state_label[first], state_values) if len(np.unique(state_label[first]))==2 else np.nan))
                for requested in KS+[0]:
                    budget = min(requested, len(label)) if requested else max(1, int(np.ceil(.1*len(label))))
                    selection = ordering(state_values)[:budget]
                    budgets.append(dict(**metadata, model=model, requested_k=requested, actual_k=budget,
                        hits=int(label[selection].sum()), precision=float(label[selection].mean()), recall=float(label[selection].sum()/label.sum())))
                    stability.append(dict(**metadata, model=model, requested_k=requested, actual_k=budget,
                        **stable_candidates(label, holo_values, state_values, budget)))
                    if requested==0 and row['input_state']=='af2':
                        assert np.all((plddt[first]>=0)&(plddt[first]<=100))
                        confidence.append(dict(**metadata, model=model, mean_candidate_plddt=float(plddt[first][selection].mean()),
                            precision=float(label[selection].mean()), correct=int(label[selection].sum()), budget=budget))
                regions.extend(dict(**metadata, model=model, **value) for value in candidate_regions(label, state_values, distances))
                packings.extend(dict(**metadata, model=model, **value) for value in support)
            residue_tables.append(residues)
        except (KeyError, ValueError, AssertionError) as error:
            if audit and audit[-1] is record:
                raise
            record.update(included=False, exclusion_reason='input_error:'+str(error))
            audit.append(record)
        if (number+1)%100==0:
            print('processed', number+1, 'included', len(residue_tables), flush=True)
    pd.DataFrame(audit).to_csv(OUT/'pair_mapping_audit.csv', index=False)
    for name, rows in [('pair_model_metrics', metrics), ('fixed_budget_case_metrics', budgets), ('spatial_region_case_metrics', regions),
                       ('confidence_case_metrics', confidence), ('packing_stratum_case_metrics', packings), ('stability_case_metrics', stability)]:
        table = pd.DataFrame(rows)
        table.to_csv(OUT/('sensitivity_all_'+name+'.csv'), index=False)
        table[table.mapping_unique].to_csv(OUT/(name+'.csv'), index=False)
    residues = pd.concat(residue_tables, ignore_index=True)
    residues.to_parquet(OUT/'sensitivity_all_paired_residue_predictions.parquet', index=False)
    residues = residues[residues.mapping_unique].copy()
    residues.to_parquet(OUT/'paired_residue_predictions.parquet', index=False)
    residues.to_csv(OUT/'paired_residue_predictions.csv.gz', index=False, compression='gzip')
    pd.DataFrame([dict(path=str(path.relative_to(ROOT)), sha256=digest(path)) for path in
        [Path(__file__), PROTOCOL, ROOT/'datasets/main/four_model_predictions.parquet', ROOT/'datasets/main/holo_test.csv', ROOT/'datasets/main/apo_af2_test.csv']]).to_csv(
            OUT/'input_and_code_hashes.csv', index=False)


def mean_ci(frame, value, iterations=10000):
    current = frame.dropna(subset=[value])
    values = current.groupby('system_id')[value].mean().to_numpy()
    if not len(values):
        return dict(estimate=np.nan, lower=np.nan, upper=np.nan, systems=0, cases=0)
    generator = np.random.default_rng(20261021)
    samples = values[generator.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    lower, upper = np.quantile(samples, [.025, .975])
    return dict(estimate=float(values.mean()), lower=float(lower), upper=float(upper), systems=len(values), cases=len(current), case_mean=float(current[value].mean()))


def summarize_table(frame, groups, metrics, name, iterations):
    values, contrasts = [], []
    for keys, group in frame.groupby(groups, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(groups, keys))
        for metric in metrics:
            for model, entries in group.groupby('model'):
                values.append(dict(**metadata, model=model, metric=metric, **mean_ci(entries, metric, iterations)))
            first = group[group.model.eq(MODELS[0])]
            for model in MODELS[1:]:
                second = group[group.model.eq(model)]
                paired = first[['case_id', 'system_id', metric]].merge(second[['case_id', metric]], on='case_id', validate='one_to_one', suffixes=('_gs', '_baseline'))
                paired['delta'] = paired[metric+'_gs']-paired[metric+'_baseline']
                contrasts.append(dict(**metadata, baseline=model, metric=metric, **mean_ci(paired, 'delta', iterations)))
    pd.DataFrame(values).to_csv(OUT/(name+'_summary.csv'), index=False)
    pd.DataFrame(contrasts).to_csv(OUT/(name+'_paired_contrasts.csv'), index=False)


def summarize(iterations):
    metrics = pd.read_csv(OUT/'pair_model_metrics.csv')
    summarize_table(metrics, ['input_state'], ['state_ap', 'holo_ap', 'ap_change', 'packing_conditional_auc', 'recall50_cost', 'recall80_cost', 'native_state_ap'], 'A_pair_and_H_packing', iterations)
    all_metrics = pd.read_csv(OUT/'sensitivity_all_pair_model_metrics.csv')
    summarize_table(all_metrics, ['input_state'], ['state_ap', 'holo_ap', 'ap_change', 'packing_conditional_auc', 'recall50_cost', 'recall80_cost'], 'sensitivity_all_pair', iterations)
    cluster_source = ROOT.parent/'GSBind_source_data/source_data/fig3/B_latent_feature_association/case_level_spearman.csv'
    clusters = pd.read_csv(cluster_source, usecols=['apex_case_id', 'cluster_id_R']).drop_duplicates()
    assert clusters.apex_case_id.is_unique
    clustered = metrics.merge(clusters, left_on='case_id', right_on='apex_case_id', validate='many_to_one')
    assert len(clustered)==len(metrics)
    clustered['system_id'] = clustered.cluster_id_R
    summarize_table(clustered, ['input_state'], ['state_ap', 'ap_change', 'packing_conditional_auc', 'recall50_cost', 'recall80_cost'], 'sensitivity_cluster_pair', iterations)
    metrics['shift_bin'] = pd.cut(metrics.interface_ca_shift, [-np.inf, 1., 2., np.inf], right=False, labels=['<1 A', '1-2 A', '>=2 A'])
    metrics.to_csv(OUT/'B_shift_case_metrics.csv', index=False)
    summarize_table(metrics, ['input_state', 'shift_bin'], ['state_ap', 'ap_change'], 'B_shift', iterations)
    for filename, groups, measures, name in [
        ('stability_case_metrics', ['input_state', 'requested_k'], ['retained_correct_per_budget', 'jaccard', 'holo_precision', 'state_precision'], 'C_stability'),
        ('fixed_budget_case_metrics', ['input_state', 'requested_k'], ['hits', 'precision', 'recall'], 'D_budget'),
        ('spatial_region_case_metrics', ['input_state', 'requested_regions'], ['interface_recall', 'inspected_fraction', 'precision'], 'F_region')]:
        summarize_table(pd.read_csv(OUT/(filename+'.csv')), groups, measures, name, iterations)
    confidence = pd.read_csv(OUT/'confidence_case_metrics.csv')
    confidence['confidence_bin'] = pd.cut(confidence.mean_candidate_plddt, [-np.inf, 70., 90., np.inf], right=False, labels=['<70', '70-90', '>=90'])
    confidence.to_csv(OUT/'G_confidence_candidates.csv', index=False)
    values = []
    for (model, confidence_bin), group in confidence.groupby(['model', 'confidence_bin'], observed=True):
        values.append(dict(model=model, confidence_bin=confidence_bin, **mean_ci(group, 'precision', iterations)))
    pd.DataFrame(values).to_csv(OUT/'G_confidence_summary.csv', index=False)
    native_confidence = confidence[confidence.model.eq(MODELS[0])][['case_id', 'confidence_bin']]
    common_confidence = confidence.drop(columns='confidence_bin').merge(native_confidence, on='case_id', validate='many_to_one')
    summarize_table(common_confidence, ['confidence_bin'], ['precision'], 'G_common_confidence', iterations)
    eligible = metrics[metrics.model.eq(MODELS[0])&metrics.input_state.eq('af2')].copy()
    eligible['representative_distance'] = sum((eligible[column].rank(pct=True)-.5).abs() for column in ['state_ap', 'interface_ca_shift', 'positive_fraction'])
    selected = eligible.sort_values(['representative_distance', 'case_id']).iloc[0]
    metrics[metrics.case_id.eq(selected.case_id)].to_csv(OUT/'I_example_metrics.csv', index=False)
    residues = pd.read_parquet(OUT/'paired_residue_predictions.parquet')
    example = residues[residues.case_id.eq(selected.case_id)].copy()
    for model in MODELS:
        chosen = ordering(example[model+'_state_score'].to_numpy())[:min(10, len(example))]
        example[model+'_top10'] = False
        example.iloc[chosen, example.columns.get_loc(model+'_top10')] = True
    example.to_csv(OUT/'I_example_residues.csv', index=False)
    eligible[['case_id', 'representative_distance', 'state_ap', 'interface_ca_shift', 'positive_fraction']].to_csv(OUT/'I_example_selection_all_candidates.csv', index=False)
    contract = dict(bootstrap=iterations, estimand='equal systems; cases equal within each system', protocol_sha256=digest(PROTOCOL),
        models=MODELS, label='fixed_holo_reference_interface_union_on_identity_matched_common_positions', example_case=selected.case_id,
        new_training=False, changed_scores=False, score_fusion=False)
    (OUT/'CONTRACT.json').write_text(json.dumps(contract, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    if not args.summarize_only:
        run_pairs()
    summarize(args.bootstrap)


if __name__=='__main__':
    main()
