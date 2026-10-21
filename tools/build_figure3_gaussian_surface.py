"""Assemble the Gaussian surface-pathway Figure 3 from completed experiments."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figure3_gaussian_surface_20261021'
SOURCE = ROOT.parent/'gaussian-direct128/results/official_mmseq_gaussian_cutoff_e11_20261019'
PAIRED = ROOT/'results/figure3_binding_reliability_20261021'
CONDITIONS = ['baseline', 'identity', 'P_half', 'P_zero', 'M_zero', 'MP_zero',
    'surface_half', 'surface_zero', 'P_top', 'P_random101', 'P_random211', 'P_random307']
RANDOM = ['P_random101', 'P_random211', 'P_random307']
STATES = ['holo', 'apo', 'af2']
BOOTSTRAP = 10000


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def case_readouts(labels, native, altered):
    budget = max(1, int(np.ceil(.1*len(labels))))
    native_order = np.argsort(-native, kind='stable')[:budget]
    altered_order = np.argsort(-altered, kind='stable')[:budget]
    native_true = native_order[labels[native_order]==1]
    common_true = np.intersect1d(native_true, altered_order)
    shift = (rankdata(native, method='average')-rankdata(altered, method='average'))/len(labels)
    return dict(ap=average_precision_score(labels, altered),
        top_fraction_precision=float(labels[altered_order].mean()),
        correct_retained_per_budget=len(common_true)/budget,
        correct_lost_per_budget=(len(native_true)-len(common_true))/budget,
        interface_priority_loss=float(shift[labels==1].mean()),
        background_priority_loss=float(shift[labels==0].mean()),
        native_candidate_score_drop=float((native[native_order]-altered[native_order]).mean()))


def bootstrap_summary(table, groups, measures):
    rows = []
    for keys, group in table.groupby(groups, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for measure in measures:
            values = group.groupby('system_id')[measure].mean().dropna().to_numpy()
            generator = np.random.default_rng(20261021)
            draws = values[generator.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
            lower, upper = np.quantile(draws, [.025, .975])
            rows.append(dict(zip(groups, keys))|dict(metric=measure, estimate=float(values.mean()),
                lower=float(lower), upper=float(upper), systems=len(values), cases=group.case_id.nunique()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'raw').mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    roster = pd.read_csv(SOURCE/'all1862_official_cases.csv', keep_default_na=False)
    original = pd.read_csv(SOURCE/'all22344_official_case_metrics.csv')
    assert len(roster)==1862 and roster.case_id.is_unique
    assert original.groupby('case_id').condition.nunique().eq(12).all()
    probabilities = pd.read_parquet(SOURCE/'all_official_residue_probabilities.parquet')
    original_lookup = original.set_index(['case_id', 'condition'])
    metadata = roster.set_index('case_id')
    records, errors = [], []
    for case, group in probabilities.groupby('case_id', sort=False):
        valid = group[group.label_mask]
        wide = valid.pivot(index='residue_index_0', columns='condition', values='probability').sort_index()
        labels_by_index = valid.groupby('residue_index_0').label
        assert labels_by_index.nunique().eq(1).all()
        labels = labels_by_index.first().reindex(wide.index).to_numpy(int)
        assert set(np.unique(labels))=={0, 1} and not wide.isna().any().any()
        np.testing.assert_array_equal(wide.baseline, wide.identity)
        info = metadata.loc[case]
        for condition in CONDITIONS:
            values = case_readouts(labels, wide.baseline.to_numpy(), wide[condition].to_numpy())
            errors.append(abs(values['ap']-original_lookup.loc[(case, condition), 'aupr']))
            records.append(dict(case_id=case, system_id=info.system_id, input_state=info.input_state,
                condition=condition, valid_positions=len(labels), positives=int(labels.sum()), **values))
    metrics = pd.DataFrame(records)
    assert max(errors)<1e-12
    identity = metrics[metrics.condition.eq('baseline')][['case_id', 'ap']].rename(columns={'ap':'native_ap'})
    metrics = metrics.merge(identity, on='case_id', validate='many_to_one')
    metrics['ap_loss'] = metrics.native_ap-metrics.ap
    random = metrics[metrics.condition.isin(RANDOM)].groupby(['case_id','system_id','input_state'], as_index=False).mean(numeric_only=True)
    random['condition'] = 'P_random_mean'
    expanded = pd.concat([metrics, random], ignore_index=True)
    measures = ['ap', 'ap_loss', 'top_fraction_precision', 'correct_retained_per_budget',
        'correct_lost_per_budget', 'interface_priority_loss', 'background_priority_loss', 'native_candidate_score_drop']
    summary = bootstrap_summary(expanded, ['input_state','condition'], measures)
    systems = expanded.groupby(['input_state','system_id','condition'], as_index=False)[measures].mean()
    target = expanded[expanded.condition.eq('P_top')]
    comparison = target.merge(random, on=['case_id','system_id','input_state'], suffixes=('_target','_random'), validate='one_to_one')
    for measure in measures:
        comparison[measure] = comparison[measure+'_target']-comparison[measure+'_random']
    contrasts = bootstrap_summary(comparison, ['input_state'], measures)
    metrics.to_csv(OUT/'intervention_case_metrics.csv', index=False)
    systems.to_csv(OUT/'intervention_system_metrics.csv', index=False)
    summary.to_csv(OUT/'intervention_summary.csv', index=False)
    comparison.to_csv(OUT/'target_minus_random_case_metrics.csv', index=False)
    contrasts.to_csv(OUT/'target_minus_random_summary.csv', index=False)
    roster.drop(columns=['manifest','compact_path']).to_csv(OUT/'intervention_roster.csv', index=False)
    for name in ['pair_model_metrics.csv','pair_mapping_audit.csv','stability_case_metrics.csv',
        'A_pair_and_H_packing_summary.csv','A_pair_and_H_packing_paired_contrasts.csv',
        'C_stability_summary.csv','C_stability_paired_contrasts.csv','sensitivity_all_pair_summary.csv',
        'sensitivity_cluster_pair_summary.csv']:
        shutil.copy2(PAIRED/name, OUT/name)
    shutil.copy2(SOURCE/'all_official_residue_probabilities.parquet', OUT/'raw/all_official_residue_probabilities.parquet')
    paired_residues=PAIRED/'paired_residue_predictions.parquet'
    if not paired_residues.exists():
        paired_residues=PAIRED.parent/'raw/paired_residue_predictions.parquet'
    shutil.copy2(paired_residues,OUT/'raw/paired_residue_predictions.parquet')
    for name in ['FROZEN_PROTOCOL.json','independent_statistics_verification.json','all_official_effects_and_strata.csv']:
        shutil.copy2(SOURCE/name, OUT/'raw'/name)
    for name in ['all1862_official_cases.csv','all22344_official_case_metrics.csv']:
        shutil.copy2(SOURCE/name, OUT/'raw'/name)
    choose_example(metrics, roster, probabilities)
    report = dict(intervention_cases=len(roster), intervention_conditions=12,
        replayed_ap=len(errors), maximum_ap_replay_error=max(errors),
        input_cases=roster.input_state.value_counts().to_dict(),
        input_systems=roster.groupby('input_state').system_id.nunique().to_dict(),
        random_seeds_averaged_within_case=True, identity_exact=True,
        estimand='equal systems, equal directed cases within each system', bootstrap=BOOTSTRAP,
        native_labels_for_intervention=True, fixed_holo_reference_for_paired_transfer=True,
        no_new_training=True, no_new_forward_inference=True)
    (OUT/'VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    paths=[SOURCE/'all_official_residue_probabilities.parquet', SOURCE/'all22344_official_case_metrics.csv',
        SOURCE/'FROZEN_PROTOCOL.json', PAIRED/'pair_model_metrics.csv', PAIRED/'pair_mapping_audit.csv',paired_residues, Path(__file__)]
    pd.DataFrame([dict(path=str(path.resolve()),sha256=checksum(path)) for path in paths]).to_csv(OUT/'SOURCE_HASHES.csv', index=False)
    print(json.dumps(report,indent=2), flush=True)
    print(contrasts[contrasts.metric.isin(['ap_loss','correct_lost_per_budget','interface_priority_loss'])].to_string(index=False))


def choose_example(metrics, roster, probabilities):
    pivot = metrics.pivot(index='case_id',columns='condition',values='ap')
    candidates = roster[['case_id','input_state','system_id','num_residues']].merge(pivot.reset_index(),on='case_id')
    candidates['surface_ap_loss'] = candidates.baseline-candidates.surface_zero
    candidates['target_ap_loss'] = candidates.baseline-candidates.P_top
    candidates['target_extra_loss'] = candidates[RANDOM].mean(axis=1)-candidates.P_top
    candidates['eligible'] = (candidates.input_state.eq('af2')&candidates.baseline.ge(.7)&
        candidates.surface_ap_loss.gt(0)&candidates.target_extra_loss.gt(0))
    candidates['selection_distance'] = np.nan
    eligible = candidates[candidates.eligible].copy()
    for column in ['baseline','surface_ap_loss','target_extra_loss','num_residues']:
        eligible[column+'_median_rank_distance'] = (eligible[column].rank(pct=True)-.5).abs()
    eligible['selection_distance'] = eligible.filter(like='_median_rank_distance').sum(axis=1)
    candidates.loc[eligible.index,'selection_distance'] = eligible.selection_distance
    selected = eligible.sort_values(['selection_distance','case_id']).iloc[0]
    candidates.to_csv(OUT/'F_all_example_candidates.csv',index=False)
    metrics[metrics.case_id.eq(selected.case_id)].to_csv(OUT/'F_example_metrics.csv',index=False)
    stem = hashlib.sha256(selected.case_id.encode()).hexdigest()[:20]
    packet_path = SOURCE/'cases'/stem/'actual_outputs.npz'
    injection_path = SOURCE/'cases'/stem/'injection_audit.json'
    if not packet_path.exists():
        packet_path = SOURCE/'F_example_actual_outputs.npz'
        injection_path = SOURCE/'F_injection_audit.json'
    with np.load(packet_path) as packet:
        keys=['centers','surface_positions','labels','label_mask','P_top','P_random101','P_random211','P_random307','top_indices']
        geometry = {key:packet[key] for key in keys}
        np.savez_compressed(OUT/'F_example_geometry.npz',**geometry)
        positions = pd.DataFrame(geometry['centers'],columns=['center_x','center_y','center_z'])
        positions['label'] = packet['labels']
        positions['label_mask'] = packet['label_mask']
    subset = probabilities[probabilities.case_id.eq(selected.case_id)]
    scores = subset.pivot(index='residue_index_0',columns='condition',values='probability').sort_index()
    positions = positions.join(scores)
    positions.index.name = 'residue_index_0'
    for condition in ['baseline','surface_zero','P_top']:
        positions[condition+'_rank'] = np.nan
        valid_mask = positions.label_mask.astype(bool)
        positions.loc[valid_mask,condition+'_rank'] = positions.loc[valid_mask,condition].rank(method='average',pct=True)
        positions[condition+'_top10pct'] = False
        valid = positions[positions.label_mask.astype(bool)]
        chosen = valid.sort_values(condition,ascending=False,kind='stable').head(int(np.ceil(.1*len(valid))))
        positions.loc[chosen.index,condition+'_top10pct'] = True
    positions.to_csv(OUT/'F_example_residues.csv')
    selection = dict(case_id=selected.case_id,system_id=selected.system_id,input_state='af2',
        eligible_cases=len(eligible), candidate_cases=len(candidates),
        rule='Illustrative AF2 case with native AP>=0.7, surface AP loss>0, targeted loss>random mean; minimize sum of median percentile distances in native AP, surface AP loss, target-minus-random loss and length; break ties by case_id. Post-analysis illustrative selection, not independent validation.',
        selection_distance=float(selected.selection_distance), packet_sha256=checksum(packet_path))
    (OUT/'F_EXAMPLE_SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
    shutil.copy2(injection_path,OUT/'F_injection_audit.json')
    shutil.copy2(packet_path,OUT/'raw/F_example_actual_outputs.npz')
    shutil.copy2(injection_path,OUT/'raw/F_injection_audit.json')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir',type=Path,default=SOURCE)
    parser.add_argument('--paired-dir',type=Path,default=PAIRED)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    arguments = parser.parse_args()
    SOURCE, PAIRED, OUT = arguments.source_dir, arguments.paired_dir, arguments.output_dir
    main()
