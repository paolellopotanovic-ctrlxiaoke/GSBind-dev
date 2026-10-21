"""Joint binding-accuracy and reference-interface screening benefits per system."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figure3_binding_reliability_20261021'
BASELINES = ['pesto_official', 'gpsfun_gpsite_pdb', 'scannet_ppi_no_msa']
QUADRANTS = ['both_better', 'accuracy_only', 'cost_only', 'neither', 'boundary_tie']


def joint_table(metrics):
    keys = ['input_state', 'system_id']
    means = metrics.groupby(keys+['model'])[['state_ap', 'recall80_cost']].mean().reset_index()
    counts = metrics.groupby(keys).case_id.nunique().rename('directed_cases').reset_index()
    reference = means[means.model.eq('gaussian_direct128')].drop(columns='model')
    rows = []
    for baseline in BASELINES:
        comparator = means[means.model.eq(baseline)].drop(columns='model')
        paired = reference.merge(comparator, on=keys, suffixes=('_gsbind', '_baseline'), validate='one_to_one')
        if len(paired)!=len(reference) or len(comparator)!=len(reference):
            raise ValueError('Every system must contain all four methods')
        paired = paired.merge(counts, on=keys, validate='one_to_one')
        paired['baseline'] = baseline
        paired['ap_gain'] = paired.state_ap_gsbind-paired.state_ap_baseline
        paired['cost_saving'] = paired.recall80_cost_baseline-paired.recall80_cost_gsbind
        paired['quadrant'] = np.select([
            paired.ap_gain.gt(0)&paired.cost_saving.gt(0),
            paired.ap_gain.gt(0)&paired.cost_saving.lt(0),
            paired.ap_gain.lt(0)&paired.cost_saving.gt(0),
            paired.ap_gain.lt(0)&paired.cost_saving.lt(0)], QUADRANTS[:4], default='boundary_tie')
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def summarize(table, iterations=10000, seed=20261021):
    rows = []
    for state, state_table in table.groupby('input_state', sort=True):
        systems = sorted(state_table.system_id.unique())
        generator = np.random.default_rng(seed)
        draws = generator.integers(0, len(systems), size=(iterations, len(systems)))
        for baseline, group in state_table.groupby('baseline', sort=True):
            group = group.set_index('system_id').loc[systems]
            for quadrant in QUADRANTS:
                values = group.quadrant.eq(quadrant).to_numpy(float)
                lower, upper = np.quantile(values[draws].mean(axis=1), [.025, .975])
                rows.append(dict(input_state=state, baseline=baseline, quadrant=quadrant,
                    count=int(values.sum()), systems=len(systems), estimate=float(values.mean()),
                    lower=float(lower), upper=float(upper)))
    return pd.DataFrame(rows)


def run():
    metrics = pd.read_csv(OUT/'pair_model_metrics.csv')
    table = joint_table(metrics)
    summary = summarize(table)
    table.to_csv(OUT/'I_joint_system_metrics.csv', index=False)
    summary.to_csv(OUT/'I_joint_summary.csv', index=False)
    expected = {('af2', 'pesto_official'):198, ('af2', 'gpsfun_gpsite_pdb'):279,
        ('af2', 'scannet_ppi_no_msa'):297, ('apo', 'pesto_official'):22,
        ('apo', 'gpsfun_gpsite_pdb'):25, ('apo', 'scannet_ppi_no_msa'):24}
    for row in summary[summary.quadrant.eq('both_better')].itertuples():
        assert row.count==expected[(row.input_state, row.baseline)]
    totals = summary.groupby(['input_state', 'baseline'])['count'].sum()
    assert totals.loc['af2'].eq(371).all() and totals.loc['apo'].eq(38).all()
    report = dict(systems_by_state=table.groupby('input_state').system_id.nunique().to_dict(),
        system_baseline_rows=len(table), shared_system_bootstrap=10000,
        no_outcome_selected_systems=True, strict_positive_gains=True,
        joint_benefit_is_descriptive_not_independent_replication=True,
        endpoint='fixed_holo_reference_binding_interface', no_new_training=True)
    (OUT/'I_joint_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(summary[summary.quadrant.eq('both_better')].to_string(index=False))


if __name__=='__main__':
    run()
