"""Export system distributions and full reference-interface recall-cost curves."""

import numpy as np
import pandas as pd

import build_figure3_binding_reliability as study

OUT = study.OUT


def recall_cost_curve(labels, scores, targets):
    order = np.argsort(-scores, kind='stable')
    cumulative = np.cumsum(labels[order])
    return [np.count_nonzero(scores>=scores[order[np.flatnonzero(cumulative>=np.ceil(labels.sum()*target))[0]]])/len(scores)
        for target in targets]


def system_distribution(frame, groups, metrics, filename):
    result = frame.groupby(groups+['system_id', 'model'], observed=True)[metrics].mean().reset_index()
    result.to_csv(OUT/filename, index=False)
    return result


def run():
    metrics = pd.read_csv(OUT/'pair_model_metrics.csv')
    system_distribution(metrics, ['input_state'], ['state_ap', 'holo_ap', 'ap_change', 'packing_conditional_auc'], 'A_H_system_distributions.csv')
    stability = pd.read_csv(OUT/'stability_case_metrics.csv')
    system_distribution(stability, ['input_state', 'requested_k'], ['jaccard', 'retained_correct_per_budget'], 'C_system_distributions.csv')
    budget = pd.read_csv(OUT/'fixed_budget_case_metrics.csv')
    systems = system_distribution(budget, ['input_state', 'requested_k'], ['precision', 'hits', 'recall'], 'D_system_distributions.csv')
    grid = systems.groupby(['input_state', 'requested_k', 'model']).precision.agg(
        mean='mean', q25=lambda values:values.quantile(.25), median='median',
        q75=lambda values:values.quantile(.75), systems='size').reset_index()
    grid.to_csv(OUT/'D_budget_distribution_grid.csv', index=False)
    confidence = pd.read_csv(OUT/'G_confidence_candidates.csv')
    bins = confidence[confidence.model.eq(study.MODELS[0])][['case_id', 'confidence_bin']]
    common = confidence.drop(columns='confidence_bin').merge(bins, on='case_id', validate='many_to_one')
    system_distribution(common, ['confidence_bin'], ['precision'], 'G_system_distributions.csv')
    packing = pd.read_csv(OUT/'packing_stratum_case_metrics.csv')
    system_distribution(packing, ['input_state', 'stratum'], ['auc'], 'H_stratum_system_distributions.csv')
    targets = np.arange(1, 10)/10
    residues = pd.read_parquet(OUT/'paired_residue_predictions.parquet')
    metadata = metrics[['case_id', 'system_id', 'input_state']].drop_duplicates().set_index('case_id')
    rows = []
    for case, group in residues.groupby('case_id', sort=False):
        labels = group.reference_label.to_numpy()
        for model in study.MODELS:
            costs = recall_cost_curve(labels, group[model+'_state_score'].to_numpy(), targets)
            for target, cost in zip(targets, costs):
                rows.append(dict(case_id=case, **metadata.loc[case].to_dict(), model=model, target_recall=float(target), inspected_fraction=cost))
    costs = pd.DataFrame(rows)
    costs.to_csv(OUT/'E_full_recall_cost_case_metrics.csv', index=False)
    study.summarize_table(costs, ['input_state', 'target_recall'], ['inspected_fraction'], 'E_full_recall_cost', 10000)
    for target, column in [(.5, 'recall50_cost'), (.8, 'recall80_cost')]:
        comparison = costs[costs.target_recall.eq(target)].merge(metrics[['case_id', 'model', column]], on=['case_id', 'model'], validate='one_to_one')
        np.testing.assert_allclose(comparison.inspected_fraction, comparison[column], rtol=0, atol=1e-15)
    print('Exported distributions and', len(costs), 'case/model/recall cost records; existing 50%/80% endpoints exactly reproduced.')


if __name__=='__main__':
    run()
