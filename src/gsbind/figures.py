import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build(root):
    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    table = pd.read_csv(root / 'results/main/case_macro_metrics.csv')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for panel, axis in zip(sorted(table.panel.unique()), axes):
        current = table[table.panel.eq(panel)]
        axis.bar(current.model, current.aupr_mixed)
        axis.set_title(panel)
        axis.set_ylabel('Case-macro AUPR (mixed labels)')
        axis.tick_params(axis='x', rotation=35)
    figure.savefig(directory / 'main_comparison.pdf')
    plt.close(figure)
    table = pd.read_csv(root / 'results/allosteric/MAIN_AUPR_TABLE.csv')
    figure, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    methods = ['geometry','amino','hidden','full']
    for axis,task in zip(axes,['AlloBench','AM']):
        current = table[table.experiment.eq(task)&table.subset.eq('all_evaluable')].iloc[0]
        axis.bar(methods,[current[method] for method in methods],color=['#8195a4','#a4b2b9','#d38c61','#bb5f48'])
        axis.set_ylabel('Component-macro AUPR')
        axis.set_title(task+' grouped nested OOF')
    figure.savefig(directory / 'allosteric_increment.pdf')
    plt.close(figure)
    table = pd.read_csv(root / 'results/functional/parent_metrics.csv')
    table = table[table.region.eq('all') & table.target.eq('functional_magnitude') & table.method.eq('full_probe') & table.condition.isin(['baseline', 'S_retain50', 'S_retain00'])]
    table.pivot(index='dataset', columns='condition', values='rho').plot.bar(figsize=(6, 4), ylabel='Functional-effect Spearman rho', rot=0)
    plt.tight_layout()
    plt.savefig(directory / 'functional_surface_intervention.pdf')
    plt.close()
    strata = root / 'results/allosteric_strata'
    table = pd.read_csv(strata / 'MAIN_AUPR_BY_SCORE_STRATUM.csv')
    complete = table[table.region.eq('all')]
    contrasts = pd.read_csv(strata / 'paired_aupr_contrasts.csv')
    high = contrasts[contrasts.region.eq('baseline_high') & contrasts.comparison.eq('full_minus_geometry')]
    high = high.set_index('experiment').loc[complete.experiment]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    positions = np.arange(len(complete))
    for offset, method in enumerate(methods):
        axes[0].barh(positions + (offset - 1.5) * .19, complete[method], height=.18, label=method)
    names = complete.experiment.str.replace('AlloBench_to_', 'Transfer: ', regex=False)
    axes[0].set_yticks(positions, names)
    axes[0].set(xlabel='Component-macro AUPR', title='All evaluable positions')
    axes[0].legend()
    axes[1].errorbar(high['mean'], positions,
        xerr=np.vstack([high['mean'] - high.lower, high.upper - high['mean']]), fmt='o', capsize=4)
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[1].set_yticks(positions, names)
    axes[1].set(xlabel='Full minus geometry AUPR, 95% CI', title='Native binding score ≥ 0.5')
    figure.savefig(directory / 'allosteric_main_and_high_score.pdf')
    plt.close(figure)
    table = pd.read_csv(root / 'results/functional/parent_metrics.csv')
    native = table[table.condition.eq('baseline') & table.region.eq('all') & table.target.eq('functional_magnitude')]
    methods = ['raw_score', 'burial', 'packing', 'basic_probe', 'full_probe']
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, dataset in enumerate(['KRAS', 'SRC']):
        current = native[native.dataset.eq(dataset)].set_index('method')
        axes[0, column].barh(methods, current.loc[methods, 'rho'])
        axes[0, column].axvline(0, color='black', linewidth=.8)
        axes[0, column].set(title=dataset + ': all measured positions', xlabel='Functional-effect Spearman rho')
        surface = table[table.dataset.eq(dataset) & table.region.eq('all')
            & table.target.eq('functional_magnitude') & table.method.eq('full_probe')].set_index('condition')
        conditions = ['baseline', 'S_retain50', 'S_retain00']
        axes[1, column].plot(np.arange(3), surface.loc[conditions, 'rho'], marker='o')
        axes[1, column].set_xticks(np.arange(3), ['native', 'surface 50%', 'surface 0%'])
        axes[1, column].set(title=dataset + ': frozen full readout', ylabel='Functional-effect Spearman rho')
    figure.savefig(directory / 'allposition_ranking_and_surface.pdf')
    figure.savefig(directory / 'allposition_ranking_and_surface.png', dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, dataset in zip(axes, ['KRAS', 'SRC']):
        current = native[native.dataset.eq(dataset)].set_index('method')
        axis.barh(methods, current.loc[methods, 'top20_fold_enrichment'])
        axis.axvline(1, color='black', linestyle='--', linewidth=.8)
        axis.set(title=dataset + ': top 20% budget', xlabel='Mean experimental effect / cohort mean')
    figure.savefig(directory / 'budget_application.pdf')
    plt.close(figure)
    separation = pd.read_csv(root / 'results/functional_separation/paired_block_bootstrap.csv')
    contrasts = separation[separation.analysis.eq('within_score_quintiles')
        & separation.comparison.eq('full_probe_minus_raw_score')].set_index('dataset').loc[['KRAS', 'SRC']]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    partial = pd.read_csv(root / 'results/functional/PAPER_functional_partial.csv')
    partial = partial[partial.controls.eq('score_geometry_AA_plddt_folding')].set_index('dataset').loc[['KRAS', 'SRC']]
    for axis, frame, title, label in [
        (axes[0], contrasts, 'Functional separation within score quintiles', 'Frozen full readout minus binding-score rho'),
        (axes[1], partial, 'Control: score, structure, AA, pLDDT and folding', 'Partial functional rank correlation')]:
        values = frame['estimate']
        axis.errorbar(np.arange(2), values,
            yerr=np.vstack([values - frame.lower, frame.upper - values]), fmt='o', capsize=5)
        axis.set_xticks(np.arange(2), frame.index)
        axis.axhline(0, color='black', linewidth=.8)
        axis.set(title=title, ylabel=label)
    figure.savefig(directory / 'functional_information_separation.pdf')
    figure.savefig(directory / 'functional_information_separation.png', dpi=180)
    plt.close(figure)
    for path in (root / 'results').rglob('*.csv'):
        destination = directory / 'source_data' / path.relative_to(root / 'results')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    print('Figure source tables and evidence plots saved to', directory)
