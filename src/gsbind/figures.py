import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    table = table[table.region.eq('low_noncontact') & table.target.eq('functional_magnitude') & table.method.eq('full_probe') & table.condition.isin(['baseline', 'S_retain50', 'S_retain00'])]
    table.pivot(index='dataset', columns='condition', values='rho').plot.bar(figsize=(6, 4), ylabel='Functional-effect Spearman rho', rot=0)
    plt.tight_layout()
    plt.savefig(directory / 'functional_surface_intervention.pdf')
    plt.close()
    for path in (root / 'results').rglob('*.csv'):
        destination = directory / 'source_data' / path.relative_to(root / 'results')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    print('Figure source tables and evidence plots saved to', directory)
