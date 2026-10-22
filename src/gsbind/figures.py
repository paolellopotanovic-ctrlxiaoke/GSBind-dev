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
        axis.set_ylabel('Case-macro AP (legacy aupr_mixed field)')
        axis.tick_params(axis='x', rotation=35)
    figure.savefig(directory / 'main_comparison.pdf')
    plt.close(figure)
    from .downstream import build_figure
    build_figure(root)
