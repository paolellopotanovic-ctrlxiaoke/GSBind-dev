"""Component-level distributions and complete-OOF Src associations."""

from pathlib import Path

import numpy as np
import pandas as pd

from .downstream import METHODS
from sklearn.metrics import auc, precision_recall_curve


LABELS = ['GSBind\nscore', 'Structure', 'GSBind\nhidden', 'PeSTo\nscore']
COLORS = ['#4FB9AF', '#FFE0AC', '#82B0D2', '#FFACB7']
SCOPES = [('AlloBench', 'all_positions'), ('AM', 'all_positions'),
          ('AM', 'source_component_disjoint')]


def source_tables(root):
    root = Path(root)
    detailed = pd.read_csv(root / 'results/downstream/DETAILED_METRICS.csv')
    selected = detailed.loc[detailed.method.isin(METHODS)]
    classification = selected.loc[selected.task.ne('Src')]
    components = classification.groupby(['task', 'scope', 'method', 'component'], as_index=False).agg(
        AUPR=('AUPR', 'mean'), proteins=('uniprot', 'nunique'))
    routing = pd.read_csv(root / 'datasets/downstream/AM/TRANSFER_ROUTING.csv')
    independence = routing.groupby('component').source_component_disjoint.first()
    components['source_component_disjoint'] = [
        bool(independence.loc[row.component]) if row.task == 'AM' else False
        for row in components.itertuples()]
    wide = components.pivot(index=['task', 'scope', 'component'], columns='method', values='AUPR')
    paired = []
    for comparator in METHODS:
        if comparator == 'gsbind_hidden_probe':
            continue
        current = wide.reset_index()[['task', 'scope', 'component']].copy()
        current['comparator'] = comparator
        current['delta_AUPR'] = (wide.gsbind_hidden_probe - wide[comparator]).to_numpy()
        paired.append(current)
    deltas = pd.concat(paired, ignore_index=True)
    paired_summary = deltas.groupby(['task', 'scope', 'comparator'], as_index=False).agg(
        components=('component', 'nunique'), mean_delta_AUPR=('delta_AUPR', 'mean'),
        median_delta_AUPR=('delta_AUPR', 'median'),
        fraction_improved=('delta_AUPR', lambda values: float(values.gt(0).mean())))
    constructs = selected.loc[selected.task.eq('Src'),
                              ['case_id', 'interaction', 'method', 'rho', 'positions']].copy()
    construct_values = constructs.pivot(index=['case_id', 'interaction'], columns='method', values='rho')
    functional_gains = []
    for comparator in METHODS:
        if comparator == 'gsbind_hidden_probe':
            continue
        values = construct_values.reset_index()[['case_id', 'interaction']].copy()
        values['comparator'] = comparator
        values['delta_rho'] = (construct_values.gsbind_hidden_probe - construct_values[comparator]).to_numpy()
        functional_gains.append(values)
    positions = pd.read_csv(root / 'results/downstream/Src/predictions.csv.gz')
    positions['functional_rank'] = positions.groupby(['case_id', 'interaction']).functional_magnitude.rank(pct=True)
    positions['hidden_prediction_rank'] = positions.groupby(['case_id', 'interaction']).gsbind_hidden_probe.rank(pct=True)
    summary = components.groupby(['task', 'scope', 'method'], as_index=False).agg(
        components=('component', 'nunique'), mean_AUPR=('AUPR', 'mean'), median_AUPR=('AUPR', 'median'))
    baselines = []
    for task in ['AlloBench', 'AM']:
        labels = pd.read_csv(root / 'results/downstream' / task / 'predictions.csv.gz', usecols=['label']).label
        baselines.append(dict(task=task, positions=len(labels), positives=int(labels.sum()),
                              prevalence=float(labels.mean())))
    return dict(component_AUPR=components, paired_component_deltas=deltas, paired_summary=paired_summary,
                src_construct_rho=constructs, src_functional_gains=pd.concat(functional_gains, ignore_index=True),
                src_position_rank_scatter=positions, panel_summary=summary,
                random_baselines=pd.DataFrame(baselines))


def violin(axis, samples, colors):
    for offset, (values, color) in enumerate(zip(samples, colors)):
        values = np.asarray(values, float)
        bodies = axis.violinplot(values, positions=[offset], widths=.76,
                                 showextrema=False)['bodies']
        for body in bodies:
            body.set_facecolor(color)
            body.set_edgecolor('#222222')
            body.set_linewidth(.8)
            body.set_alpha(.85)
        axis.boxplot(values, positions=[offset], widths=.16, showfliers=False,
                     patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.1),
                     boxprops=dict(facecolor='white', edgecolor='#222222', linewidth=.8),
                     whiskerprops=dict(color='#222222', linewidth=.7),
                     capprops=dict(color='#222222', linewidth=.7))
    axis.set_xlim(-.55, len(samples) - .45)
    axis.set_xticks(range(len(samples)))
    axis.spines[['top', 'right']].set_visible(False)


def build_figure(root, output=None):
    root = Path(root)
    directory = Path(output) if output else root / 'figures'
    source = directory / 'source_data/downstream_selected'
    source.mkdir(parents=True, exist_ok=True)
    tables = source_tables(root)
    for name, table in tables.items():
        table.to_csv(source / (name + '.csv'), index=False)
    metrics = pd.read_csv(root / 'results/downstream/METRICS.csv')
    metrics.to_csv(directory / 'downstream_selected_transfer_source.csv', index=False)
    from .downstream_publication import render_results

    render_results(directory, tables, metrics)
    return tables


def build_pr_figure(root, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    directory = Path(output) if output else root / 'figures'
    source = directory / 'source_data/downstream_pooled_pr'
    source.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(root / 'results/downstream/METRICS.csv')
    curves, summaries = [], []
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42, 'path.simplify': False}):
        figure, axes = plt.subplots(1, 2, figsize=(11.6, 5.1))
        figure.subplots_adjust(left=.065, right=.985, top=.82, bottom=.19, wspace=.24)
        for task, axis in zip(['AlloBench', 'AM'], axes):
            frame = pd.read_csv(root / 'results/downstream' / task / 'predictions.csv.gz')
            prevalence = float(frame.label.mean())
            for method, label, color in zip(METHODS, LABELS, COLORS):
                precision, recall, thresholds = precision_recall_curve(frame.label, frame[method])
                area = float(auc(recall, precision))
                expected = reference.loc[reference.task.eq(task) & reference.scope.eq('all_positions')
                                         & reference.method.eq(method), 'pooled_AUPR'].iloc[0]
                np.testing.assert_allclose(area, expected, rtol=1e-10, atol=1e-12)
                curves.append(pd.DataFrame(dict(task=task, method=method, recall=recall,
                                                precision=precision, threshold=np.r_[thresholds, np.nan])))
                summaries.append(dict(task=task, method=method, pooled_AUPR=area,
                                      positions=len(frame), positives=int(frame.label.sum()), prevalence=prevalence))
                hidden = method == 'gsbind_hidden_probe'
                axis.plot(recall, precision, color=color, linewidth=2 if hidden else 1.4,
                          alpha=1 if hidden else .9, zorder=4 if hidden else 3,
                          label=f'{label.replace(chr(10), " ")}: {area:.4f}')
                if hidden:
                    axis.fill_between(recall, precision, color=color, alpha=.09, zorder=1)
            axis.axhline(prevalence, color='#626d77', linestyle='--', linewidth=1,
                         label=f'Prevalence: {prevalence:.4f}')
            title = 'A  AlloBench: selected OOF' if task == 'AlloBench' else 'B  AM: frozen AlloBench transfer'
            axis.set(title=title, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.025))
            axis.grid(alpha=.15)
            axis.spines[['top', 'right']].set_visible(False)
            axis.legend(loc='upper right', frameon=False, fontsize=9, title='Pooled trapezoidal AUPR', title_fontsize=10)
        figure.suptitle('Pooled precision–recall curves', fontsize=15, y=.96)
        figure.text(.5, .04,
                    'All evaluated residues pooled within each dataset; shaded area: GSBind hidden trapezoidal AUPR.\n'
                    'Same checkpoints as the component plots; no pooled-metric reselection. '
                    'AlloBench is source-test-selected; AM has no target fitting or selection.',
                    ha='center', fontsize=8.5, color='#46505a')
        for suffix in ['pdf', 'png']:
            figure.savefig(directory / ('downstream_pooled_pr.' + suffix), dpi=220)
        plt.close(figure)
    curves = pd.concat(curves, ignore_index=True)
    summaries = pd.DataFrame(summaries)
    curves.to_csv(source / 'PR_CURVES.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summaries.to_csv(source / 'AUPR_SUMMARY.csv', index=False)
    return curves, summaries
