"""Clean downstream result panels with the GSBind Figure 2 visual vocabulary."""

import numpy as np


STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
         'axes.labelsize': 9, 'axes.linewidth': 1.1, 'xtick.labelsize': 8,
         'ytick.labelsize': 8, 'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
         'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
         'axes.spines.top': False, 'axes.spines.right': False,
         'pdf.fonttype': 42, 'ps.fonttype': 42}
PANELS = ['AlloBench', 'AM', 'Src', 'pooled_AUPR']


def draw_panel(axis, panel, tables, metrics):
    from .downstream import METHODS
    from .downstream_figures import COLORS, LABELS, violin

    if panel in ['AlloBench', 'AM']:
        components = tables['component_AUPR']
        rows = components.loc[components.task.eq(panel) & components.scope.eq('all_positions')]
        samples = [rows.loc[rows.method.eq(method), 'AUPR'].to_numpy() for method in METHODS]
        violin(axis, samples, COLORS)
        axis.set_xticks(range(4), LABELS)
        axis.set(ylabel='Component AUPR', ylim=(0, 1), yticks=[0, .25, .5, .75, 1])
        axis.set_title('AlloBench' if panel == 'AlloBench' else 'AM: frozen transfer', pad=12)
    elif panel == 'Src':
        means = tables['src_construct_rho'].groupby('method').rho.mean().loc[METHODS].to_numpy()
        axis.bar(range(4), means, width=.66, color=COLORS, edgecolor='#222222', linewidth=.8)
        for offset, value in enumerate(means):
            axis.text(offset, value+.022, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        axis.set_xticks(range(4), LABELS)
        axis.set(ylabel='Mean Spearman ρ', ylim=(0, .8), yticks=[0, .2, .4, .6, .8])
        axis.set_title('Src: functional readout', pad=12)
    else:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        for method_index, (method, color) in enumerate(zip(METHODS, COLORS)):
            values = [metrics.loc[metrics.task.eq(task) & metrics.scope.eq('all_positions')
                                 & metrics.method.eq(method), 'pooled_AUPR'].iloc[0]
                      for task in ['AlloBench', 'AM']]
            positions = np.arange(2)+(method_index-1.5)*.19
            axis.bar(positions, values, width=.18, color=color, edgecolor='#222222', linewidth=.8)
            for task, position, value in zip(['AlloBench', 'AM'], positions, values):
                prevalence = tables['random_baselines'].set_index('task').loc[task, 'prevalence']
                label_height = value+.014
                if label_height-.005 <= prevalence <= label_height+.035:
                    label_height = prevalence+.02
                axis.text(position, label_height, f'{value:.3f}', ha='center', va='bottom',
                          fontsize=6, rotation=0)
        for group_index, task in enumerate(['AlloBench', 'AM']):
            prevalence = tables['random_baselines'].set_index('task').loc[task, 'prevalence']
            axis.plot([group_index-.4, group_index+.4], [prevalence, prevalence],
                      color='#555555', linewidth=1, linestyle=(0, (3, 2)), zorder=5)
        axis.set_xticks([0, 1], ['AlloBench', 'AM'])
        axis.set(ylabel='Pooled AUPR', ylim=(0, .82), yticks=[0, .2, .4, .6, .8])
        axis.set_title('Pooled AUPR comparison', pad=44)
        axis.legend(handles=[Patch(facecolor=color, edgecolor='#222222', label=label.replace('\n', ' '))
                             for color, label in zip(COLORS, LABELS)] +
                            [Line2D([], [], color='#555555', linewidth=1, linestyle=(0, (3, 2)),
                                    label='Random (prevalence)')],
                    frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.01),
                    fontsize=6.5, handlelength=1, columnspacing=.9, handletextpad=.4)
    axis.set_box_aspect(1)


def render_results(directory, tables, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .downstream import METHODS

    pooled = metrics.loc[metrics.task.isin(['AlloBench', 'AM']) & metrics.scope.eq('all_positions')
                         & metrics.method.isin(METHODS),
                         ['task', 'method', 'pooled_AUPR']]
    pooled.to_csv(directory / 'source_data/downstream_selected/pooled_AUPR_bars.csv', index=False)
    with plt.rc_context(STYLE):
        for count, name in [(3, 'downstream_selected_transfer'), (4, 'downstream_results_four_panel')]:
            figure, axes = plt.subplots(1, count, figsize=(3.6*count, 3.8))
            figure.subplots_adjust(left=.065 if count == 3 else .05, right=.99,
                                   bottom=.20, top=.79, wspace=.42)
            for index, (axis, panel) in enumerate(zip(axes, PANELS[:count])):
                draw_panel(axis, panel, tables, metrics)
                axis.text(-.20, 1.16, chr(97+index), transform=axis.transAxes,
                          fontsize=14, fontweight='bold', va='top')
            for suffix in ['pdf', 'png']:
                figure.savefig(directory / (name+'.'+suffix), dpi=350)
            plt.close(figure)
        for index, panel in enumerate(PANELS):
            figure, axis = plt.subplots(figsize=(3.6, 3.8))
            figure.subplots_adjust(left=.20, right=.97, bottom=.20, top=.79)
            draw_panel(axis, panel, tables, metrics)
            name = f'downstream_{chr(65+index)}_{panel}'
            for suffix in ['pdf', 'png']:
                figure.savefig(directory / (name+'.'+suffix), dpi=350)
            plt.close(figure)
