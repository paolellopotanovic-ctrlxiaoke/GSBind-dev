"""Redraw the four approved panels from packaged CSV files only."""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from publication_panels import STYLE, PANELS, draw_panel


CODE = Path(__file__).resolve().parents[1]
DATA = CODE.parent / 'source_data/fig4_downstream_v20261023'


def main():
    tables = {name: pd.read_csv(DATA / 'shared' / (name+'.csv'))
              for name in ['component_AUPR', 'src_construct_rho', 'random_baselines']}
    metrics = pd.read_csv(DATA / 'shared/METRICS.csv')
    with plt.rc_context(STYLE):
        for count, name in [(3, 'GSBind_figure4_ABC'), (4, 'GSBind_figure4_ABCD')]:
            figure, axes = plt.subplots(1, count, figsize=(3.6*count, 3.8))
            figure.subplots_adjust(left=.065 if count == 3 else .05, right=.99,
                                   bottom=.20, top=.79, wspace=.42)
            for index, (axis, panel) in enumerate(zip(axes, PANELS[:count])):
                draw_panel(axis, panel, tables, metrics)
                axis.text(-.20, 1.16, chr(97+index), transform=axis.transAxes,
                          fontsize=14, fontweight='bold', va='top')
            for suffix in ['png', 'pdf']:
                figure.savefig(CODE / 'fig' / (name+'.'+suffix), dpi=350)
                shutil.copy2(CODE / 'fig' / (name+'.'+suffix), DATA / (name+'.'+suffix))
            plt.close(figure)
        for index, panel in enumerate(PANELS):
            figure, axis = plt.subplots(figsize=(3.6, 3.8))
            figure.subplots_adjust(left=.20, right=.97, bottom=.20, top=.79)
            draw_panel(axis, panel, tables, metrics)
            for suffix in ['png', 'pdf']:
                target = CODE / 'fig' / (f'fig4{chr(97+index)}_{panel}.'+suffix)
                figure.savefig(target, dpi=350)
                shutil.copy2(target, DATA / (chr(65+index)+'_'+panel) / ('reference.'+suffix))
            plt.close(figure)


if __name__ == '__main__':
    main()
