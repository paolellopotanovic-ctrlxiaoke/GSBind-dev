"""Compact conceptual panels; no experimental coordinates or simulated results."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as DrawingPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
INK, BLUE, CORAL, PURPLE = '#293641', '#527e9a', '#c65d53', '#8b72a2'


def arrow(axis, start, end, color=INK, dashed=False):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8,
                                  linewidth=.8, color=color, linestyle='--' if dashed else '-', zorder=6))


def protein(axis, center, scale=1, color='#e8ecef', edge='#adb9c1'):
    outline = np.array([[-.17, -.05], [-.24, .09], [-.14, .24], [-.04, .18],
                        [.06, .30], [.20, .18], [.13, .09], [.29, .02],
                        [.23, -.18], [.12, -.13], [.02, -.29], [-.19, -.22], [-.17, -.05]])
    codes = [DrawingPath.MOVETO] + [DrawingPath.CURVE4] * 12
    axis.add_patch(PathPatch(DrawingPath(outline * scale + center, codes), facecolor=color,
                             edgecolor=edge, linewidth=.9, zorder=2))
    for offset in [-.08, -.02, .04, .10]:
        control = np.array([[-.10, offset-.07], [-.17, offset+.11],
                            [.13, offset+.13], [.12, offset-.05]]) * scale + center
        axis.add_patch(PathPatch(DrawingPath(control, [DrawingPath.MOVETO] + [DrawingPath.CURVE4] * 3),
                                 facecolor='none', edgecolor=edge, alpha=.45, linewidth=.65, zorder=3))


def label(axis, text, position, size=7, color=INK, **kwargs):
    axis.text(*position, text, fontsize=size, color=color, **kwargs)


def render():
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'}):
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
        figure.subplots_adjust(left=.035, right=.985, top=.94, bottom=.035, wspace=.12)
        for axis in axes:
            axis.set(xlim=(0, 1), ylim=(0, 1))
            axis.axis('off')
        axis = axes[0]
        label(axis, 'a', (0, .97), size=11, fontweight='bold')
        label(axis, 'Regulatory-site identification', (.065, .97), size=8.5, fontweight='bold')
        protein(axis, np.array([.49, .56]), scale=1.05)
        protein(axis, np.array([.14, .60]), scale=.48, color='#dae6ee', edge=BLUE)
        for height in np.linspace(.53, .67, 5):
            axis.add_patch(Circle((.28, height), .013, facecolor=BLUE, edgecolor='white', linewidth=.4, zorder=5))
        axis.plot([.28, .24, .06], [.68, .80, .80], color=BLUE, linewidth=.6)
        label(axis, 'Protein interface', (.02, .82), color=BLUE)
        axis.add_patch(Ellipse((.64, .735), .095, .055, angle=-28, facecolor=CORAL,
                               edgecolor='white', linewidth=.5, zorder=5))
        axis.plot([.64, .77, .94], [.76, .84, .84], color=CORAL, linewidth=.6)
        label(axis, 'Allosteric site', (.98, .86), color=CORAL, ha='right')
        for point in [(.65, .82), (.69, .84), (.72, .805)]:
            axis.add_patch(Circle(point, .010, facecolor=PURPLE, zorder=5))
        axis.plot([.65, .69, .72], [.82, .84, .805], color=PURPLE, linewidth=1)
        axis.add_patch(Ellipse((.60, .40), .085, .05, angle=25, facecolor='#627c70', zorder=5))
        arrow(axis, (.65, .69), (.61, .44), CORAL, dashed=True)
        label(axis, 'Functional site', (.74, .35), size=6.5, ha='center', color='#627c70')
        axis.plot([.63, .71], [.39, .36], color='#627c70', linewidth=.6)
        label(axis, 'Binding-trained features', (.50, .23), ha='center', size=7.2)
        label(axis, 'AlloBench', (.23, .12), ha='center', fontweight='bold')
        label(axis, 'Source readout', (.23, .065), ha='center', size=6.2, color='#687780')
        arrow(axis, (.39, .13), (.62, .13), CORAL)
        label(axis, 'AM', (.76, .12), ha='center', fontweight='bold')
        label(axis, 'Frozen transfer', (.76, .065), ha='center', size=6.2, color='#687780')
        axis = axes[1]
        label(axis, 'b', (0, .97), size=11, fontweight='bold')
        label(axis, 'Experimental functional sensitivity', (.065, .97), size=8.5, fontweight='bold')
        for baseline, full in [(.83, True), (.70, False)]:
            label(axis, 'Src FL' if full else 'Src KD', (.015, baseline+.018), size=6.5)
            axis.plot([.18, .90], [baseline+.035, baseline+.035], color='#aab4bc', linewidth=.7)
            if full:
                for left, name in [(.23, 'SH3'), (.38, 'SH2')]:
                    axis.add_patch(Rectangle((left, baseline), .13, .07, facecolor='#e4e9ed', edgecolor='#a9b6c0', linewidth=.6))
                    label(axis, name, (left+.065, baseline+.035), size=6.2, ha='center', va='center')
            axis.add_patch(Rectangle((.55, baseline), .31, .07, facecolor='#efcfca', edgecolor=CORAL, linewidth=.6))
            label(axis, 'Kinase', (.705, baseline+.035), size=6.2, ha='center', va='center')
        protein(axis, np.array([.20, .45]), scale=.56)
        for point in [(.15, .50), (.24, .38), (.29, .48)]:
            axis.add_patch(Circle(point, .014, facecolor=CORAL, edgecolor='white', linewidth=.5, zorder=5))
        arrow(axis, (.36, .47), (.47, .47))
        values = np.array([[.08, .45, .22, .8, .1, .2], [.2, .7, .04, .4, .15, .4],
                           [.13, .2, .6, .9, .05, .3], [.05, .35, .16, .5, .2, .08]])
        for row in range(4):
            for column in range(6):
                axis.add_patch(Rectangle((.51+column*.052, .39+row*.043), .048, .039,
                                         facecolor=plt.cm.Reds(values[row, column]*.7), edgecolor='none'))
        label(axis, 'Mutational effects', (.665, .60), ha='center', size=7)
        label(axis, 'Substitutions', (.49, .47), size=5.8, rotation=90, ha='right', va='center')
        label(axis, 'Positions', (.665, .355), size=6, ha='center')
        arrow(axis, (.665, .33), (.665, .26), CORAL)
        label(axis, r'$m_i=\mathrm{mean}_{a\ne WT}|\mathrm{activity\ effect}_{i,a}|$',
              (.50, .19), ha='center', size=8)
        label(axis, 'Position-level functional ranking', (.50, .10), ha='center', size=7.2)
        label(axis, 'Schematic; not measured structures or effects', (.50, .025),
              ha='center', size=5.5, color='#7b8790')
        directory = ROOT / 'figures'
        for extension in ['svg', 'pdf', 'png']:
            figure.savefig(directory / ('downstream_tasks_publication.' + extension), dpi=400, facecolor='white')
        plt.close(figure)


if __name__ == '__main__':
    render()
