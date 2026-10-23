"""Render editable vector task explanations, not measured structures or pathways."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
INK = '#243442'
BLUE = '#557d9d'
RED = '#cb584b'
PURPLE = '#8a6aa6'


def box(axis, left, bottom, width, height, text, color='#eef2f5', size=11):
    axis.add_patch(FancyBboxPatch((left, bottom), width, height,
                                boxstyle='round,pad=0.012,rounding_size=0.025',
                                facecolor=color, edgecolor='#d1d9df', linewidth=1))
    axis.text(left + width / 2, bottom + height / 2, text, ha='center', va='center',
              fontsize=size, color=INK, linespacing=1.5)


def arrow(axis, start, end, color=INK, dashed=False):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                  linewidth=1.5, color=color, linestyle='--' if dashed else '-'))


def save(figure, name):
    directory = ROOT / 'figures'
    directory.mkdir(exist_ok=True)
    for suffix in ['svg', 'pdf', 'png']:
        figure.savefig(directory / (name + '.' + suffix), dpi=200, facecolor='white')
    plt.close(figure)


def biology():
    figure, axis = plt.subplots(figsize=(13, 6))
    figure.subplots_adjust(left=.025, right=.975, top=.88, bottom=.12)
    axis.set(xlim=(-.02, 1.02), ylim=(0, 1))
    axis.axis('off')
    figure.suptitle('Different questions on the same protein', fontsize=18, color=INK, y=.96)
    axis.add_patch(Ellipse((.44, .60), .36, .45, facecolor='#e9edf0', edgecolor='#a9b8c3', linewidth=2))
    axis.text(.44, .61, 'Target protein', ha='center', fontsize=14, color=INK)
    axis.add_patch(Ellipse((.205, .65), .16, .27, facecolor='#dae6ef', edgecolor=BLUE, linewidth=2))
    axis.text(.20, .65, 'Protein\npartner', ha='center', va='center', fontsize=11, color=BLUE)
    for height in [.57, .61, .65, .69, .73]:
        axis.add_patch(Circle((.276, height), .009, facecolor=BLUE, edgecolor='white', linewidth=.5))
    axis.text(.09, .87, 'CONTACT', color=BLUE, fontweight='bold', fontsize=12)
    arrow(axis, (.17, .84), (.275, .73), BLUE)
    axis.add_patch(Circle((.505, .812), .026, facecolor=PURPLE, edgecolor='white'))
    axis.text(.64, .87, 'ALLOSTERIC PERTURBATION', color=PURPLE, fontweight='bold', fontsize=12)
    arrow(axis, (.65, .84), (.53, .815), PURPLE)
    axis.text(.44, .43, 'Functional / active site', ha='center', fontsize=10, color='#65885f')
    arrow(axis, (.505, .77), (.47, .48), PURPLE, dashed=True)
    axis.text(.55, .67, 'Regulatory\ncoupling', fontsize=10, color=PURPLE, ha='left')
    axis.scatter([.36, .50, .58], [.72, .48, .61], marker='*', s=170, color=RED,
                 edgecolor='white', linewidth=.7, zorder=6)
    arrow(axis, (.62, .55), (.77, .55), RED)
    box(axis, .78, .43, .19, .25, 'Mutate residue i\nMeasure functional change\nAggregate effect magnitude', '#fbefed', 11)
    box(axis, .015, .02, .30, .25,
        'Binding-interface task\nWhich residues contact a protein partner?\nTarget: contact / non-contact', '#edf3f8', 11)
    box(axis, .35, .02, .30, .25,
        'Allosteric-site task\nWhich residues have allosteric annotations?\nTarget: annotated site / background', '#f3eef8', 11)
    box(axis, .685, .02, .30, .25,
        'Src functional-readout task\nWhich positions are mutation-sensitive?\nTarget: continuous effect magnitude', '#fbefed', 11)
    figure.text(.5, .03,
                'Conceptual schematic only: no measured distances or inferred signaling paths. '
                'Contact, allosteric annotation and functional sensitivity are different labels; they may overlap.\n'
                'Allosteric sites may bind small-molecule effectors; not every function-sensitive position is an allosteric site.',
                ha='center', fontsize=9, color='#52606d')
    save(figure, 'downstream_task_biology')


def workflow():
    figure, axis = plt.subplots(figsize=(14, 6.6))
    figure.subplots_adjust(left=.025, right=.975, top=.87, bottom=.13)
    axis.set(xlim=(-.02, 1.02), ylim=(0, 1))
    axis.axis('off')
    figure.suptitle('What do the three result panels evaluate?', fontsize=18, color=INK, y=.96)
    box(axis, .005, .43, .18, .24, 'Input protein structure\nSame protein, different targets\nContact / annotation / function', '#edf3f8', 10.5)
    box(axis, .23, .43, .20, .24, 'Frozen per-residue features\nGSBind: hidden 128D / score\nControls: structure / PeSTo', '#f4f5f6', 11)
    arrow(axis, (.19, .55), (.225, .55))
    box(axis, .49, .70, .26, .25,
        'A  AlloBench: site classification\n383 proteins; 216,556 positions\nFit source 5-fold logistic heads\nLabels: annotated allosteric residues', '#f3eef8', 10.5)
    box(axis, .49, .37, .26, .23,
        'B  AM: frozen transfer\n127 proteins; 1,801 annotated positions\nReuse AlloBench heads; no AM fitting\nLabels: supplied mutation-site annotations', '#f3eef8', 10.5)
    box(axis, .49, .02, .26, .25,
        'C  Src: functional ranking\nFull-length + kinase-domain constructs\n269 labeled positions per construct\nFit Src-only purged 5-fold ridge heads', '#fbefed', 10.5)
    arrow(axis, (.435, .59), (.485, .82), PURPLE)
    arrow(axis, (.62, .695), (.62, .605), PURPLE)
    arrow(axis, (.435, .46), (.485, .18), RED)
    box(axis, .805, .72, .19, .20, 'Find annotated sites\nPooled PR curve\nTrapezoidal AUPR', '#f3eef8', 11)
    box(axis, .805, .39, .19, .20, 'Transfer site ranking\nPooled PR curve\nTrapezoidal AUPR', '#f3eef8', 11)
    box(axis, .805, .04, .19, .21, 'Predict experimental sensitivity\nmean |activity effect| at i\nComplete-OOF Spearman rho', '#fbefed', 10.5)
    for height in [.82, .49, .15]:
        arrow(axis, (.755, height), (.80, height))
    axis.text(.62, .31, 'Not AlloBench → Src transfer', ha='center', fontsize=10, color=RED)
    figure.text(.5, .035,
                'A/B ask whether binding-trained features identify regulatory annotations. '
                'C asks whether those features predict experimentally derived functional sensitivity.\n'
                'Source heads are test-selected (exploratory); AM has no target fitting or selection. '
                'Unannotated AM positions are not assigned negative labels. No causal mechanism is inferred.',
                ha='center', fontsize=9, color='#52606d')
    save(figure, 'downstream_task_workflow')


if __name__ == '__main__':
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'}):
        biology()
        workflow()
