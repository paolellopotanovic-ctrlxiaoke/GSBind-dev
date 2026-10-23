"""Compose PyMOL-rendered case panels with vector titles and rank-scale legends."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases'


def image(axis, name):
    artwork = Image.open(DIRECTORY / (name + '.png')).convert('RGBA')
    alpha = artwork.getchannel('A')
    bounds = alpha.getbbox()
    if bounds:
        artwork = artwork.crop(bounds)
    axis.imshow(artwork)
    axis.axis('off')


def compose(names, titles, name, footnote, scale=True):
    count = len(names)
    figure, axes = plt.subplots(1, count, figsize=(3.3*count, 3.8))
    figure.subplots_adjust(left=.01, right=.99, top=.88, bottom=.29, wspace=.03)
    for axis, artwork, title, letter in zip(axes, names, titles, 'abcdefgh'):
        image(axis, artwork)
        axis.set_title(title, fontsize=10, pad=8)
        axis.text(0, 1.12, letter, transform=axis.transAxes, fontweight='bold', fontsize=12)
    if scale:
        color_axis = figure.add_axes([.37, .22, .27, .025])
        color_map = LinearSegmentedColormap.from_list('pymol_rank', ['white', 'yellow', 'red'])
        bar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=color_map),
                             cax=color_axis, orientation='horizontal', ticks=[0, .5, 1])
        bar.ax.tick_params(labelsize=7, length=2)
        bar.set_label('Within-protein / construct percentile', fontsize=8, labelpad=1)
    figure.text(.5, .025, footnote, ha='center', va='bottom', fontsize=7, color='#485760')
    for extension in ['pdf', 'png']:
        figure.savefig(DIRECTORY / (name + '.' + extension), dpi=300, facecolor='white')
    plt.close(figure)


if __name__ == '__main__':
    with plt.rc_context({'pdf.fonttype': 42}):
        compose(['abl1_annotation', 'abl1_binding_score', 'abl1_hidden'],
                ['ABL1: annotated allosteric pocket', 'Binding-score probe', 'Hidden-feature probe'],
                'abl1_case_comparison',
                'Display structure: PDB 5MO4 (engineered mutant). Purple: asciminib; green: nilotinib.\n'
                'Left: source-annotated pocket in coral. Middle / right: saved monomer-readout percentiles; same camera.')
        compose(['src_full_context', 'src_activation_segment'],
                ['Src: full-length domain context', 'Kinase domain: activation segment'],
                'src_activation_context',
                'AF2 display structures, human P12931 numbering. Activation segment: DFG407–APE435; Y419 highlighted.\n'
                'Domain context is not an experimental active-state assignment. Blue: SH3; green: SH2; coral: activation segment.',
                scale=False)
        from assemble_src_functional_case import assemble as assemble_src
        assemble_src()
