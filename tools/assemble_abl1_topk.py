"""Compact two-threshold, three-method ABL1 case preview."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from assemble_structural_readout_cases import DIRECTORY, image


def assemble():
    summary = pd.read_csv(DIRECTORY / 'ABL1_TOPK_SUMMARY.csv')
    methods = ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe']
    names = ['Binding-score probe', 'Structure-only probe', 'Hidden-feature probe']
    with plt.rc_context({'pdf.fonttype': 42}):
        figure, axes = plt.subplots(2, 3, figsize=(10.2, 7.8))
        figure.subplots_adjust(left=.055, right=.99, top=.92, bottom=.18, wspace=.04, hspace=.24)
        for row_index, fraction in enumerate([.05, .10]):
            for column, (method, name) in enumerate(zip(methods, names)):
                axis = axes[row_index, column]
                image(axis, f'abl1_top{int(fraction*100)}_{method}')
                record = summary.loc[summary.fraction.eq(fraction) & summary.method.eq(method)].iloc[0]
                if row_index == 0:
                    axis.set_title(name, fontsize=11, pad=10)
                axis.text(.5, -.02,
                          f'Mapped: {int(record.mapped_hits)} hits / {int(record.mapped_off_annotation)} off-annotation\n'
                          f'Full protein: precision {record.full_protein_precision:.1%}, recall {record.full_protein_recall:.1%}',
                          ha='center', va='top', fontsize=8, transform=axis.transAxes)
                if column == 0:
                    axis.text(-.07, .5, f'Top {int(fraction*100)}%\nk={int(record.top_k)}', transform=axis.transAxes,
                              ha='center', va='center', rotation=90, fontsize=10)
        figure.legend(handles=[Patch(facecolor='#c73b38', label='Selected + annotated'),
                               Patch(facecolor='#efa33b', label='Annotated, not selected'),
                               Patch(facecolor='#3885b3', label='Selected, off annotation'),
                               Patch(facecolor='#ccd4d9', label='Other / sequence mismatch')],
                      loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False, fontsize=9)
        figure.text(.5, .015,
                    'ABL1 / 5MO4: exact top-k selected among all 1,130 evaluated protein positions; only mapped structure shown.\n'
                    'Purple ligand: asciminib; green: nilotinib. Blue is not a newly discovered pocket or proven biological negative.\n'
                    'Same camera for every panel; engineered sequence mismatches remain gray. Curated, source-test-selected illustration.',
                    ha='center', fontsize=7.5, color='#4b5962')
        for extension in ['png', 'pdf']:
            figure.savefig(DIRECTORY / ('abl1_topk_comparison.' + extension), dpi=300, facecolor='white')
        plt.close(figure)


if __name__ == '__main__':
    assemble()
