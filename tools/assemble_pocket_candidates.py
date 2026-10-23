"""Compact top-five-percent pocket comparisons without interface coloring."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

import assemble_structural_readout_cases as assembler


def assemble():
    root = assembler.DIRECTORY / 'pocket_screen'
    summary = pd.read_csv(root / 'CANDIDATE_SPATIAL_SUMMARY.csv')
    for pdb in ['2XPU', '1ZA2', '1H9G']:
        assembler.DIRECTORY = root / pdb
        figure, axes = plt.subplots(1, 4, figsize=(12, 4.0), squeeze=False)
        figure.subplots_adjust(left=.045,right=.99,top=.82,bottom=.30,wspace=.03)
        for row_index,fraction in enumerate([.05]):
            for column,(method,title) in enumerate([
                    (None,'Source pocket annotation'),('gsbind_score_probe','Binding-score probe'),
                    ('structure_only','Structure-only probe'),('gsbind_hidden_probe','Hidden-feature probe')]):
                axis=axes[row_index,column]
                assembler.image(axis,'source_annotation' if method is None else f'top{int(fraction*100)}_{method}')
                if row_index==0:
                    axis.set_title(title,fontsize=10,pad=8)
                if method is not None:
                    record=summary.loc[summary.pdb.eq(pdb)&summary.fraction.eq(fraction)&summary.method.eq(method)].iloc[0]
                    axis.text(.5,-.01,f'{int(record.hits)}/{int(record.top_k)} selected annotated\n'
                              f'Precision {record.precision:.1%}; recall {record.recall:.1%}',
                              transform=axis.transAxes,ha='center',va='top',fontsize=8)
                else:
                    record=summary.loc[summary.pdb.eq(pdb)&summary.fraction.eq(fraction)].iloc[0]
                    axis.text(-.08,.5,f'Top {int(fraction*100)}%\nk={int(record.top_k)}',rotation=90,
                              transform=axis.transAxes,ha='center',va='center',fontsize=9)
        record=summary.loc[summary.pdb.eq(pdb)].iloc[0]
        figure.suptitle(f'{record.gene} / {pdb} · {int(record.annotation_N)} source-pocket residues',fontsize=12,y=.975)
        figure.legend(handles=[Patch(facecolor='#c73b38',label='Annotated / selected + annotated'),
                               Patch(facecolor='#efa33b',label='Annotated, not selected'),
                               Patch(facecolor='#3885b3',label='Selected, off annotation'),
                               Patch(facecolor='#ccd4d9',label='Other / sequence mismatch')],
                      loc='lower center',bbox_to_anchor=(.5,.095),ncol=2,frameon=False,fontsize=8)
        figure.text(.5,.02,'Full-protein fixed top-k; source-pocket labels, not annotation union. Purple: real bound effector.\n'
                    f'Only target chain {record.chain} displayed; all selected mapped residues retained. '
                    'Curated post-hoc cases, not independent validation.',ha='center',fontsize=7.5,color='#485760')
        for extension in ['png','pdf']:
            figure.savefig(root/(pdb+'_pocket_comparison.'+extension),dpi=300,facecolor='white')
        plt.close(figure)


if __name__=='__main__':
    with plt.rc_context({'pdf.fonttype':42}):
        assemble()
