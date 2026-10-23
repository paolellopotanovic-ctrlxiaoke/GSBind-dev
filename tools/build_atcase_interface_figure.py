"""Add biological-interface context without changing allosteric labels or ranks."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import assemble_structural_readout_cases as assembler


DIRECTORY = assembler.DIRECTORY / 'pocket_screen/1ZA2'
PALETTE = dict(pocket_hit='#c73b38', pocket_missed='#efa33b',
               interface_hit='#198477', interface_context='#bedbd5',
               other_high='#8664aa', overlap='#cc6ca4', background='#ccd4d9')
METHODS = ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe']


def prepare():
    rows = pd.read_csv(DIRECTORY / 'TOPK_RESIDUES.csv')
    contacts = pd.read_csv(DIRECTORY / 'ALL_RESIDUE_BIOLOGICAL_CONTACTS.csv')
    rows = rows.merge(contacts, on='pdb_resi', validate='many_to_one')
    if rows.contact_5A.isna().any():
        raise ValueError('Incomplete biological interface mapping')
    rows['display_category'] = np.select(
        [~rows.identity_match, rows.source_site & rows.contact_5A,
         rows.source_site & rows.selected, rows.source_site & ~rows.selected,
         rows.contact_5A & rows.selected, rows.contact_5A & ~rows.selected, rows.selected],
        ['background','overlap','pocket_hit','pocket_missed','interface_hit','interface_context','other_high'],
        default='background')
    rows.to_csv(DIRECTORY / 'INTERFACE_AWARE_TOPK.csv', index=False)
    metadata = dict(interface_definition='Deposited biological assembly; protein heavy-atom distance <=5 A',
                    independent_of_prediction=True, allosteric_labels_unchanged=True,
                    overlap_precedence='Any annotated-pocket/interface overlap uses distinct overlap color',
                    palette=PALETTE, source='ALL_RESIDUE_BIOLOGICAL_CONTACTS.csv',
                    interface_residues=int(contacts.contact_5A.sum()))
    (DIRECTORY / 'INTERFACE_FIGURE.json').write_text(json.dumps(metadata,indent=2)+'\n')


def assemble():
    rows=pd.read_csv(DIRECTORY / 'INTERFACE_AWARE_TOPK.csv')
    assembler.DIRECTORY=DIRECTORY
    figure,axes=plt.subplots(2,4,figsize=(12,7.1))
    figure.subplots_adjust(left=.045,right=.99,top=.89,bottom=.23,wspace=.03,hspace=.23)
    for row_index,fraction in enumerate([.05,.10]):
        for column,(method,title) in enumerate(zip([None]+METHODS,
                ['Pocket + direct interface','Binding-score probe','Structure-only probe','Hidden-feature probe'])):
            axis=axes[row_index,column]
            name='interface_reference' if method is None else f'interface_top{int(fraction*100)}_{method}'
            assembler.image(axis,name)
            if row_index==0:
                axis.set_title(title,fontsize=10,pad=8)
            current=rows.loc[rows.fraction.eq(fraction)&rows.method.eq(method or 'gsbind_hidden_probe')]
            if method is not None:
                pocket=int(current.display_category.eq('pocket_hit').sum())
                interface=int(current.display_category.eq('interface_hit').sum())
                other=int(current.display_category.eq('other_high').sum())
                axis.text(.5,-.015,f'Mapped: {pocket} pocket / {interface} interface / {other} other high',
                          transform=axis.transAxes,ha='center',va='top',fontsize=7.5)
            else:
                axis.text(-.08,.5,f'Top {int(fraction*100)}%\nk={int(current.top_k.iloc[0])}',
                          transform=axis.transAxes,rotation=90,ha='center',va='center',fontsize=9)
    labels=[('pocket_hit','Pocket: selected (reference: annotation)'),('pocket_missed','Pocket: not selected'),
            ('interface_hit','Direct interface: selected'),('interface_context','Direct interface: context'),
            ('other_high','Other selected position'),('background','Other / sequence mismatch')]
    if rows.display_category.eq('overlap').any():
        labels.append(('overlap','Pocket + interface overlap'))
    figure.legend(handles=[Patch(facecolor=PALETTE[key],label=label) for key,label in labels],
                  loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=8)
    figure.suptitle('ATCase regulatory chain B / 1ZA2',fontsize=12,y=.975)
    figure.text(.5,.015,
                'Interface: ≤5 Å protein heavy-atom contact in the deposited biological assembly; annotation independent of predictions.\n'
                'Purple sticks: CTP. Same camera; target chain B shown. Allosteric labels and fixed full-protein top-k are unchanged.\n'
                'Curated illustration; interface hits are not counted as allosteric-pocket positives.',
                ha='center',fontsize=7.5,color='#485760')
    for extension in ['png','pdf']:
        figure.savefig(DIRECTORY / ('interface_aware_comparison.'+extension),dpi=300,facecolor='white')
    plt.close(figure)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['prepare','assemble'],required=True)
    action=parser.parse_args().action
    if action=='prepare':
        prepare()
    else:
        with plt.rc_context({'pdf.fonttype':42}):
            assemble()
