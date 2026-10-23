"""One structural map of signed rank-error reduction, without hiding regressions."""

import argparse
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parents[1] / 'figures/structural_cases'


def prepare():
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    original = pd.read_csv(DIRECTORY.parents[1] / 'results/downstream/Src/predictions.csv.gz')
    rows = original.loc[original.case_id.eq('src_kd_af2')].copy()
    actual = rows.functional_magnitude.rank(pct=True)
    rows['experimental_rank'] = actual
    for method in ['structure_only', 'gsbind_hidden_probe']:
        rows[method+'_rank'] = rows[method].rank(pct=True)
        rows[method+'_rank_error'] = (rows[method+'_rank']-actual).abs()
    rows['rank_error_reduction'] = rows.structure_only_rank_error-rows.gsbind_hidden_probe_rank_error
    np.testing.assert_allclose(rows.rank_error_reduction.mean(),
                               rows.structure_only_rank_error.mean()-rows.gsbind_hidden_probe_rank_error.mean())
    rows.to_csv(DIRECTORY / 'SRC_SPATIAL_GAIN.csv', index=False)
    epsilon = 1e-12
    metadata = dict(construct='Src kinase-domain AF2', positions=len(rows),
                    metric='Per-position reduction in absolute rank-percentile error',
                    formula='abs(rank(structure)-rank(experiment))-abs(rank(hidden)-rank(experiment))',
                    improved=int(rows.rank_error_reduction.gt(epsilon).sum()),
                    worsened=int(rows.rank_error_reduction.lt(-epsilon).sum()),
                    unchanged=int(rows.rank_error_reduction.abs().le(epsilon).sum()),
                    structure_MAE=float(rows.structure_only_rank_error.mean()),
                    hidden_MAE=float(rows.gsbind_hidden_probe_rank_error.mean()),
                    structure_rho=float(spearmanr(rows.functional_magnitude,rows.structure_only).statistic),
                    hidden_rho=float(spearmanr(rows.functional_magnitude,rows.gsbind_hidden_probe).statistic),
                    symmetric_color_limit=float(rows.rank_error_reduction.abs().max()),
                    all_positions_retained=True, zero_centered_symmetric_scale=True,
                    active_state_not_assigned=True, source_test_selected=True,
                    spatial_map_is_not_a_spearman_decomposition=True)
    (DIRECTORY / 'SRC_SPATIAL_GAIN.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


def render():
    from importlib.metadata import version
    import pymol
    import render_structural_readout_cases as renderer

    version('pymol-open-source')
    pymol.finish_launching(['pymol','-cq'])
    cmd=pymol.cmd
    renderer.cmd=cmd
    renderer.setup(DIRECTORY / 'assets/src_kd_af2.pdb')
    rows=renderer.read_table('SRC_SPATIAL_GAIN.csv')
    specification=json.loads((DIRECTORY / 'SRC_SPATIAL_GAIN.json').read_text())
    cmd.alter('protein','b=-9')
    for row in rows:
        cmd.alter('protein and chain A and resi '+row['physical_position'],
                  'b='+row['rank_error_reduction'])
    cmd.set_color('gain_blue',[.20,.40,.65])
    cmd.set_color('gain_red',[.77,.23,.21])
    limit=specification['symmetric_color_limit']
    cmd.spectrum('b','gain_blue white gain_red','protein and b > -2',minimum=-limit,maximum=limit)
    cmd.select('activation_segment','protein and resi 407-435')
    cmd.show('sticks','protein and resi 419')
    cmd.set('stick_radius',.20)
    cmd.label('protein and resi 419 and name CA','"Y419"')
    cmd.set('label_size',16)
    cmd.orient('polymer.protein')
    cmd.turn('y',25)
    cmd.zoom('polymer.protein',buffer=3)
    renderer.save('src_spatial_gain_render')
    cmd.quit()


def assemble():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    from PIL import Image

    metadata=json.loads((DIRECTORY / 'SRC_SPATIAL_GAIN.json').read_text())
    artwork=Image.open(DIRECTORY / 'src_spatial_gain_render.png').convert('RGBA')
    artwork=artwork.crop(artwork.getchannel('A').getbbox())
    with plt.rc_context({'pdf.fonttype':42}):
        figure,axis=plt.subplots(figsize=(4.5,5.3))
        figure.subplots_adjust(left=.04,right=.96,top=.83,bottom=.24)
        axis.imshow(artwork)
        axis.axis('off')
        figure.text(.5,.97,'Src: improved agreement with experiment',ha='center',fontsize=11,fontweight='bold')
        figure.text(.5,.925,
                    f'Hidden ρ = {metadata["hidden_rho"]:.3f}   vs   structure ρ = {metadata["structure_rho"]:.3f}',
                    ha='center',fontsize=10,color='#a93c36')
        reduction=(1-metadata['hidden_MAE']/metadata['structure_MAE'])*100
        figure.text(.5,.88,f'Mean absolute rank error reduced by {reduction:.1f}%',ha='center',fontsize=9)
        color_axis=figure.add_axes([.17,.185,.66,.025])
        limit=metadata['symmetric_color_limit']
        palette=LinearSegmentedColormap.from_list('signed_gain',['#3366a6','white','#c43b36'])
        bar=figure.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap=palette),
                           cax=color_axis,orientation='horizontal',ticks=[-limit,0,limit])
        bar.ax.set_xticklabels([f'{-limit:.2f}','0',f'+{limit:.2f}'])
        bar.ax.tick_params(labelsize=8,length=2)
        figure.text(.17,.125,'Structure closer',ha='left',fontsize=8,color='#3366a6')
        figure.text(.83,.125,'Hidden closer',ha='right',fontsize=8,color='#c43b36')
        figure.text(.5,.075,
                    f'{metadata["improved"]} positions improved · {metadata["worsened"]} worsened · {metadata["unchanged"]} tied',
                    ha='center',fontsize=8)
        figure.text(.5,.02,'Color: reduction in absolute rank-percentile error; all 269 positions.\n'
                    'Same saved OOF predictions; AF2 structure; source-test-selected.\n'
                    'Gray: no evaluated effect. Functional sensitivity, not activation direction.',
                    ha='center',fontsize=6.5,color='#53616b')
        for extension in ['png','pdf']:
            figure.savefig(DIRECTORY / ('src_spatial_gain.'+extension),dpi=350,facecolor='white')
        plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['prepare','render','assemble'],required=True)
    action=parser.parse_args().action
    {'prepare':prepare,'render':render,'assemble':assemble}[action]()
