"""Matched-view Src case with quantitative complete-OOF agreement annotations."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases'


def assemble():
    original = pd.read_csv(ROOT / 'results/downstream/Src/predictions.csv.gz')
    summaries = []
    for case, group in original.groupby('case_id'):
        for method in ['structure_only', 'gsbind_hidden_probe']:
            summaries.append(dict(case_id=case, method=method, positions=len(group),
                                  rho=float(spearmanr(group.functional_magnitude, group[method]).statistic)))
    summary = pd.DataFrame(summaries)
    reference = pd.read_csv(ROOT / 'results/downstream/DETAILED_METRICS.csv')
    for row in summary.itertuples():
        expected = reference.loc[reference.task.eq('Src') & reference.case_id.eq(row.case_id)
                                 & reference.method.eq(row.method), 'rho'].iloc[0]
        np.testing.assert_allclose(row.rho, expected, rtol=1e-10, atol=1e-12)
    summary.to_csv(DIRECTORY / 'SRC_CASE_AGREEMENT.csv', index=False)
    current = summary.loc[summary.case_id.eq('src_kd_af2')].set_index('method')
    baseline, hidden = current.loc['structure_only', 'rho'], current.loc['gsbind_hidden_probe', 'rho']
    names = ['src_activation_segment', 'src_experimental', 'src_structure', 'src_hidden']
    images = [Image.open(DIRECTORY / (name + '.png')).convert('RGBA') for name in names]
    if len({artwork.size for artwork in images}) != 1:
        raise ValueError('Matched-view panels require equal render dimensions')
    bounds = [artwork.getchannel('A').getbbox() for artwork in images]
    crop = (min(bound[0] for bound in bounds), min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds), max(bound[3] for bound in bounds))
    metadata = dict(display_construct='src_kd_af2', positions=269, no_retraining=True,
                    agreement_metric='Complete-OOF Spearman, all 269 evaluated KD positions',
                    structure_rho=float(baseline), hidden_rho=float(hidden),
                    absolute_delta_rho=float(hidden-baseline), common_crop=list(crop),
                    display_scale='Separate within-construct percentile ranks, same 0-1 color mapping',
                    metric_computed_from='Original experimental functional magnitudes and saved OOF predictions',
                    no_site_selection_or_color_remapping=True,
                    source_test_selected=True,
                    full_length_results_retained_in='SRC_CASE_AGREEMENT.csv and main complete-OOF figure')
    (DIRECTORY / 'SRC_CASE_FIGURE.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with plt.rc_context({'pdf.fonttype': 42}):
        figure, axes = plt.subplots(1, 4, figsize=(12, 4.4))
        figure.subplots_adjust(left=.02, right=.99, top=.82, bottom=.28, wspace=.04)
        titles = ['Activation-segment reference', 'Experimental sensitivity',
                  'Structure-only probe', 'GSBind hidden probe']
        for index, (axis, artwork, title) in enumerate(zip(axes, images, titles)):
            axis.imshow(artwork.crop(crop))
            axis.axis('off')
            axis.set_title(title, fontsize=10, pad=10, color='#c65d53' if index==3 else '#263744',
                           fontweight='bold' if index==3 else 'normal')
            axis.text(0, 1.12, 'abcd'[index], transform=axis.transAxes, fontsize=12, fontweight='bold')
        axes[0].text(.5, -.035, 'DFG407–APE435 · Y419', transform=axes[0].transAxes,
                     ha='center', va='top', fontsize=9, color='#53616b')
        axes[1].text(.5, -.035, 'Mean |activity effect| per position', transform=axes[1].transAxes,
                     ha='center', va='top', fontsize=9, color='#53616b')
        for axis, value, color in [(axes[2], baseline, '#53616b'), (axes[3], hidden, '#c65d53')]:
            axis.text(.5, -.035, f'Spearman ρ = {value:.3f}', transform=axis.transAxes,
                      ha='center', va='top', fontsize=11, color=color,
                      fontweight='bold' if axis is axes[3] else 'normal')
        figure.suptitle('Src kinase domain: agreement with experimental functional sensitivity',
                       fontsize=12, y=.975, color='#263744')
        axes[3].text(.5, -.14, f'Δρ = +{hidden-baseline:.3f} over structure',
                     transform=axes[3].transAxes, ha='center', va='top',
                     fontsize=9, color='#c65d53', fontweight='bold')
        color_axis = figure.add_axes([.40, .17, .26, .022])
        palette = LinearSegmentedColormap.from_list('pymol_rank', ['white', 'yellow', 'red'])
        bar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=palette),
                             cax=color_axis, orientation='horizontal', ticks=[0, .5, 1])
        bar.ax.tick_params(labelsize=7, length=2)
        bar.set_label('B–D: within-construct percentile (low → high)', fontsize=8, labelpad=1)
        figure.text(.5, .025,
                    'Same AF2 structure, camera, crop and color mapping; gray: no evaluated effect. '
                    'Correlations use all 269 positions, not selected sites.\n'
                    'Activation segment is anatomical context, not a prediction target. '
                    'Functional magnitude does not specify activation direction. Source-test-selected OOF.',
                    ha='center', fontsize=7.3, color='#53616b')
        for extension in ['png', 'pdf']:
            figure.savefig(DIRECTORY / ('src_functional_comparison.'+extension), dpi=300, facecolor='white')
        plt.close(figure)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    assemble()
