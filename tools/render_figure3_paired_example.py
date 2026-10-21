"""Render fixed-rule paired CA traces with true and missed interface markers."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figure3_binding_reliability_20261021'


def run():
    (OUT/'figures').mkdir(exist_ok=True)
    table = pd.read_csv(OUT/'I_example_residues.csv')
    metrics = pd.read_csv(OUT/'I_example_metrics.csv')
    reference = table[['holo_x', 'holo_y', 'holo_z']].to_numpy()
    center = reference.mean(axis=0)
    _, _, basis = np.linalg.svd(reference-center, full_matrices=False)
    geometry = (reference-center)@basis.T
    moving = (table[['state_x', 'state_y', 'state_z']].to_numpy()-center)@basis.T
    figure, axes = plt.subplots(1, 3, figsize=(8.6, 3.3))
    arrays = [(geometry, table.reference_label.to_numpy(), 'Holo reference interface'),
        (moving, table.gaussian_direct128_state_score.to_numpy(), 'AF2: GSBind'),
        (moving, table.pesto_official_state_score.to_numpy(), 'AF2: PeSTo')]
    image = None
    for index, (axis, (coordinates, values, title)) in enumerate(zip(axes, arrays)):
        segments = np.stack([coordinates[:-1, :2], coordinates[1:, :2]], axis=1)
        connected = np.diff(table.holo_index.to_numpy() if index==0 else table.state_index.to_numpy())==1
        axis.add_collection(LineCollection(segments[connected], color='#aaaaaa', linewidth=.9, alpha=.6))
        image = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=values, cmap='coolwarm', vmin=0, vmax=1, s=10, zorder=2)
        if index:
            model = 'gaussian_direct128' if index==1 else 'pesto_official'
            chosen = table[model+'_top10'].astype(bool).to_numpy()
            correct = chosen & table.reference_label.eq(1).to_numpy()
            wrong = chosen & ~table.reference_label.eq(1).to_numpy()
            axis.scatter(coordinates[correct, 0], coordinates[correct, 1], s=35, facecolors='none', edgecolors='#168a45', linewidths=1.1)
            axis.scatter(coordinates[wrong, 0], coordinates[wrong, 1], s=25, marker='x', c='black', linewidths=1.)
            value = metrics[metrics.model.eq(model)].iloc[0]
            title += f'\nAP={value.state_ap:.3f}; top10 hit={int(correct.sum())}/10'
        axis.set_title(title, fontsize=9)
        axis.set_aspect('equal')
        axis.autoscale_view()
        axis.axis('off')
    figure.subplots_adjust(left=.01, right=.93, bottom=.16, top=.82, wspace=.05)
    bar_axis = figure.add_axes([.945, .25, .012, .43])
    figure.colorbar(image, cax=bar_axis, label='Label / score')
    figure.text(.5, .035, 'CA trace, common PCA view; green rings: correct top10; black crosses: wrong top10', ha='center', fontsize=8)
    figure.suptitle(table.case_id.iloc[0].split('::')[1].split('__')[0]+' | fixed median-representative selection', fontsize=10)
    for suffix in ['png', 'pdf']:
        figure.savefig(OUT/'figures'/f'I_paired_structure.{suffix}', dpi=220)
    plt.close(figure)


if __name__=='__main__':
    run()
