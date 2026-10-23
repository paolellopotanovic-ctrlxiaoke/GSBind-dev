"""Matched-view top-k error maps using open-source PyMOL."""

from importlib.metadata import version
import json

import pymol
import render_structural_readout_cases as renderer


def render():
    version('pymol-open-source')
    pymol.finish_launching(['pymol', '-cq'])
    cmd = pymol.cmd
    renderer.cmd = cmd
    renderer.setup(renderer.DIRECTORY / 'assets/5mo4.pdb')
    cmd.set_color('hit_color', [.78, .23, .22])
    cmd.set_color('miss_color', [.94, .64, .23])
    cmd.set_color('off_color', [.22, .52, .70])
    cmd.show('surface', 'polymer.protein')
    cmd.set('transparency', .25)
    cmd.show('sticks', 'resn AY7+NIL')
    cmd.set('stick_radius', .22)
    cmd.color('violet', 'resn AY7 and elem C')
    cmd.color('olive', 'resn NIL and elem C')
    cmd.orient('polymer.protein')
    cmd.turn('y', 35)
    cmd.turn('x', -15)
    cmd.zoom('polymer.protein or resn AY7+NIL', buffer=4)
    camera = cmd.get_view()
    rows = renderer.read_table('ABL1_TOPK_RESIDUES.csv')
    colors = dict(hit='hit_color', missed_annotation='miss_color', off_annotation_high='off_color')
    for fraction in [.05, .10]:
        for method in ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe']:
            cmd.color('neutral', 'polymer.protein')
            current = [row for row in rows if float(row['fraction']) == fraction and row['method']==method]
            for category, color in colors.items():
                residues = '+'.join(row['pdb_resi'] for row in current if row['category']==category)
                if residues:
                    cmd.color(color, 'polymer.protein and chain A and resi ' + residues)
            cmd.set_view(camera)
            renderer.save(f'abl1_top{int(fraction*100)}_{method}')
    (renderer.DIRECTORY / 'ABL1_TOPK_RENDER.json').write_text(json.dumps(dict(
        renderer='pymol-open-source', version=cmd.get_version()[0], same_view=True,
        selection='Exact ceil(fraction * full_protein_N), descending score; ties by ascending position',
        gray='background or sequence mismatch',
        blue='selected outside supplied annotation, not an experimentally proven false positive',
        hidden_off_annotation_predictions_retained=True), indent=2)+'\n')
    cmd.quit()


if __name__ == '__main__':
    render()
