"""Headless PyMOL ray tracing of real ABL1 and Src display structures."""

import csv
import json
from importlib.metadata import version
from pathlib import Path

import pymol


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases'


def read_table(name):
    with (DIRECTORY / name).open() as handle:
        return list(csv.DictReader(handle))


def setup(path):
    cmd.reinitialize()
    cmd.load(str(path), 'protein')
    cmd.remove('solvent or hydro')
    cmd.hide('everything')
    cmd.bg_color('white')
    cmd.set('orthoscopic', 1)
    cmd.set('ray_opaque_background', 0)
    cmd.set('ray_shadows', 0)
    cmd.set('antialias', 2)
    cmd.set('ambient', .55)
    cmd.set('specular', .15)
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('cartoon_sampling', 14)
    cmd.set('label_color', 'black')
    cmd.set('label_size', 18)
    cmd.set_color('neutral', [.80, .83, .85])
    cmd.set_color('coral', [.78, .32, .29])
    cmd.set_color('violet', [.53, .39, .66])
    cmd.set_color('olive', [.41, .58, .39])
    cmd.color('neutral', 'protein')
    cmd.show('cartoon', 'polymer.protein')


def save(name):
    cmd.save(str(DIRECTORY / (name + '.pse')))
    cmd.png(str(DIRECTORY / (name + '.png')), width=1400, height=1200, dpi=300, ray=1)


def values(rows, key, resi='pdb_resi', mapped_only=False):
    cmd.alter('protein', 'b=-1')
    for row in rows:
        if mapped_only and row['identity_match'] != 'True':
            continue
        cmd.alter('protein and chain A and resi ' + row[resi], 'b=' + row[key])
    selection = 'protein and b > -0.5'
    if cmd.count_atoms(selection) == 0:
        raise ValueError('No mapped atoms for ' + key)
    cmd.spectrum('b', 'white_yellow_red', selection, minimum=0, maximum=1)


def abl():
    setup(DIRECTORY / 'assets/5mo4.pdb')
    rows = read_table('ABL1_RESIDUE_MAPPING.csv')
    cmd.select('asciminib', 'resn AY7 and chain A and resi 602')
    cmd.select('nilotinib', 'resn NIL and chain A and resi 601')
    assert cmd.count_atoms('asciminib') > 0 and cmd.count_atoms('nilotinib') > 0
    cmd.show('sticks', 'asciminib or nilotinib')
    cmd.color('violet', 'asciminib and elem C')
    cmd.color('olive', 'nilotinib and elem C')
    cmd.set('stick_radius', .20)
    sites = '+'.join(row['pdb_resi'] for row in rows if row['site_5mo4']=='True')
    cmd.select('allosteric_annotation', 'polymer.protein and chain A and resi ' + sites)
    cmd.show('surface', 'polymer.protein')
    cmd.set('transparency', .38)
    cmd.orient('polymer.protein')
    cmd.turn('y', 35)
    cmd.turn('x', -15)
    cmd.zoom('polymer.protein or asciminib or nilotinib', buffer=4)
    camera = cmd.get_view()
    cmd.color('coral', 'allosteric_annotation')
    save('abl1_annotation')
    for key, name in [('hidden_percentile', 'abl1_hidden'), ('score_percentile', 'abl1_binding_score')]:
        cmd.color('neutral', 'polymer.protein')
        values(rows, key, mapped_only=True)
        cmd.color('violet', 'asciminib and elem C')
        cmd.color('olive', 'nilotinib and elem C')
        cmd.set_view(camera)
        save(name)


def src():
    rows = [row for row in read_table('SRC_POSITION_OVERLAYS.csv') if row['case_id']=='src_kd_af2']
    setup(DIRECTORY / 'assets/src_fl_af2.pdb')
    cmd.hide('cartoon', 'resi 1-83')
    cmd.color('skyblue', 'resi 84-145')
    cmd.color('palegreen', 'resi 151-248')
    cmd.color('coral', 'resi 407-435')
    cmd.show('sticks', 'resi 419+530')
    cmd.label('name CA and resi 419', '"Y419"')
    cmd.label('name CA and resi 530', '"Y530"')
    cmd.orient('polymer.protein and resi 84-536')
    cmd.zoom('polymer.protein and resi 84-536', buffer=5)
    save('src_full_context')
    setup(DIRECTORY / 'assets/src_kd_af2.pdb')
    cmd.select('activation_segment', 'resi 407-435')
    cmd.color('coral', 'activation_segment')
    cmd.show('sticks', 'resi 419 or resi 407-409 or resi 433-435')
    cmd.color('olive', 'resi 419')
    cmd.label('name CA and resi 419', '"Y419"')
    cmd.label('name CA and resi 407', '"DFG"')
    cmd.label('name CA and resi 435', '"APE"')
    cmd.orient('polymer.protein')
    cmd.turn('y', 25)
    cmd.zoom('polymer.protein', buffer=4)
    camera = cmd.get_view()
    save('src_activation_segment')
    cmd.hide('sticks')
    cmd.label('all', '""')
    for key, name in [('functional_percentile', 'src_experimental'),
                      ('gsbind_hidden_probe_percentile', 'src_hidden'),
                      ('structure_only_percentile', 'src_structure')]:
        cmd.color('neutral', 'protein')
        values(rows, key, resi='physical_position')
        cmd.set_view(camera)
        save(name)
    (DIRECTORY / 'PYMOL_RENDER.json').write_text(json.dumps(dict(
        version=cmd.get_version()[0], renderer_distribution='pymol-open-source',
        ray_traced=True, width=1400, height=1200,
        identical_view_within_comparisons=True, map_scale='within-protein/construct percentile, 0 to 1',
        uncovered_residues='neutral gray; not zero effect',
        src_activation_segment='human P12931 DFG407-409 through APE433-435',
        src_input='AF2 monomer; not evidence of active or inactive conformational state'), indent=2)+'\n')


if __name__ == '__main__':
    version('pymol-open-source')
    pymol.finish_launching(['pymol', '-cq'])
    cmd = pymol.cmd
    abl()
    src()
    cmd.quit()
