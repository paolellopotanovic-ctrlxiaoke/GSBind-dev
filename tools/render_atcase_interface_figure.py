"""Interface-aware error-map rendering from independent contact labels."""

from importlib.metadata import version
import json
import pymol

import render_structural_readout_cases as renderer


def render():
    version('pymol-open-source')
    pymol.finish_launching(['pymol','-cq'])
    cmd=pymol.cmd
    renderer.cmd=cmd
    renderer.DIRECTORY=renderer.ROOT / 'figures/structural_cases/pocket_screen/1ZA2'
    specification=json.loads((renderer.DIRECTORY / 'INTERFACE_FIGURE.json').read_text())
    renderer.setup(renderer.DIRECTORY / '1za2.pdb')
    cmd.remove('polymer.protein and not chain B')
    for name,hexadecimal in specification['palette'].items():
        cmd.set_color(name,[int(hexadecimal[index:index+2],16)/255 for index in [1,3,5]])
    cmd.show('surface','polymer.protein')
    cmd.set('transparency',.3)
    cmd.show('sticks','resn CTP and chain B')
    cmd.color('violet','resn CTP and chain B and elem C')
    cmd.set('stick_radius',.24)
    cmd.orient('polymer.protein')
    cmd.turn('y',25)
    cmd.turn('x',-15)
    cmd.zoom('polymer.protein or (resn CTP and chain B)',buffer=3)
    camera=cmd.get_view()
    rows=renderer.read_table('INTERFACE_AWARE_TOPK.csv')
    for fraction in [.05,.10]:
        for method in ['gsbind_score_probe','structure_only','gsbind_hidden_probe']:
            cmd.color('background','polymer.protein')
            current=[row for row in rows if float(row['fraction'])==fraction and row['method']==method]
            for category in specification['palette']:
                residues='+'.join(row['pdb_resi'] for row in current if row['display_category']==category)
                if residues:
                    cmd.color(category,'polymer.protein and chain B and resi '+residues)
            cmd.set_view(camera)
            renderer.save(f'interface_top{int(fraction*100)}_{method}')
    cmd.color('background','polymer.protein')
    current=[row for row in rows if float(row['fraction'])==.05 and row['method']=='gsbind_hidden_probe']
    for condition,color in [(lambda row:row['contact_5A']=='True','interface_context'),
                             (lambda row:row['source_site']=='True','pocket_hit'),
                             (lambda row:row['source_site']=='True' and row['contact_5A']=='True','overlap')]:
        residues='+'.join(row['pdb_resi'] for row in current if condition(row) and row['identity_match']=='True')
        if residues:
            cmd.color(color,'polymer.protein and chain B and resi '+residues)
    cmd.set_view(camera)
    renderer.save('interface_reference')
    cmd.quit()


if __name__=='__main__':
    render()
