"""Render all selected and missed residues, with matched camera across probes."""

from importlib.metadata import version
import json
import pymol

import render_structural_readout_cases as renderer


def render():
    version('pymol-open-source')
    pymol.finish_launching(['pymol','-cq'])
    cmd = pymol.cmd
    renderer.cmd = cmd
    root = renderer.DIRECTORY / 'pocket_screen'
    provenance = json.loads((root / 'CANDIDATE_PROVENANCE.json').read_text())
    for case in provenance['cases']:
        pdb, chain = case['pdb'], case['chain']
        renderer.DIRECTORY = root / pdb
        renderer.setup(renderer.DIRECTORY / (pdb.lower()+'.pdb'))
        cmd.remove('polymer.protein and not chain '+chain)
        cmd.select('real_ligand', ' or '.join(
            '(resn {resn} and chain {chain} and resi {resi})'.format(**ligand) for ligand in case['ligands']))
        assert cmd.count_atoms('real_ligand')==case['ligand_heavy_atoms']
        cmd.show('sticks','real_ligand')
        cmd.color('violet','real_ligand and elem C')
        cmd.set('stick_radius',.24)
        cmd.show('surface','polymer.protein')
        cmd.set('transparency',.3)
        cmd.set_color('hit_color',[.78,.23,.22])
        cmd.set_color('miss_color',[.94,.64,.23])
        cmd.set_color('off_color',[.22,.52,.70])
        cmd.orient('polymer.protein')
        cmd.turn('y',25)
        cmd.turn('x',-15)
        cmd.zoom('polymer.protein or real_ligand',buffer=3)
        camera = cmd.get_view()
        rows = renderer.read_table('TOPK_RESIDUES.csv')
        for fraction in [.05,.10]:
            for method in ['gsbind_score_probe','structure_only','gsbind_hidden_probe']:
                cmd.color('neutral','polymer.protein')
                current=[row for row in rows if float(row['fraction'])==fraction and row['method']==method]
                for category,color in [('hit','hit_color'),('missed_annotation','miss_color'),('off_annotation_high','off_color')]:
                    residues='+'.join(row['pdb_resi'] for row in current if row['category']==category)
                    if residues:
                        cmd.color(color,'polymer.protein and chain '+chain+' and resi '+residues)
                cmd.set_view(camera)
                renderer.save(f'top{int(fraction*100)}_{method}')
        cmd.color('neutral','polymer.protein')
        annotated='+'.join(sorted({row['pdb_resi'] for row in rows if row['source_site']=='True'}))
        cmd.color('coral','polymer.protein and chain '+chain+' and resi '+annotated)
        renderer.save('source_annotation')
    (root / 'PYMOL_RENDER.json').write_text(json.dumps(dict(renderer='pymol-open-source',version=cmd.get_version()[0],
        same_camera_across_methods_and_thresholds=True, omitted_partner_chains=True,
        displayed_chain='Only annotated target chain; selected mapped residues all retained; real ligands shown',
        no_holo_input_to_probe=True),indent=2)+'\n')
    cmd.quit()


if __name__ == '__main__':
    render()
