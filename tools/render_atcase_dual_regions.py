"""Display independently checked CTP-pocket and regulatory-dimer contacts."""

from importlib.metadata import version
from pathlib import Path

import pymol
import render_structural_readout_cases as renderer


def render():
    version('pymol-open-source')
    pymol.finish_launching(['pymol','-cq'])
    cmd=pymol.cmd
    renderer.cmd=cmd
    renderer.DIRECTORY=renderer.ROOT / 'figures/structural_cases/pocket_screen/1ZA2'
    renderer.setup(renderer.DIRECTORY / '1za2.pdb')
    cmd.remove('polymer.protein and not chain B+D')
    cmd.color('neutral','chain B')
    cmd.set_color('partnerblue',[.58,.72,.82])
    cmd.color('partnerblue','chain D')
    cmd.show('surface','polymer.protein and chain B')
    cmd.set('transparency',.40,'chain B')
    cmd.show('cartoon','chain D')
    cmd.show('sticks','chain B and resi 11+12+17+58+60+86+24+27')
    cmd.color('coral','chain B and resi 11+12+17+58+60+86')
    cmd.color('marine','chain B and resi 24+27')
    cmd.show('sticks','resn CTP and chain B')
    cmd.color('violet','resn CTP and chain B and elem C')
    cmd.label('chain B and resi 24+27 and name CA','resn+resi')
    cmd.orient('polymer.protein and chain B+D')
    cmd.turn('y',25)
    cmd.turn('x',-15)
    cmd.zoom('polymer.protein and chain B+D',buffer=4)
    renderer.save('top5_pocket_and_interface_context')
    cmd.quit()


if __name__=='__main__':
    render()
