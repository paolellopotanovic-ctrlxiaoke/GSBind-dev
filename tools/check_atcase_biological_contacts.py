"""Test contacts against deposited 1ZA2 biological-assembly operators."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases/pocket_screen'


def check():
    path = DIRECTORY / '1ZA2/1za2.pdb'
    lines = path.read_text().splitlines()
    matrices, atoms = {}, {}
    for line in lines:
        if line.startswith('REMARK 350') and 'BIOMT' in line:
            match = re.search(r'BIOMT([123])\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',line)
            if match:
                row,operation=map(int,match.group(1,2))
                matrices.setdefault(operation,np.zeros((3,4)))[row-1]=list(map(float,match.group(3,4,5,6)))
        if line.startswith('ATOM') and line[17:20].strip() in {
                'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}:
            if line[76:78].strip() in ['H','D'] or line[16] not in [' ','A']:
                continue
            key=(line[21],line[22:27].strip(),line[17:20].strip())
            atoms.setdefault(key,[]).append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    assert len(matrices)==3
    partners=[]
    for operation,matrix in matrices.items():
        for (chain,residue,amino),coordinates in atoms.items():
            if operation==1 and chain=='B':
                continue
            transformed=np.asarray(coordinates)@matrix[:,:3].T+matrix[:,3]
            partners.append((operation,chain,residue,amino,transformed))
    top=pd.read_csv(DIRECTORY / '1ZA2/TOPK_RESIDUES.csv')
    chosen=top.loc[top.fraction.eq(.05)&top.method.eq('gsbind_hidden_probe')&top.selected]
    rows=[]
    for residue in chosen.itertuples():
        coordinates=np.asarray(atoms[('B',str(residue.pdb_resi),
                                      next(key[2] for key in atoms if key[:2]==('B',str(residue.pdb_resi))))])
        distances=[]
        for operation,chain,number,amino,target in partners:
            distances.append((float(cdist(coordinates,target).min()),operation,chain,number,amino))
        closest=min(distances)
        close=[item for item in distances if item[0]<=5]
        rows.append(dict(pdb_resi=residue.pdb_resi,canonical_position=residue.canonical_position,
                         amino=residue.amino,source_pocket=residue.source_site,
                         ligand_distance_A=residue.ligand_min_distance,
                         nearest_partner_distance_A=closest[0],nearest_operation=closest[1],
                         nearest_chain=closest[2],nearest_resi=closest[3],nearest_amino=closest[4],
                         contact_4A=closest[0]<=4,contact_5A=closest[0]<=5,
                         contact_partner_residues_5A=';'.join(f'{item[2]}@{item[1]}:{item[3]}{item[4]}' for item in sorted(close))))
    table=pd.DataFrame(rows)
    table.to_csv(DIRECTORY / '1ZA2/TOP5_BIOLOGICAL_CONTACTS.csv',index=False)
    all_selected=[]
    for (chain,number,amino),coordinates in atoms.items():
        if chain!='B':
            continue
        best=min((float(cdist(coordinates,target).min()),operation,other_chain,residue)
                 for operation,other_chain,residue,other_amino,target in partners)
        all_selected.append(dict(pdb_resi=int(number),partner_min_distance_A=best[0],contact_5A=best[0]<=5))
    pd.DataFrame(all_selected).to_csv(DIRECTORY / '1ZA2/ALL_RESIDUE_BIOLOGICAL_CONTACTS.csv',index=False)
    provenance=dict(structure='1ZA2',target='chain B, identity operator 1',
                    assembly='Deposited REMARK350 biomolecule1, author/PISA dodecamer, operators1-3 on chainsA-D',
                    source_of_partners='Only deposited biological assembly, not arbitrary crystal neighbors',
                    definition='Minimum non-hydrogen protein-atom distance; alternate locations blank/A',
                    thresholds_A=[4,5],source_pocket_hits=int(table.source_pocket.sum()),
                    off_annotation_rows=table.loc[~table.source_pocket].to_dict('records'),
                    no_claim_of_joint_training_or_causal_mechanism=True)
    (DIRECTORY / '1ZA2/CONTACT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(table.to_string(index=False))


if __name__=='__main__':
    check()
