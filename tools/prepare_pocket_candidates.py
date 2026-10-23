"""Validate curated source pockets, sequence mapping and ligand proximity."""

import ast
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases/pocket_screen'
METHODS = ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe', 'pesto_score_probe']


def prepare():
    raw = pd.read_csv(ROOT / 'datasets/allosteric/raw/AlloBench.csv')
    source_mapping = pd.read_csv(ROOT / 'datasets/allosteric/allobench_record_mapping.csv')
    manifest = pd.read_csv(ROOT / 'datasets/allosteric/manifest.csv')
    predictions = pd.read_csv(ROOT / 'results/downstream/AlloBench/predictions.csv.gz')
    screening = pd.read_csv(DIRECTORY / 'ALL_SOURCE_POCKET_SCREEN.csv')
    summary, provenance = [], []
    for pdb in ['2XPU', '1ZA2', '1H9G']:
        candidate = screening.loc[screening.pdb.eq(pdb)].iloc[0]
        source = raw.iloc[int(candidate.record_id)]
        identifier = source.pdb_uniprot
        destination = DIRECTORY / pdb
        destination.mkdir(exist_ok=True)
        original = DIRECTORY / 'inputs' / (pdb.lower()+'.pdb')
        if not original.exists():
            original = ROOT.parent / 'GSbind/datasets/allosteric/raw/pdb' / (pdb.lower()+'.pdb')
        structure = destination / (pdb.lower()+'.pdb')
        shutil.copyfile(original, structure)
        annotation = []
        for item in ast.literal_eval(source.allosteric_site_residue):
            chain, amino, residue = item.split('-')
            annotation.append((chain, amino, residue))
        chains = {item[0] for item in annotation}
        assert len(chains)==1
        chain = next(iter(chains))
        sequence = manifest.loc[manifest.uniprot.eq(identifier), 'sequence'].iloc[0]
        residues, atoms = {}, {}
        ligand_atoms = []
        ligand_keys = set(zip(str(source.modulator_alias).split(';'), str(source.modulator_chain).split(';'),
                              str(source.modulator_resi).split(';')))
        ligand_lines = []
        for line in structure.read_text().splitlines():
            if not line.startswith(('ATOM  ', 'HETATM')) or line[76:78].strip()=='H':
                continue
            coordinate = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            name, atom_chain, number = line[17:20].strip(), line[21], line[22:27].strip()
            if (name, atom_chain, number) in ligand_keys:
                ligand_atoms.append(coordinate)
                ligand_lines.append(line)
            if line.startswith('ATOM') and atom_chain==chain and name in protein_letters_3to1:
                atoms.setdefault(number, []).append(coordinate)
                if line[12:16].strip()=='CA' and number not in residues:
                    residues[number]=(name, protein_letters_3to1[name], coordinate)
        assert len(ligand_atoms)>0
        observed = ''.join(item[1] for item in residues.values())
        aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-3,
                                 open_gap_score=-8, extend_gap_score=-.5)
        alignment = aligner.align(sequence, observed)[0]
        mapping = {}
        for target_block, source_block in zip(*alignment.aligned):
            mapping.update(zip(range(*source_block), range(*target_block)))
        frame = predictions.loc[predictions.uniprot.eq(identifier)].copy()
        indexed = frame.set_index('residue_index_0')
        sites = {item[2]: item[1] for item in annotation}
        mapped = []
        for index, (number, (resname, amino, coordinate)) in enumerate(residues.items()):
            target = mapping.get(index)
            if target is None or target not in indexed.index:
                continue
            actual = indexed.loc[target]
            site = number in sites
            if site:
                assert resname==sites[number]
            mapped.append(dict(pdb_resi=number, canonical_position=target+1, amino=amino,
                               identity_match=amino==sequence[target], source_site=site,
                               ca_x=coordinate[0], ca_y=coordinate[1], ca_z=coordinate[2],
                               ligand_min_distance=float(cdist(atoms[number], ligand_atoms).min()),
                               **{method: actual[method] for method in METHODS}))
        mapped = pd.DataFrame(mapped)
        reference = source_mapping.loc[source_mapping.record_id.eq(candidate.record_id)].iloc[0]
        expected = set(map(int, reference.positions_0.split(';')))
        assert set(mapped.loc[mapped.source_site, 'canonical_position']-1)==expected
        assert mapped.loc[mapped.source_site, 'identity_match'].all()
        mapped.to_csv(destination / 'RESIDUE_MAPPING.csv', index=False)
        frame.to_csv(destination / 'ALL_PROTEIN_PREDICTIONS.csv', index=False)
        rows = []
        for fraction in [.05, .10]:
            count = math.ceil(len(frame)*fraction)
            for method in METHODS:
                ordered = frame.sort_values([method,'residue_index_0'], ascending=[False, True])
                selected = set(ordered.head(count).residue_index_0+1)
                hits = len(selected & {position+1 for position in expected})
                current = mapped.copy()
                current['selected'] = current.canonical_position.isin(selected)
                current['category'] = np.select(
                    [~current.identity_match, current.selected & current.source_site,
                     ~current.selected & current.source_site, current.selected & ~current.source_site],
                    ['sequence_mismatch','hit','missed_annotation','off_annotation_high'], default='background')
                current['fraction'], current['top_k'], current['method'] = fraction, count, method
                rows.append(current)
                evaluable = current.loc[current.selected & current.identity_match]
                coords = evaluable[['ca_x','ca_y','ca_z']].to_numpy()
                if len(coords):
                    clusters, labels = connected_components(cdist(coords, coords)<=8, directed=False)
                    largest = int(np.bincount(labels).max())
                else:
                    clusters, largest = 0, 0
                summary.append(dict(pdb=pdb, gene=source.target_gene, uniprot=identifier, chain=chain,
                                    fraction=fraction, method=method, protein_N=len(frame), top_k=count,
                                    annotation_N=len(expected), hits=hits, precision=hits/count,
                                    recall=hits/len(expected), mapped_selected=len(evaluable),
                                    selected_within_6A=int(evaluable.ligand_min_distance.le(6).sum()),
                                    selected_within_8A=int(evaluable.ligand_min_distance.le(8).sum()),
                                    mapped_off_annotation=int(current.category.eq('off_annotation_high').sum()),
                                    ca8A_components=int(clusters), ca8A_largest_component_N=largest))
        pd.concat(rows, ignore_index=True).to_csv(destination / 'TOPK_RESIDUES.csv', index=False)
        provenance.append(dict(pdb=pdb, uniprot=identifier, chain=chain, record_id=int(candidate.record_id),
                               observed_mapped_N=len(mapped), mismatches=int((~mapped.identity_match).sum()),
                               ligands=[dict(resn=name,chain=part,resi=number) for name,part,number in sorted(ligand_keys)],
                               ligand_heavy_atoms=len(ligand_atoms), annotation_N=len(expected),
                               title=[line[10:].strip() for line in structure.read_text().splitlines() if line.startswith('TITLE')],
                               doi=[line.split('DOI',1)[1].strip() for line in structure.read_text().splitlines() if line.startswith('JRNL') and 'DOI' in line],
                               sha256=hashlib.sha256(structure.read_bytes()).hexdigest(),
                               source_annotation_mapping_matches=True,
                               full_protein_predictions_not_refit=True))
    pd.DataFrame(summary).to_csv(DIRECTORY / 'CANDIDATE_SPATIAL_SUMMARY.csv', index=False)
    (DIRECTORY / 'CANDIDATE_PROVENANCE.json').write_text(json.dumps(dict(
        screen_source_records=len(screening), screen_proteins=int(screening.uniprot.nunique()),
        eligibility='Mapped source pocket with >=5 positions, all in evaluated protein',
        case_selection='Post-hoc illustrative source-test-selected cases; not independent evidence',
        metrics='Source-pocket-specific labels, not the full AlloBench union metric',
        spatial_check='Heavy-atom minimum distance to specified real source ligand(s); CA graph threshold 8 A',
        exclusions='FKBP12/rapamycin example not prioritized: ternary recruitment may confuse the regulatory-pocket story',
        cases=provenance), indent=2)+'\n')
    print(pd.DataFrame(summary).query('method=="gsbind_hidden_probe"').to_string(index=False))


if __name__ == '__main__':
    prepare()
