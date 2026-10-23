"""Export traceable ABL1 / Src case overlays without fitting or model selection."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1


ROOT = Path(__file__).resolve().parents[1]


def prepare(experiment):
    directory = ROOT / 'figures/structural_cases'
    assets = directory / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    original = experiment / 'datasets/allosteric/raw/pdb/5mo4.pdb'
    structure = assets / '5mo4.pdb'
    shutil.copyfile(original, structure)
    raw = pd.read_csv(ROOT / 'datasets/allosteric/raw/AlloBench.csv')
    record = raw.loc[raw.pdb_uniprot.eq('P00519') & raw.allosteric_pdb.eq('5MO4')].iloc[0]
    manifest = pd.read_csv(ROOT / 'datasets/allosteric/manifest.csv')
    sequence = manifest.loc[manifest.uniprot.eq('P00519'), 'sequence'].iloc[0]
    residues = {}
    for line in structure.read_text().splitlines():
        if line.startswith('ATOM') and line[21] == 'A' and line[12:16].strip() == 'CA':
            residues[line[22:27].strip()] = protein_letters_3to1[line[17:20].strip()]
    observed = ''.join(residues.values())
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-3,
                             open_gap_score=-8, extend_gap_score=-.5)
    alignment = aligner.align(sequence, observed)[0]
    mapping = {}
    for target_block, source_block in zip(*alignment.aligned):
        for target_index, source_index in zip(range(*target_block), range(*source_block)):
            mapping[source_index] = target_index
    sites = {str(item.split('-')[-1]) for item in ast.literal_eval(record.allosteric_site_residue)}
    frame = pd.read_csv(ROOT / 'results/downstream/AlloBench/predictions.csv.gz')
    frame = frame.loc[frame.uniprot.eq('P00519')].copy()
    for method in ['gsbind_hidden_probe', 'gsbind_score_probe', 'structure_only']:
        frame[method + '_percentile'] = frame[method].rank(pct=True)
    frame = frame.set_index('residue_index_0')
    rows = []
    for source_index, (number, amino) in enumerate(residues.items()):
        target_index = mapping[source_index]
        current = frame.loc[target_index]
        rows.append(dict(pdb_resi=number, canonical_position=target_index+1, observed_amino=amino,
                         canonical_amino=sequence[target_index], identity_match=amino==sequence[target_index],
                         site_5mo4=number in sites, allobench_union_label=int(current.label),
                         hidden_percentile=current.gsbind_hidden_probe_percentile,
                         score_percentile=current.gsbind_score_probe_percentile,
                         structure_percentile=current.structure_only_percentile))
    mapped = pd.DataFrame(rows)
    official = pd.read_csv(ROOT / 'datasets/allosteric/allobench_record_mapping.csv')
    expected = official.loc[official.uniprot.eq('P00519') & official.source_pdb.eq('5MO4'), 'positions_0'].iloc[0]
    np.testing.assert_array_equal(sorted(mapped.loc[mapped.site_5mo4, 'canonical_position']-1),
                                   sorted(map(int, expected.split(';'))))
    mapped.to_csv(directory / 'ABL1_RESIDUE_MAPPING.csv', index=False)
    frame.reset_index().to_csv(directory / 'ABL1_ALL_EVALUATED_PREDICTIONS.csv', index=False)
    detail = pd.read_csv(ROOT / 'results/downstream/DETAILED_METRICS.csv')
    detail.loc[detail.task.eq('AlloBench') & detail.uniprot.eq('P00519')].to_csv(
        directory / 'ABL1_CASE_METRICS.csv', index=False)
    src = pd.read_csv(ROOT / 'results/downstream/Src/predictions.csv.gz')
    for method in ['gsbind_hidden_probe', 'structure_only']:
        src[method + '_percentile'] = src.groupby('case_id')[method].rank(pct=True)
    src['functional_percentile'] = src.groupby('case_id').functional_magnitude.rank(pct=True)
    src['activation_segment'] = src.physical_position.between(407, 435)
    src.to_csv(directory / 'SRC_POSITION_OVERLAYS.csv', index=False)
    functional_manifest = pd.read_csv(ROOT / 'datasets/functional/manifest.csv')
    for case in ['src_fl_af2', 'src_kd_af2']:
        row = functional_manifest.loc[functional_manifest.apex_case_id.eq(case)].iloc[0]
        shutil.copyfile(ROOT / row.receptor_pdb_path, assets / (case + '.pdb'))
    uniprot_path = ROOT / 'runs/case_source_check/P12931.json'
    uniprot = json.loads(uniprot_path.read_text())
    sequence = uniprot['sequence']['value']
    assert sequence[406:409] == 'DFG' and sequence[432:435] == 'APE' and sequence[418] == 'Y'
    assert any(feature['description'].startswith('Phosphotyrosine') and
               feature['location']['start']['value']==419 for feature in uniprot['features'])
    annotation = dict(accession='P12931', source='UniProt REST P12931',
                      source_sha256=hashlib.sha256(uniprot_path.read_bytes()).hexdigest(),
                      domains=[feature for feature in uniprot['features'] if feature['type']=='Domain'],
                      activation_segment=dict(start=407, end=435,
                                              definition='DFG through APE inclusive, sequence-defined'),
                      phosphosite_419=[feature for feature in uniprot['features']
                                      if feature['location']['start'].get('value')==419],
                      no_activation_state_assignment=True)
    (directory / 'SRC_ANNOTATIONS.json').write_text(json.dumps(annotation, indent=2)+'\n')
    provenance = dict(case_choice='Illustrative curated ABL1 case with two bound pocket ligands and positive hidden-readout gain; not a random or unbiased sample',
                      source_pdb='5MO4', source_chain='A', uniprot='P00519',
                      pdb_title='ABL1 kinase (T334I_D382N) in complex with asciminib and nilotinib',
                      experimental_construct_is_mutant=True,
                      ligand_allosteric=dict(resn='AY7', chain='A', resi='602', name='asciminib'),
                      ligand_atp_pocket=dict(resn='NIL', chain='A', resi='601', name='nilotinib'),
                      publication=dict(doi='10.1038/nature21702', pubmed='28329763'),
                      mapped_residues=len(mapped), mismatches=mapped.loc[~mapped.identity_match].to_dict('records'),
                      annotated_5mo4_sites=int(mapped.site_5mo4.sum()),
                      holo_used_for_visualization_only=True,
                      prediction_input='Saved monomer features, not the ligand-bound display complex',
                      display_rank_reference='All evaluated residues of the corresponding protein/construct; ties use average rank',
                      benchmark_labels='AlloBench union across source records; displayed ground truth restricted to 5MO4 annotation',
                      source_checkpoint_test_selected=True,
                      no_training_or_checkpoint_reselection=True,
                      files={str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in assets.iterdir()})
    (directory / 'CASE_PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(directory=str(directory), mapped_residues=len(mapped),
                          ligand_annotation_sites=int(mapped.site_5mo4.sum()),
                          construct_mismatches=int((~mapped.identity_match).sum())), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    prepare(parser.parse_args().experiment_root.resolve())
