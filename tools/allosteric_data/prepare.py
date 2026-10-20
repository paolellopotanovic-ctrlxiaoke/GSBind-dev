import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil

from Bio.Align import PairwiseAligner
import numpy as np
import pandas as pd

from common import LEGACY, PACKAGE, ROOT, SUPPLIED, fetch, write_json


AMINO = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 'ARNDCQEGHILKMFPSTWYV'))


def observed(path, chain=None):
    residues = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM  ') or line[12:16].strip() != 'CA' or line[16] not in ' A' or (chain and line[21] != chain):
            continue
        name = line[17:20].strip()
        if name in AMINO:
            residues.setdefault((line[21], line[22:27].strip()), AMINO[name])
    return residues


def align(target, query):
    if target == query:
        return dict(enumerate(range(len(target)))), 1.
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-3, open_gap_score=-8, extend_gap_score=-.5)
    alignment = aligner.align(target, query)[0]
    mapping = {}
    identical = 0
    for target_block, query_block in zip(*alignment.aligned):
        for target_index, query_index in zip(range(*target_block), range(*query_block)):
            mapping[query_index] = target_index
            identical += target[target_index] == query[query_index]
    return mapping, identical / max(1, len(mapping))


def prepare_structure(item):
    identifier, expected, previous = item
    try:
        if previous is not None:
            row = dict(previous)
            row['uniprot'] = identifier
            row['structure_status'] = 'reused_verified_AFDB'
            return row
        entries = json.loads(fetch('https://alphafold.ebi.ac.uk/api/prediction/' + identifier, ROOT / 'raw/afdb' / (identifier + '.json')))
        candidates = [entry for entry in entries if entry.get('sequence') == expected]
        entry = candidates[0] if candidates else entries[0]
        sequence = entry['sequence']
        if not candidates and expected:
            mapping, identity = align(sequence, expected)
            if identity < .95 or len(mapping) / max(1, len(expected)) < .8:
                raise ValueError('AFDB/source sequence mismatch')
        path = ROOT / 'raw/structures' / (identifier + '.pdb')
        fetch(entry['pdbUrl'], path)
        chain = entry.get('chainId', 'A')
        residues = observed(path, chain)
        if ''.join(residues.values()) != sequence:
            raise ValueError('AFDB observed sequence mismatch')
        numbers = [key[1] for key in residues]
        if any(not number.isdigit() for number in numbers):
            raise ValueError('Unsupported AFDB residue identifiers')
        return dict(uniprot=identifier, receptor_uniprot=identifier, apex_case_id='allov2_' + identifier, system_id=identifier,
                    input_state='afdb_monomer', receptor_side='monomer', receptor_chain_id=chain,
                    receptor_pdb_path=str(path), receptor_pdb_relpath=identifier + '.pdb', sequence=sequence,
                    num_residues=len(sequence), input_residue_ids=';'.join(numbers), mapped_uniprot_positions=';'.join(map(str, range(1, len(sequence)+1))),
                    labels=';'.join(['0']*len(sequence)), label_mask=';'.join(['0']*len(sequence)), structure_status='downloaded_AFDB', compact_path='')
    except Exception as error:
        return dict(uniprot=identifier, structure_status='failed', reason=type(error).__name__ + ': ' + str(error))


def map_record(item):
    index, record, structure, processed = item
    output = dict(record_id=index, uniprot=record['pdb_uniprot'], source_pdb=record['allosteric_pdb'], source_pubmed=str(record.get('pubmed_id', '')))
    try:
        sequence = structure['sequence']
        if processed is not None:
            upstream = processed['uniprot_sequence']
            sites = ast.literal_eval(processed['allosteric_site_resid'])
            mapping, identity = align(sequence, upstream)
            if identity < .95:
                raise ValueError('processed/source sequence identity below .95')
            positions = [mapping[int(site)-1] for site in sites if int(site)-1 in mapping and sequence[mapping[int(site)-1]] == upstream[int(site)-1]]
            if len(positions) != len(sites):
                raise ValueError('processed positions incomplete')
            method = 'processed_verified_sequence_positions'
        else:
            path = ROOT / 'raw/pdb' / (str(record['allosteric_pdb']).lower() + '.pdb')
            if not path.exists():
                raise FileNotFoundError('source PDB download unavailable')
            residues = observed(path)
            sites = ast.literal_eval(record['allosteric_site_residue'])
            parsed = []
            for site in sites:
                match = re.fullmatch(r'([A-Za-z0-9]+)-([A-Z]{3})-(-?\d+[A-Za-z]?)', site)
                if match is None or match[2] not in AMINO:
                    raise ValueError('Unsupported source residue ' + site)
                parsed.append((match[1], match[3], AMINO[match[2]]))
            maps = {}
            for chain in {site[0] for site in parsed}:
                keys = [key for key in residues if key[0] == chain]
                query = ''.join(residues[key] for key in keys)
                if not query:
                    raise ValueError('source chain absent')
                mapping, identity = align(sequence, query)
                if identity < .95:
                    continue
                maps[chain] = {key: mapping[ordinal] for ordinal, key in enumerate(keys) if ordinal in mapping and query[ordinal] == sequence[mapping[ordinal]]}
            positions = []
            for chain, number, amino in parsed:
                if chain not in maps:
                    continue
                key = (chain, number)
                if residues.get(key) != amino or key not in maps[chain]:
                    raise ValueError('source residue absent/identity/mapping mismatch')
                positions.append(maps[chain][key])
            method = 'observed_source_PDB_alignment'
        if not positions:
            raise ValueError('No mapped sites')
        output.update(status='mapped', method=method, positions_0=';'.join(map(str, sorted(set(positions)))), sites=len(set(positions)))
    except Exception as error:
        output.update(status='failed', reason=type(error).__name__ + ': ' + str(error))
    return output


def source_download(identifier):
    try:
        fetch('https://files.rcsb.org/download/' + identifier + '.pdb', ROOT / 'raw/pdb' / (identifier.lower()+'.pdb'), retries=2)
        return dict(pdb=identifier, status='downloaded')
    except Exception as error:
        return dict(pdb=identifier, status='failed', reason=str(error))


def main():
    raw = pd.read_csv(PACKAGE / 'datasets/allosteric/raw/AlloBench.csv')
    processed = pd.read_csv(PACKAGE / 'datasets/allosteric/raw/AlloBench_processed.csv', sep='\t')
    mutations = pd.read_csv(SUPPLIED / 'AM/AM_processed.csv', sep='\t')
    previous = pd.read_csv(LEGACY / 'manifest.csv', keep_default_na=False).set_index('receptor_uniprot').to_dict('index')
    sequences = {identifier: group.sequence.mode().iloc[0] for identifier, group in raw.groupby('pdb_uniprot')}
    sequences.update(dict(zip(mutations.uniprot_id, mutations.wt_sequence)))
    items = [(identifier, expected, previous.get(identifier)) for identifier, expected in sorted(sequences.items())]
    rows = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for future in as_completed([executor.submit(prepare_structure, item) for item in items]):
            rows.append(future.result())
            print('structure', len(rows), '/', len(items), rows[-1]['uniprot'], rows[-1]['structure_status'], flush=True)
    structures = pd.DataFrame(rows).sort_values('uniprot')
    structures.to_csv(ROOT / 'structures_audit.csv', index=False)
    structures = structures[structures.structure_status.ne('failed')].copy()
    structures.to_csv(ROOT / 'manifest.csv', index=False)
    lookup = structures.set_index('uniprot').to_dict('index')
    processed_lookup = {(row.pdb_uniprot, str(row.allosteric_pdb).upper(), row.allosteric_site_residue): row._asdict() for row in processed.itertuples()}
    jobs, downloads = [], set()
    for index, record in enumerate(raw.to_dict('records')):
        identifier = record['pdb_uniprot']
        if identifier not in lookup:
            continue
        processed_record = processed_lookup.get((identifier, str(record['allosteric_pdb']).upper(), record['allosteric_site_residue']))
        jobs.append((index, record, lookup[identifier], processed_record))
        if processed_record is None:
            downloads.add(str(record['allosteric_pdb']).upper())
    with ThreadPoolExecutor(max_workers=8) as executor:
        audit = list(executor.map(source_download, sorted(downloads)))
    pd.DataFrame(audit).to_csv(ROOT / 'source_pdb_downloads.csv', index=False)
    mappings = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for result in executor.map(map_record, jobs):
            mappings.append(result)
    mappings = pd.DataFrame(mappings)
    mappings.to_csv(ROOT / 'allobench_record_mapping.csv', index=False)
    sites = []
    for identifier, group in mappings[mappings.status.eq('mapped')].groupby('uniprot'):
        positions = sorted({int(value) for values in group.positions_0 for value in values.split(';')})
        sites.extend(dict(uniprot=identifier, residue_index_0=position, label=1) for position in positions)
    pd.DataFrame(sites).to_csv(ROOT / 'allobench_positive_positions.csv', index=False)
    write_json(ROOT / 'preparation_summary.json', dict(requested_proteins=len(items), available_structures=len(structures), raw_allobench_proteins=430,
        mapped_allobench_proteins=int(mappings[mappings.status.eq('mapped')].uniprot.nunique()), mapped_records=int(mappings.status.eq('mapped').sum()), registered_records=len(raw), mutation_proteins=int(mutations.uniprot_id.nunique())))


if __name__ == '__main__':
    main()
