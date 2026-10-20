from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from common import ROOT, fetch, write_json
from prepare import AMINO, align, map_record, observed, prepare_structure, source_download


def parse_description(text):
    sites = []
    for block in str(text).split(';'):
        match = re.match(r'\s*Chain\s+([^:]+):\s*(.*)', block)
        if not match:
            continue
        for residue in match[2].split(','):
            residue = residue.strip()
            amino = residue[:3]
            number = residue[3:]
            if amino in AMINO and re.fullmatch(r'-?\d+[A-Za-z]?', number):
                sites.append(match[1].strip() + '-' + amino + '-' + number)
    return str(sites)


def asbench_record(row, structure, path):
    identifier = row['Uniprot ID']
    output = dict(dataset='ASBench', uniprot=identifier, record_id=row['Allosteric Site'], source_pdb=row['PDB ID'])
    try:
        chain = str(row['Chain ID']).strip()
        residue_number = str(row['Residue ID (PDB)']).split('.')[0].strip()
        ligand_name = str(row['Modulator Name']).strip()
        ligand, protein = [], []
        for line in path.read_text().splitlines():
            if line.startswith('ENDMDL'):
                break
            if not line.startswith(('ATOM  ', 'HETATM')) or line[16] not in ' A' or line[76:78].strip() == 'H':
                continue
            coordinate = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            if line.startswith('HETATM') and line[17:20].strip() == ligand_name and line[21] == chain and line[22:26].strip() == residue_number:
                ligand.append(coordinate)
            if line.startswith('ATOM  ') and line[17:20].strip() in AMINO:
                protein.append(((line[21], line[22:27].strip()), coordinate))
        if not ligand:
            raise ValueError('Specified ASBench modulator absent')
        contacts = {protein[index][0] for index in np.flatnonzero(cdist(np.asarray([item[1] for item in protein]), np.asarray(ligand)).min(axis=1) <= 5.)}
        residues = observed(path)
        positions = set()
        for source_chain in {key[0] for key in contacts}:
            keys = [key for key in residues if key[0] == source_chain]
            query = ''.join(residues[key] for key in keys)
            mapping, identity = align(structure['sequence'], query)
            if identity < .95:
                continue
            for ordinal, key in enumerate(keys):
                if key in contacts and ordinal in mapping and query[ordinal] == structure['sequence'][mapping[ordinal]]:
                    positions.add(mapping[ordinal])
        if not positions:
            raise ValueError('No target protein ligand-contact positions mapped')
        output.update(status='mapped', positions_0=';'.join(map(str, sorted(positions))), method='annotated_ligand_5A_contact_source_to_AFDB')
    except Exception as error:
        output.update(status='failed', reason=type(error).__name__ + ': ' + str(error))
    return output


def main():
    base = ROOT / 'raw/external'
    asbench = pd.read_excel(base / 'asbench_statistics.xls')
    passer = pd.read_csv(base / 'passerrank_data_source_data_ASD_Release_201909_AS.txt', sep='\t').dropna(subset=['pdb_uniprot', 'allosteric_pdb', 'allosteric_site_residue'])
    ppi_records = json.loads((base / 'alloppi_records.json').read_text())['data']
    target_ids = set(asbench['Uniprot ID'].dropna()) | set(passer.pdb_uniprot) | {row['target_uniprot'] for row in ppi_records}
    manifest = pd.read_csv(ROOT / 'manifest.csv', keep_default_na=False)
    previous = manifest.set_index('uniprot').to_dict('index')
    missing = sorted(target_ids - set(previous))
    with ThreadPoolExecutor(max_workers=6) as executor:
        additions = list(executor.map(prepare_structure, [(identifier, '', None) for identifier in missing]))
    audit = pd.DataFrame(additions)
    audit.to_csv(ROOT / 'external_structures_audit.csv', index=False)
    successful = audit[audit.structure_status.ne('failed')].copy() if len(audit) else audit
    if len(successful):
        successful.to_csv(ROOT / 'external_new_manifest.csv', index=False)
        manifest = pd.concat([manifest, successful], ignore_index=True).drop_duplicates('uniprot')
    manifest.to_csv(ROOT / 'manifest_with_external.csv', index=False)
    lookup = manifest.set_index('uniprot').to_dict('index')
    complex_dir = base / 'asbench_complexes'
    complex_dir.mkdir(exist_ok=True)
    with tarfile.open(base / 'asbench_core.tar.gz') as archive:
        for entry in archive:
            if entry.isfile() and '/protein-modulator_complexes/' in entry.name and entry.name.endswith('.pdb'):
                destination = complex_dir / Path(entry.name).name
                if not destination.exists():
                    destination.write_bytes(archive.extractfile(entry).read())
    result = []
    for row in asbench.to_dict('records'):
        identifier = row['Uniprot ID']
        path = complex_dir / (str(row['Allosteric Site']) + '_' + str(row['PDB ID']) + '_complex.pdb')
        if identifier in lookup:
            result.append(asbench_record(row, lookup[identifier], path))
        else:
            result.append(dict(dataset='ASBench', uniprot=identifier, record_id=row['Allosteric Site'], status='failed', reason='AFDB_structure_unavailable'))
    converted = []
    for index, row in passer.iterrows():
        converted.append((index, dict(pdb_uniprot=row.pdb_uniprot, allosteric_pdb=row.allosteric_pdb, allosteric_site_residue=parse_description(row.allosteric_site_residue), pubmed_id=row.pubmed_id)))
    for row in ppi_records:
        path = base / 'alloppi_details' / (row['main_id']+'.json')
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        detail = data.get('choose_detail', {})
        converted.append(('ppi_'+row['main_id'], dict(pdb_uniprot=row['target_uniprot'], allosteric_pdb=detail.get('apo_pdb_id') or data.get('choose_apo_pdbid') or data['target_pdbid'],
            allosteric_site_residue=parse_description(detail.get('ref_allo_site_detail', '')), pubmed_id=detail.get('allo_pubmed_id', ''))))
    pdb_ids = {str(row['allosteric_pdb']).upper() for _, row in converted}
    with ThreadPoolExecutor(max_workers=8) as executor:
        downloads = list(executor.map(source_download, sorted(pdb_ids)))
    pd.DataFrame(downloads).to_csv(ROOT / 'external_pdb_downloads.csv', index=False)
    jobs = [(index, row, lookup[row['pdb_uniprot']], None) for index, row in converted if row['pdb_uniprot'] in lookup]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for record in executor.map(map_record, jobs):
            record['dataset'] = 'ASD_AlloPPI' if str(record['record_id']).startswith('ppi_') else 'PASSer_ASD_source'
            result.append(record)
    records = pd.DataFrame(result)
    records.to_csv(ROOT / 'external_record_mapping.csv', index=False)
    labels = []
    for (dataset, identifier), group in records[records.status.eq('mapped')].groupby(['dataset', 'uniprot']):
        positions = sorted({int(value) for text in group.positions_0 for value in text.split(';')})
        labels.extend(dict(dataset=dataset, uniprot=identifier, residue_index_0=position, label=1) for position in positions)
    pd.DataFrame(labels).to_csv(ROOT / 'external_positive_positions.csv', index=False)
    write_json(ROOT / 'external_preparation_summary.json', dict(registered_asbench_records=len(asbench), registered_passer_source_records=len(passer), registered_alloppi_records=len(ppi_records),
        requested_additional_structures=len(missing), available_additional_structures=len(successful), mapped=records[records.status.eq('mapped')].groupby('dataset').agg(records=('uniprot','size'), proteins=('uniprot','nunique')).to_dict('index'),
        passer_prepared_pocket_dataset='downloaded numeric pocket features/labels lack protein identity mapping; not misrepresented as residue benchmark; PASSer_ASD_source is an explicitly reconstructed source cohort'))


if __name__ == '__main__':
    main()
