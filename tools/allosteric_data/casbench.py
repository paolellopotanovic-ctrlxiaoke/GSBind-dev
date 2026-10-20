import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile

import pandas as pd

from common import ROOT, fetch, write_json
from prepare import AMINO, align, observed


def sifts(identifier):
    path = ROOT/'raw/external/casbench_sifts'/(identifier+'.json')
    try:
        data = json.loads(fetch('https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/'+identifier,path,retries=2))
        return identifier,data[identifier]['UniProt']
    except Exception:
        return identifier,{}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-structures',action='store_true')
    args = parser.parse_args()
    archive_path = ROOT/'raw/external/casbench_compact.tgz'
    if not archive_path.exists():
        fetch('https://biokinet.belozersky.msu.ru/sites/default/data/casbench.tgz',archive_path)
    folder = ROOT/'raw/external/casbench_selected'
    folder.mkdir(exist_ok=True)
    registered = []
    with tarfile.open(archive_path) as archive:
        for entry in archive:
            parts = Path(entry.name).parts[-3:]
            if not entry.isfile() or len(parts)!=3 or not parts[0].startswith('cas'):
                continue
            if entry.name.endswith('/ALLOSTERIC_SITES.txt'):
                registered.append(dict(cas_entry=parts[0],source_pdb=parts[1],source_annotation='/'.join(parts)))
            if entry.name.endswith(('.pdb','/ALLOSTERIC_SITES.txt','/CATALYTIC_SITES.txt')):
                destination = folder/parts[0]/parts[1]/parts[2]
                if not destination.exists():
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(archive.extractfile(entry).read())
    registry = pd.DataFrame(registered).sort_values(['cas_entry','source_pdb'])
    representatives = registry.groupby('cas_entry',sort=True).head(1)
    registry['selected_representative'] = registry.source_annotation.isin(representatives.source_annotation)
    registry.to_csv(ROOT/'casbench_source_registry.csv',index=False)
    total_structures = len(registered)
    if not args.all_structures:
        registered = representatives.to_dict('records')
    pdbs = sorted({row['source_pdb'] for row in registered})
    with ThreadPoolExecutor(max_workers=10) as executor:
        mapping = dict(executor.map(sifts,pdbs))
    manifest = pd.read_csv(ROOT/'manifest_with_external.csv').set_index('uniprot')
    results = []
    for count,row in enumerate(registered,1):
        identifier = row['source_pdb']
        path = folder/row['cas_entry']/identifier/(identifier+'.pdb')
        allowed = set(mapping.get(identifier,{}))&set(manifest.index)
        result = dict(row)
        try:
            if not allowed:
                raise ValueError('No SIFTS target identity in frozen 504-input cohort')
            residues = observed(path)
            annotations = []
            for line in (folder/row['source_annotation']).read_text().splitlines():
                parts = line.split()
                if len(parts)==4:
                    annotations.append((parts[2],parts[3],parts[1].upper()))
            for target in sorted(allowed):
                sequence = manifest.loc[target,'sequence']
                chain_maps = {}
                for chain in {item[0] for item in annotations}:
                    keys = [key for key in residues if key[0]==chain]
                    query = ''.join(residues[key] for key in keys)
                    if not query:
                        continue
                    mapped,identity = align(sequence,query)
                    if identity>=.95:
                        chain_maps[chain] = {key:mapped[index] for index,key in enumerate(keys) if index in mapped and query[index]==sequence[mapped[index]]}
                positions = set()
                for chain,number,name in annotations:
                    key = (chain,number)
                    if name in AMINO and residues.get(key)==AMINO[name] and key in chain_maps.get(chain,{}):
                        positions.add(chain_maps[chain][key])
                if positions:
                    results.append(dict(result,uniprot=target,status='mapped',positions_0=';'.join(map(str,sorted(positions))),method='official_CASBench_modified_source_PDB_SIFTS_and_sequence_verified'))
            if not any(item.get('source_annotation')==row['source_annotation'] and item.get('status')=='mapped' for item in results[-len(allowed):]):
                raise ValueError('No coordinate/identity-valid allosteric annotation')
        except Exception as error:
            results.append(dict(result,status='failed',reason=str(error)))
        if count%100==0:
            print('CASBench',count,'/',len(registered),flush=True)
    result = pd.DataFrame(results)
    result.to_csv(ROOT/'casbench_record_mapping.csv',index=False)
    labels = []
    for target,group in result[result.status.eq('mapped')].groupby('uniprot'):
        positions = {int(value) for text in group.positions_0 for value in text.split(';')}
        labels.extend(dict(dataset='CASBench',uniprot=target,residue_index_0=position,label=1) for position in sorted(positions))
    positive = pd.DataFrame(labels)
    positive.to_csv(ROOT/'casbench_positive_positions.csv',index=False)
    features = pd.read_parquet(ROOT/'complete_features.parquet')
    current = features[features.uniprot.isin(positive.uniprot)].copy()
    pairs = set(zip(positive.uniprot,positive.residue_index_0))
    current['label'] = [int((row.uniprot,row.residue_index_0) in pairs) for row in current.itertuples()]
    current.to_parquet(ROOT/'tasks/CASBench.parquet',index=False)
    summary = json.loads((ROOT/'task_summary.json').read_text())
    summary['CASBench'] = dict(proteins=int(current.uniprot.nunique()),positions=len(current),positives=int(current.label.sum()))
    write_json(ROOT/'task_summary.json',summary)
    write_json(ROOT/'casbench_summary.json',dict(registered_enzyme_entries=int(registry.cas_entry.nunique()),registered_structures=total_structures,evaluated_source_structures=len(registered),representative_policy='lexicographically_first_PDB_per_CAS_entry_before_outcome_evaluation' if not args.all_structures else 'all_structures',
        mapped_enzyme_entries=int(result[result.status.eq('mapped')].cas_entry.nunique()),mapped_uniprot_proteins=int(current.uniprot.nunique()),evaluated_only_exact_SIFTS_UniProt_identity=True,
        other_entries_explicitly_audited_as_missing_frozen_input_or_mapping=True,not_claimed_identical_to_PASSer_anonymous_1049_ordering=True))


if __name__ == '__main__':
    main()
