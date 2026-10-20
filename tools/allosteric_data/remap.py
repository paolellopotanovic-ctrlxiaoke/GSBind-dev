from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import PACKAGE, ROOT, write_json
from prepare import map_record


def main():
    raw = pd.read_csv(PACKAGE / 'datasets/allosteric/raw/AlloBench.csv')
    processed = pd.read_csv(PACKAGE / 'datasets/allosteric/raw/AlloBench_processed.csv', sep='\t')
    structures = pd.read_csv(ROOT / 'manifest.csv', keep_default_na=False).set_index('uniprot').to_dict('index')
    lookup = {(row.pdb_uniprot, str(row.allosteric_pdb).upper(), row.allosteric_site_residue): row._asdict() for row in processed.itertuples()}
    jobs, failures = [], []
    for index, row in enumerate(raw.to_dict('records')):
        if row['pdb_uniprot'] not in structures:
            failures.append(dict(record_id=index,uniprot=row['pdb_uniprot'],source_pdb=row['allosteric_pdb'],status='failed',reason='AFDB_structure_unavailable_or_sequence_mismatch'))
        else:
            jobs.append((index,row,structures[row['pdb_uniprot']],lookup.get((row['pdb_uniprot'],str(row['allosteric_pdb']).upper(),row['allosteric_site_residue']))))
    with ThreadPoolExecutor(max_workers=6) as executor:
        records = pd.DataFrame(list(executor.map(map_record,jobs))+failures)
    records.sort_values('record_id').to_csv(ROOT / 'allobench_record_mapping.csv',index=False)
    labelled = records[records.status.eq('mapped')]
    positions = []
    for identifier,group in labelled.groupby('uniprot'):
        sites = sorted({int(value) for values in group.positions_0 for value in values.split(';')})
        positions.extend(dict(uniprot=identifier,residue_index_0=site,label=1) for site in sites)
    pd.DataFrame(positions).to_csv(ROOT / 'allobench_positive_positions.csv',index=False)
    registry = raw.groupby('pdb_uniprot').size().rename('source_records').reset_index().rename(columns={'pdb_uniprot':'uniprot'})
    registry['structure_available'] = registry.uniprot.isin(structures)
    registry['mapped_records'] = registry.uniprot.map(labelled.groupby('uniprot').size()).fillna(0).astype(int)
    registry['mapped_positive_positions'] = registry.uniprot.map(pd.DataFrame(positions).groupby('uniprot').size()).fillna(0).astype(int)
    registry['status'] = registry.apply(lambda row:'eligible_mapped' if row.mapped_records else ('source_label_mapping_failed' if row.structure_available else 'AFDB_structure_unavailable_or_sequence_mismatch'),axis=1)
    registry.to_csv(ROOT / 'allobench_430_registry.csv',index=False)
    write_json(ROOT / 'allobench_mapping_summary.json',dict(registered_proteins=len(registry),registered_records=len(raw),mapped_proteins=int(labelled.uniprot.nunique()),mapped_records=len(labelled),excluded_proteins=registry[registry.status.ne('eligible_mapped')].to_dict('records')))
    print('mapped',labelled.uniprot.nunique(),len(labelled),flush=True)


if __name__ == '__main__':
    main()
