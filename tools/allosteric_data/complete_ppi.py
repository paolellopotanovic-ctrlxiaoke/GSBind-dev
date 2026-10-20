from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd

from common import ROOT, write_json
from external import parse_description
from prepare import map_record, source_download


def main():
    base = ROOT / 'raw/external'
    registry = {str(row['main_id']): row for row in json.loads((base / 'alloppi_records.json').read_text())['data']}
    manifest = pd.read_csv(ROOT / 'manifest_with_external.csv',keep_default_na=False).set_index('uniprot').to_dict('index')
    entries = []
    for path in sorted((base / 'alloppi_modulators').glob('*.json')):
        for detail in json.loads(path.read_text()).get('data',[]):
            parent = registry[str(detail['main_id'])]
            entries.append(dict(main_id=detail['main_id'],ppi_id=detail['ppi_id'],uniprot=parent['target_uniprot'],ppi_type=parent['ppi_type'],source_pdb=detail.get('allo_ppi_pdbid'),
                source_sites=detail.get('allo_allosite_resi'),source_pubmed=detail.get('allo_pubmed_id'),modulator=detail.get('modulator_serial')))
    pd.DataFrame(entries).to_csv(ROOT / 'alloppi_748_registry.csv',index=False)
    pdbs = {str(row['source_pdb']).upper() for row in entries if row['source_pdb']}
    with ThreadPoolExecutor(max_workers=8) as executor:
        downloads = list(executor.map(source_download,sorted(pdbs)))
    pd.DataFrame(downloads).to_csv(ROOT / 'alloppi_source_downloads.csv',index=False)
    jobs,failures = [],[]
    for row in entries:
        if row['uniprot'] not in manifest or not row['source_pdb'] or not row['source_sites']:
            failures.append(dict(dataset='ASD_AlloPPI',uniprot=row['uniprot'],record_id='ppi_'+str(row['ppi_id']),status='failed',reason='missing_AFDB_or_coordinate_site_annotation'))
            continue
        record = dict(pdb_uniprot=row['uniprot'],allosteric_pdb=row['source_pdb'],allosteric_site_residue=parse_description(row['source_sites']),pubmed_id=row['source_pubmed'])
        jobs.append(('ppi_'+str(row['ppi_id']),record,manifest[row['uniprot']],None))
    with ThreadPoolExecutor(max_workers=6) as executor:
        mapped = list(executor.map(map_record,jobs))
    records = pd.DataFrame(mapped+failures)
    records['dataset'] = 'ASD_AlloPPI'
    records.to_csv(ROOT / 'alloppi_748_mapping.csv',index=False)
    original = pd.read_csv(ROOT / 'external_record_mapping.csv')
    original = pd.concat([original[original.dataset.ne('ASD_AlloPPI')],records],ignore_index=True)
    original.to_csv(ROOT / 'external_record_mapping.csv',index=False)
    labels = []
    for (dataset,identifier),group in original[original.status.eq('mapped')].groupby(['dataset','uniprot']):
        positions = sorted({int(value) for text in group.positions_0 for value in text.split(';')})
        labels.extend(dict(dataset=dataset,uniprot=identifier,residue_index_0=position,label=1) for position in positions)
    pd.DataFrame(labels).to_csv(ROOT / 'external_positive_positions.csv',index=False)
    feature_path = ROOT / 'complete_features.parquet'
    if feature_path.exists():
        frame = pd.read_parquet(feature_path)
        positive = pd.DataFrame(labels)
        group = positive[positive.dataset.eq('ASD_AlloPPI')]
        pairs = set(zip(group.uniprot,group.residue_index_0))
        current = frame[frame.uniprot.isin(group.uniprot)].copy()
        current['label'] = [int((row.uniprot,row.residue_index_0) in pairs) for row in current.itertuples()]
        current.to_parquet(ROOT / 'tasks/ASD_AlloPPI.parquet',index=False)
        summary = json.loads((ROOT / 'task_summary.json').read_text())
        summary['ASD_AlloPPI'] = dict(proteins=int(current.uniprot.nunique()),positions=len(current),positives=int(current.label.sum()))
        write_json(ROOT / 'task_summary.json',summary)
    write_json(ROOT / 'alloppi_complete_summary.json',dict(registered_relationships=len(registry),registered_modulator_records=len(entries),mapped_records=int(records.status.eq('mapped').sum()),mapped_proteins=int(records[records.status.eq('mapped')].uniprot.nunique()),
        mapping_uses_actual_allosteric_complex=True,not_only_displayed_example=True))
    print('PPI',records.status.value_counts().to_dict(),records[records.status.eq('mapped')].uniprot.nunique(),flush=True)


if __name__ == '__main__':
    main()
