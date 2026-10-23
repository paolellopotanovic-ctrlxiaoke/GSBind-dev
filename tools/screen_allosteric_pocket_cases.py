"""Descriptive full-protein top-k screening against individual source pockets."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases/pocket_screen'


def screen():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(ROOT / 'results/downstream/AlloBench/predictions.csv.gz')
    grouped = {name: group for name, group in predictions.groupby('uniprot')}
    mapping = pd.read_csv(ROOT / 'datasets/allosteric/allobench_record_mapping.csv')
    raw = pd.read_csv(ROOT / 'datasets/allosteric/raw/AlloBench.csv')
    rows = []
    for record in mapping.loc[mapping.status.eq('mapped')].itertuples():
        if record.uniprot not in grouped:
            continue
        frame = grouped[record.uniprot]
        positions = set(map(int, record.positions_0.split(';')))
        if len(positions) < 5:
            continue
        source = raw.iloc[int(record.record_id)]
        if source.pdb_uniprot != record.uniprot or source.allosteric_pdb.upper() != record.source_pdb.upper():
            raise ValueError('Source index mismatch')
        labels = frame.residue_index_0.isin(positions)
        if int(labels.sum()) != len(positions):
            continue
        row = dict(record_id=record.record_id, uniprot=record.uniprot, pdb=record.source_pdb,
                   gene=source.target_gene, ligand=source.modulator_alias,
                   ligand_name=source.modulator_name, ligand_chain=source.modulator_chain,
                   ligand_resi=source.modulator_resi, pubmed=source.pubmed_id,
                   title=source.ref_title, positions=len(frame), site_positions=len(positions),
                   background_prevalence=len(positions)/len(frame), site_overlap=source.site_overlap)
        for method, short in [('gsbind_hidden_probe', 'hidden'), ('structure_only', 'structure'),
                              ('gsbind_score_probe', 'score'), ('pesto_score_probe', 'pesto')]:
            precision, recall, thresholds = precision_recall_curve(labels, frame[method])
            row[short+'_source_pocket_AUPR'] = float(auc(recall, precision))
            ordered = frame.sort_values([method, 'residue_index_0'], ascending=[False, True])
            for fraction, suffix in [(.05, 'top5'), (.10, 'top10')]:
                count = math.ceil(len(frame)*fraction)
                chosen = set(ordered.head(count).residue_index_0)
                hits = len(chosen & positions)
                row[short+'_'+suffix+'_k'] = count
                row[short+'_'+suffix+'_hits'] = hits
                row[short+'_'+suffix+'_precision'] = hits/count
                row[short+'_'+suffix+'_recall'] = hits/len(positions)
        row['hidden_gain_over_structure'] = row['hidden_source_pocket_AUPR']-row['structure_source_pocket_AUPR']
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ['hidden_top5_precision', 'hidden_top5_recall', 'hidden_source_pocket_AUPR'], ascending=False)
    result.to_csv(DIRECTORY / 'ALL_SOURCE_POCKET_SCREEN.csv', index=False)
    result.drop_duplicates('uniprot').to_csv(DIRECTORY / 'BEST_SOURCE_PER_PROTEIN.csv', index=False)
    columns = ['uniprot', 'gene', 'pdb', 'ligand', 'positions', 'site_positions', 'hidden_source_pocket_AUPR',
               'hidden_top5_precision', 'hidden_top5_recall', 'hidden_top10_precision',
               'hidden_top10_recall', 'hidden_gain_over_structure']
    print(result.drop_duplicates('uniprot')[columns].head(30).to_string(index=False))
    print('SCREENED source records:',len(result),'proteins:',result.uniprot.nunique())


if __name__ == '__main__':
    screen()
