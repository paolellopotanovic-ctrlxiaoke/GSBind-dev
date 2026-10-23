"""Fixed full-protein top-k selections, including off-annotation predictions."""

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'figures/structural_cases'
METHODS = ['gsbind_score_probe', 'structure_only', 'gsbind_hidden_probe']


def prepare():
    predictions = pd.read_csv(DIRECTORY / 'ABL1_ALL_EVALUATED_PREDICTIONS.csv')
    mapping = pd.read_csv(DIRECTORY / 'ABL1_RESIDUE_MAPPING.csv')
    annotations = set(mapping.loc[mapping.site_5mo4, 'canonical_position'])
    assert annotations == set(predictions.loc[predictions.label.eq(1), 'residue_index_0'] + 1)
    rows, summary = [], []
    for fraction in [.05, .10]:
        count = math.ceil(len(predictions) * fraction)
        for method in METHODS:
            ordered = predictions.sort_values([method, 'residue_index_0'], ascending=[False, True])
            selected = set(ordered.head(count).residue_index_0 + 1)
            mapped_selected, mapped_hits, missed = 0, 0, 0
            for row in mapping.itertuples():
                high = row.canonical_position in selected
                if not row.identity_match:
                    category = 'sequence_mismatch'
                elif row.site_5mo4:
                    category = 'hit' if high else 'missed_annotation'
                else:
                    category = 'off_annotation_high' if high else 'background'
                mapped_selected += category in ['hit', 'off_annotation_high']
                mapped_hits += category == 'hit'
                missed += category == 'missed_annotation'
                rows.append(dict(fraction=fraction, top_k=count, method=method,
                                 pdb_resi=row.pdb_resi, canonical_position=row.canonical_position,
                                 site_5mo4=row.site_5mo4, selected=high, identity_match=row.identity_match,
                                 category=category))
            hits = len(selected & annotations)
            summary.append(dict(fraction=fraction, method=method, full_protein_positions=len(predictions),
                                top_k=count, annotated_positions=len(annotations), full_protein_hits=hits,
                                full_protein_precision=hits/count, full_protein_recall=hits/len(annotations),
                                mapped_evaluable_selected=mapped_selected, mapped_hits=mapped_hits,
                                mapped_off_annotation=mapped_selected-mapped_hits,
                                mapped_missed_annotation=missed,
                                mapped_precision=mapped_hits/mapped_selected if mapped_selected else None))
    pd.DataFrame(rows).to_csv(DIRECTORY / 'ABL1_TOPK_RESIDUES.csv', index=False)
    pd.DataFrame(summary).to_csv(DIRECTORY / 'ABL1_TOPK_SUMMARY.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    prepare()
